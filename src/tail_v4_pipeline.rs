use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray,
};
use chrono::{Datelike, NaiveDateTime};
use crossbeam::channel::{unbounded, Receiver, Sender};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use ndarray_npy::{read_npy, write_npy};
use numpy::IntoPyArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

use crate::factor_neutralization_io_optimized::IOOptimizedStyleData;
use crate::tail_v2_rank_roll_factor::rank_roll_block_f32_with_parallel;

const EPS: f64 = 1e-12;

fn format_hms(total_secs: u64) -> (u64, u64, u64) {
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    (hours, minutes, seconds)
}

#[derive(Clone)]
struct TailSelectionConfig {
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
}

#[derive(Clone)]
struct SharedInputs {
    dates: Arc<Vec<i32>>,
    stocks: Arc<Vec<String>>,
    windows: Arc<Vec<usize>>,
    fold: bool,
    min_valid: usize,
    backtest_start: i32,
    legacy_style_data: Arc<IOOptimizedStyleData>,
    ret_gap1: Arc<Array2<f32>>,
    ret_sum_gap1: Arc<Array2<f32>>,
    ret_gap5: Arc<Array2<f32>>,
    ret_sum_gap5: Arc<Array2<f32>>,
    restrict: Arc<Array2<f32>>,
    index_ret: Arc<Array1<f32>>,
    config: Arc<TailSelectionConfig>,
}

#[derive(Clone)]
struct TailTask {
    source_factor: String,
    factor_path: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct SummaryRowRecord {
    factor_name: String,
    stage: String,
    gap: i32,
    source_factor: String,
    #[serde(rename = "IC_mean")]
    ic_mean: f64,
    #[serde(rename = "IR")]
    ir: f64,
    annualized_return: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    date_size: i32,
    ratio_mean: f64,
    hedge_annualized_return: f64,
    hedge_annualized_sharpe_ratio: f64,
    hedge_max_drawdown: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct IcRecord {
    factor_name: String,
    #[serde(default)]
    dates: Vec<i32>,
    values: Vec<f32>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct TailTaskResult {
    source_factor: String,
    raw_summary_gap1: Vec<SummaryRowRecord>,
    raw_summary_gap5: Vec<SummaryRowRecord>,
    neu_summary_gap1: Vec<SummaryRowRecord>,
    neu_summary_gap5: Vec<SummaryRowRecord>,
    raw_ic_gap1: Vec<IcRecord>,
    raw_ic_gap5: Vec<IcRecord>,
    neu_ic_gap1: Vec<IcRecord>,
    neu_ic_gap5: Vec<IcRecord>,
    derived_factor_count: usize,
}

#[derive(Default)]
struct AggregatedCandidates {
    raw_summary_gap1: Vec<SummaryRowRecord>,
    raw_summary_gap5: Vec<SummaryRowRecord>,
    neu_summary_gap1: Vec<SummaryRowRecord>,
    neu_summary_gap5: Vec<SummaryRowRecord>,
    raw_ic_gap1: HashMap<String, IcRecord>,
    raw_ic_gap5: HashMap<String, IcRecord>,
    neu_ic_gap1: HashMap<String, IcRecord>,
    neu_ic_gap5: HashMap<String, IcRecord>,
}

impl AggregatedCandidates {
    fn merge_task(&mut self, task: TailTaskResult) {
        self.raw_summary_gap1.extend(task.raw_summary_gap1);
        self.raw_summary_gap5.extend(task.raw_summary_gap5);
        self.neu_summary_gap1.extend(task.neu_summary_gap1);
        self.neu_summary_gap5.extend(task.neu_summary_gap5);
        for record in task.raw_ic_gap1 {
            self.raw_ic_gap1
                .insert(record.factor_name.clone(), record);
        }
        for record in task.raw_ic_gap5 {
            self.raw_ic_gap5
                .insert(record.factor_name.clone(), record);
        }
        for record in task.neu_ic_gap1 {
            self.neu_ic_gap1
                .insert(record.factor_name.clone(), record);
        }
        for record in task.neu_ic_gap5 {
            self.neu_ic_gap5
                .insert(record.factor_name.clone(), record);
        }
    }
}

#[derive(Clone)]
struct LegacyBacktestResult {
    summary: [f64; 10],
    ic_dates: Vec<i32>,
    ic_values: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
struct TailV4FulltestTask {
    task_key: String,
    factor_name: String,
    stage: String,
    gap: i32,
}

#[derive(Serialize, Deserialize)]
struct TailV4FulltestWorkerConfig {
    ver: String,
    temp_root: String,
    source_dir: String,
    factor_names: Vec<String>,
    start_date: String,
    backtest_start_date: String,
    end_date: String,
    style_data_path: String,
    min_valid: usize,
    index_name: String,
}

#[derive(Serialize, Deserialize)]
struct TailV4FulltestWorkerResult {
    ok: bool,
    task_key: String,
    factor_name: String,
    stage: String,
    gap: i32,
    error: Option<String>,
}

fn nanmean_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn nanstd_population(values: &[f64]) -> f64 {
    let mean = nanmean_f64(values);
    if mean.is_nan() {
        return f64::NAN;
    }
    let mut sq_sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            let delta = value - mean;
            sq_sum += delta * delta;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        (sq_sum / count as f64).sqrt()
    }
}

fn sample_std(values: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for &value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count < 2 {
        return f64::NAN;
    }
    let mean = sum / count as f64;
    let mut sq_sum = 0.0;
    for &value in values {
        if !value.is_nan() {
            let delta = value - mean;
            sq_sum += delta * delta;
        }
    }
    (sq_sum / (count as f64 - 1.0)).sqrt()
}

fn annualized_sharpe_sample(values: &[f64]) -> f64 {
    let std = sample_std(values);
    if std.is_nan() || std <= EPS {
        return f64::NAN;
    }
    nanmean_f64(values) / std * 250.0_f64.sqrt()
}

fn max_drawdown_from_returns(values: &[f64]) -> f64 {
    let mut cumulative = 0.0;
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;
    for &value in values {
        if !value.is_nan() {
            cumulative += value;
        }
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = peak - cumulative;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    max_drawdown
}

fn average_ranks(values: &[f32]) -> Vec<f64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    indexed.sort_by(|lhs, rhs| {
        lhs.1
            .partial_cmp(&rhs.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });

    let mut ranks = vec![f64::NAN; values.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let value = indexed[start].1;
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for item in indexed.iter().take(end).skip(start) {
            ranks[item.0] = avg_rank;
        }
        start = end;
    }
    ranks
}

fn ordinal_ranks(values: &[f32]) -> Vec<i64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    indexed.sort_by(|lhs, rhs| {
        match (lhs.1.is_nan(), rhs.1.is_nan()) {
            (true, true) => lhs.0.cmp(&rhs.0),
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => lhs
                .1
                .partial_cmp(&rhs.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| lhs.0.cmp(&rhs.0)),
        }
    });
    let mut ranks = vec![0i64; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as i64;
    }
    ranks
}

fn legacy_spearman_correlation(x: &[f32], y: &[f32]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let xx = ordinal_ranks(x);
    let yy = ordinal_ranks(y);
    let n = x.len() as f64;
    let mut diff_sq_sum = 0.0;
    for idx in 0..x.len() {
        let diff = xx[idx] - yy[idx];
        diff_sq_sum += (diff * diff) as f64;
    }
    1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
}

fn count_open_symbols(restrict_row: &[f32]) -> usize {
    restrict_row
        .iter()
        .filter(|&&value| value.is_finite() && value == 0.0)
        .count()
}

fn has_enough_unique_values(
    factor: &ArrayView3<'_, f32>,
    slot_idx: usize,
    min_unique: usize,
) -> bool {
    let mut seen = HashSet::<u32>::new();
    for raw_idx in 0..factor.shape()[0].saturating_sub(1) {
        for stock_idx in 0..factor.shape()[1] {
            let value = factor[[raw_idx, stock_idx, slot_idx]];
            if value.is_finite() {
                seen.insert(value.to_bits());
                if seen.len() >= min_unique {
                    return true;
                }
            }
        }
    }
    false
}

fn legacy_backtest_single_factor(
    factor: &ArrayView3<'_, f32>,
    ret: &ArrayView2<'_, f32>,
    ret_sum: &ArrayView2<'_, f32>,
    restrict: &ArrayView2<'_, f32>,
    index: &ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
) -> LegacyBacktestResult {
    let default_result = LegacyBacktestResult {
        summary: [f64::NAN; 10],
        ic_dates: Vec::new(),
        ic_values: Vec::new(),
    };
    if factor.shape()[0] < 2 || gap == 0 || portf_num == 0 {
        return default_result;
    }
    if !has_enough_unique_values(factor, slot_idx, 10) {
        return default_result;
    }

    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let mut effective_raw_indices = Vec::<usize>::new();
    for raw_eff_idx in 1..n_dates {
        if dates[raw_eff_idx] <= backtest_start {
            continue;
        }
        let signal_row_idx = raw_eff_idx - 1;
        let mut all_nan = true;
        for stock_idx in 0..n_stocks {
            if factor[[signal_row_idx, stock_idx, slot_idx]].is_finite() {
                all_nan = false;
                break;
            }
        }
        if !all_nan {
            effective_raw_indices.push(raw_eff_idx);
        }
    }
    if effective_raw_indices.is_empty() {
        return default_result;
    }

    let date_size = effective_raw_indices.len();
    let mut group_returns = vec![vec![0.0_f64; date_size]; portf_num];
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut filtered_signal = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_future = Vec::<f32>::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }

        filtered_signal.clear();
        filtered_ret.clear();
        filtered_future.clear();
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                filtered_signal.push(signal_value);
                filtered_ret.push(ret_value);
                filtered_future.push(ret_sum[[raw_eff_idx, stock_idx]]);
            }
        }

        if (local_t + 1) % gap == 0 {
            let ic_value = legacy_spearman_correlation(&filtered_future, &filtered_signal);
            ic_dates.push(dates[raw_eff_idx]);
            ic_values_f64.push(ic_value);
            ic_values_f32.push(ic_value as f32);
        }

        let stocks_num = filtered_signal.len();
        if stocks_num < portf_num {
            continue;
        }

        let valid_symbol_num = count_open_symbols(restrict.row(raw_eff_idx - 1).as_slice().unwrap_or(&[]));
        if valid_symbol_num > 0 {
            ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
        }

        group_sums.fill(0.0);
        group_counts.fill(0);
        let ranks = average_ranks(&filtered_signal);
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += filtered_ret[idx] as f64;
            group_counts[bucket] += 1;
        }
        for bucket in 0..portf_num {
            group_returns[bucket][local_t] = if group_counts[bucket] == 0 {
                0.0
            } else {
                group_sums[bucket] / group_counts[bucket] as f64
            };
        }
    }

    let first_leg_cum = group_returns[0].iter().sum::<f64>();
    let last_leg_cum = group_returns[portf_num - 1].iter().sum::<f64>();
    let (long_idx, short_idx) = if first_leg_cum > last_leg_cum {
        (0usize, portf_num - 1)
    } else {
        (portf_num - 1, 0usize)
    };

    let mut ls_returns = vec![0.0_f64; date_size];
    let mut hedge_returns = vec![0.0_f64; date_size];
    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        let long_ret = group_returns[long_idx][local_t];
        let short_ret = group_returns[short_idx][local_t];
        ls_returns[local_t] = long_ret - short_ret;
        hedge_returns[local_t] = long_ret - index[raw_eff_idx] as f64;
    }

    let ic_mean = nanmean_f64(&ic_values_f64);
    let ic_std = nanstd_population(&ic_values_f64);
    let ir = if ic_std.is_nan() || ic_std <= EPS {
        f64::NAN
    } else {
        ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
    };
    let summary = [
        ic_mean,
        ir,
        nanmean_f64(&ls_returns) * 250.0,
        annualized_sharpe_sample(&ls_returns),
        max_drawdown_from_returns(&ls_returns),
        date_size as f64,
        nanmean_f64(&ratio_values),
        nanmean_f64(&hedge_returns) * 250.0,
        annualized_sharpe_sample(&hedge_returns),
        max_drawdown_from_returns(&hedge_returns),
    ];
    LegacyBacktestResult {
        summary,
        ic_dates,
        ic_values: ic_values_f32,
    }
}

fn legacy_backtest_block_f32(
    factor: ArrayView3<'_, f32>,
    ret: ArrayView2<'_, f32>,
    ret_sum: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    gap: usize,
    portf_num: usize,
) -> Result<Vec<LegacyBacktestResult>, String> {
    if gap == 0 {
        return Err("gap 必须大于 0".to_string());
    }
    if portf_num == 0 {
        return Err("portf_num 必须大于 0".to_string());
    }
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    if ret.shape() != [n_dates, n_stocks]
        || ret_sum.shape() != [n_dates, n_stocks]
        || restrict.shape() != [n_dates, n_stocks]
        || index.len() != n_dates
        || dates.len() != n_dates
    {
        return Err("legacy backtest 输入形状不匹配".to_string());
    }

    let n_factors = factor.shape()[2];
    let mut results = Vec::with_capacity(n_factors);
    for slot_idx in 0..n_factors {
        results.push(legacy_backtest_single_factor(
            &factor,
            &ret,
            &ret_sum,
            &restrict,
            &index,
            dates,
            backtest_start,
            slot_idx,
            gap,
            portf_num,
        ));
    }
    Ok(results)
}

fn factor_result_path(task_results_dir: &Path, source_factor: &str) -> PathBuf {
    task_results_dir.join(format!("{}.msgpack", source_factor))
}

fn write_task_result(path: &Path, result: &TailTaskResult) -> Result<(), String> {
    let tmp_path = path.with_extension("msgpack.tmp");
    let bytes = rmp_serde::to_vec_named(result).map_err(|e| format!("序列化任务结果失败: {}", e))?;
    fs::write(&tmp_path, bytes).map_err(|e| format!("写入任务结果失败: {}", e))?;
    fs::rename(&tmp_path, path).map_err(|e| format!("原子替换任务结果失败: {}", e))?;
    Ok(())
}

fn read_task_result(path: &Path) -> Result<TailTaskResult, String> {
    let bytes = fs::read(path).map_err(|e| format!("读取任务结果失败: {}", e))?;
    rmp_serde::from_slice::<TailTaskResult>(&bytes).map_err(|e| format!("解析任务结果失败: {}", e))
}

fn append_completed_source(completed_log_path: &Path, source_factor: &str) -> Result<(), String> {
    if let Some(parent) = completed_log_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建日志目录失败: {}", e))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(completed_log_path)
        .map_err(|e| format!("打开 completed log 失败: {}", e))?;
    writeln!(file, "{}", source_factor).map_err(|e| format!("写 completed log 失败: {}", e))?;
    Ok(())
}

fn sanitize_task_component(value: &str) -> String {
    value.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '_',
        })
        .collect()
}

fn fulltest_task_key(stage: &str, gap: i32, factor_name: &str) -> String {
    format!("{}_gap{}_{}", stage, gap, sanitize_task_component(factor_name))
}

fn fulltest_done_path(done_dir: &Path, task: &TailV4FulltestTask) -> PathBuf {
    done_dir.join(format!("{}.json", task.task_key))
}

fn write_fulltest_done(path: &Path, task: &TailV4FulltestTask) -> Result<(), String> {
    let tmp_path = path.with_extension("json.tmp");
    let payload = serde_json::json!({
        "task_key": task.task_key,
        "factor_name": task.factor_name,
        "stage": task.stage,
        "gap": task.gap,
    });
    fs::write(&tmp_path, serde_json::to_vec_pretty(&payload).map_err(|e| format!("序列化 fulltest done 失败: {}", e))?)
        .map_err(|e| format!("写入 fulltest done 失败: {}", e))?;
    fs::rename(&tmp_path, path).map_err(|e| format!("原子替换 fulltest done 失败: {}", e))?;
    Ok(())
}

fn create_tail_v4_fulltest_worker_script() -> String {
    r#"#!/usr/bin/env python3
import os
import sys

project_root = os.environ.get("TAIL_V4_PROJECT_ROOT", "/home/chenzongwei")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from design_whatever.tail_v4 import run_tail_v4_fulltest_worker

if __name__ == "__main__":
    run_tail_v4_fulltest_worker()
"#
    .to_string()
}

fn build_fulltest_tasks(
    gap5_selected: &[String],
    gap1_selected: &[String],
) -> Vec<TailV4FulltestTask> {
    let mut ordered_factors = Vec::<String>::new();
    let mut seen = HashSet::<String>::new();
    for factor_name in gap5_selected.iter().chain(gap1_selected.iter()) {
        if seen.insert(factor_name.clone()) {
            ordered_factors.push(factor_name.clone());
        }
    }

    let gap5_set = gap5_selected.iter().cloned().collect::<HashSet<_>>();
    let gap1_set = gap1_selected.iter().cloned().collect::<HashSet<_>>();
    let mut tasks = Vec::new();
    for factor_name in ordered_factors {
        if gap5_set.contains(&factor_name) {
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("raw", 5, &factor_name),
                factor_name: factor_name.clone(),
                stage: "raw".to_string(),
                gap: 5,
            });
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("neu", 5, &factor_name),
                factor_name: factor_name.clone(),
                stage: "neu".to_string(),
                gap: 5,
            });
        }
        if gap1_set.contains(&factor_name) {
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("raw", 1, &factor_name),
                factor_name: factor_name.clone(),
                stage: "raw".to_string(),
                gap: 1,
            });
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("neu", 1, &factor_name),
                factor_name: factor_name.clone(),
                stage: "neu".to_string(),
                gap: 1,
            });
        }
    }
    tasks
}

fn increment_fulltest_bucket(bucket_counts: &mut HashMap<String, usize>, stage: &str, gap: i32) {
    let key = match (stage, gap) {
        ("raw", 5) => "g5-raw",
        ("neu", 5) => "g5-neu",
        ("raw", 1) => "g1-raw",
        ("neu", 1) => "g1-neu",
        _ => return,
    };
    *bucket_counts.entry(key.to_string()).or_insert(0) += 1;
}

fn render_fulltest_progress(
    processed: usize,
    restored: usize,
    total: usize,
    started: Instant,
    bucket_counts: &HashMap<String, usize>,
) -> Result<(), String> {
    let finished = processed + restored;
    let progress = if total > 0 {
        finished as f64 / total as f64
    } else {
        1.0
    };
    let elapsed = started.elapsed();
    let estimated_total_secs = if progress > 0.0 {
        elapsed.as_secs_f64() / progress
    } else {
        elapsed.as_secs_f64()
    };
    let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() {
        (estimated_total_secs - elapsed.as_secs_f64()) as u64
    } else {
        0
    };
    let (elapsed_h, elapsed_m, elapsed_s) = format_hms(elapsed.as_secs());
    let (remaining_h, remaining_m, remaining_s) = format_hms(remaining_secs);
    let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let g5_raw = bucket_counts.get("g5-raw").copied().unwrap_or(0);
    let g5_neu = bucket_counts.get("g5-neu").copied().unwrap_or(0);
    let g1_raw = bucket_counts.get("g1-raw").copied().unwrap_or(0);
    let g1_neu = bucket_counts.get("g1-neu").copied().unwrap_or(0);
    print!(
        "\r[{}] Fulltest 进度 {}/{} ({:.1}%)，已恢复 {} 个，g5-raw {}，g5-neu {}，g1-raw {}，g1-neu {}，已用{}h{}m{}s，预计剩余{}h{}m{}s",
        current_time,
        finished,
        total,
        progress * 100.0,
        restored,
        g5_raw,
        g5_neu,
        g1_raw,
        g1_neu,
        elapsed_h,
        elapsed_m,
        elapsed_s,
        remaining_h,
        remaining_m,
        remaining_s,
    );
    std::io::stdout()
        .flush()
        .map_err(|e| format!("刷新 fulltest 进度失败: {}", e))?;
    Ok(())
}

fn run_tail_v4_fulltest_worker_process(
    worker_id: usize,
    task_queue: Arc<Mutex<VecDeque<TailV4FulltestTask>>>,
    result_sender: Sender<Result<TailV4FulltestWorkerResult, String>>,
    stop_flag: Arc<AtomicBool>,
    python_path: String,
    worker_config_json: String,
) {
    let script_content = create_tail_v4_fulltest_worker_script();
    let script_path = format!("/tmp/tail_v4_fulltest_worker_{}.py", worker_id);
    if let Err(err) = fs::write(&script_path, script_content) {
        let _ = result_sender.send(Err(format!("创建 fulltest worker 脚本失败: {}", err)));
        return;
    }

    let mut command = Command::new(&python_path);
    command
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("NUMEXPR_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .env("BLIS_NUM_THREADS", "1")
        .env("POLARS_MAX_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .env("TAIL_V4_PROJECT_ROOT", "/home/chenzongwei")
        .env("PYTHONPATH", "/home/chenzongwei")
        .env("TAIL_V4_FULLTEST_WORKER_CONFIG", worker_config_json);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            let _ = result_sender.send(Err(format!("启动 fulltest worker 失败: {}", err)));
            let _ = fs::remove_file(&script_path);
            return;
        }
    };

    let mut stdin = match child.stdin.take() {
        Some(stdin) => stdin,
        None => {
            let _ = result_sender.send(Err("获取 fulltest worker stdin 失败".to_string()));
            let _ = fs::remove_file(&script_path);
            return;
        }
    };
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = result_sender.send(Err("获取 fulltest worker stdout 失败".to_string()));
            let _ = fs::remove_file(&script_path);
            return;
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = result_sender.send(Err("获取 fulltest worker stderr 失败".to_string()));
            let _ = fs::remove_file(&script_path);
            return;
        }
    };

    let stderr_worker_id = worker_id;
    let _stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(text) => eprintln!("[Tail V4 Fulltest][worker {}] {}", stderr_worker_id, text),
                Err(_) => break,
            }
        }
    });

    loop {
        if stop_flag.load(AtomicOrdering::Relaxed) {
            break;
        }
        let next_task = {
            let mut guard = match task_queue.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    let _ = result_sender.send(Err("获取 fulltest 任务队列锁失败".to_string()));
                    stop_flag.store(true, AtomicOrdering::Relaxed);
                    break;
                }
            };
            guard.pop_front()
        };
        let Some(task) = next_task else {
            break;
        };

        let packed_data = match rmp_serde::to_vec_named(&task) {
            Ok(data) => data,
            Err(err) => {
                let _ = result_sender.send(Err(format!("序列化 fulltest 任务失败: {}", err)));
                stop_flag.store(true, AtomicOrdering::Relaxed);
                break;
            }
        };
        let length_bytes = (packed_data.len() as u32).to_le_bytes();
        if stdin.write_all(&length_bytes).is_err()
            || stdin.write_all(&packed_data).is_err()
            || stdin.flush().is_err()
        {
            let _ = result_sender.send(Err(format!(
                "发送 fulltest 任务失败: {}",
                task.task_key
            )));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }

        let mut len_buf = [0u8; 4];
        if stdout.read_exact(&mut len_buf).is_err() {
            let _ = result_sender.send(Err(format!(
                "读取 fulltest worker 结果长度失败: {}",
                task.task_key
            )));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
        let result_len = u32::from_le_bytes(len_buf) as usize;
        let mut result_data = vec![0u8; result_len];
        if stdout.read_exact(&mut result_data).is_err() {
            let _ = result_sender.send(Err(format!(
                "读取 fulltest worker 结果失败: {}",
                task.task_key
            )));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
        let result = match rmp_serde::from_slice::<TailV4FulltestWorkerResult>(&result_data) {
            Ok(result) => result,
            Err(err) => {
                let _ = result_sender.send(Err(format!("解析 fulltest worker 结果失败: {}", err)));
                stop_flag.store(true, AtomicOrdering::Relaxed);
                break;
            }
        };
        if !result.ok {
            let _ = result_sender.send(Err(format!(
                "fulltest 任务 {} 失败: {}",
                result.task_key,
                result.error.unwrap_or_else(|| "未知错误".to_string())
            )));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
        if result_sender.send(Ok(result)).is_err() {
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
    }

    let _ = stdin.write_all(&[0u8; 4]);
    let _ = stdin.flush();
    let _ = child.wait();
    let _ = fs::remove_file(&script_path);
}

fn timestamp_ns_to_date_key(value: i64) -> Result<i32, String> {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    let dt = NaiveDateTime::from_timestamp_opt(secs, nanos)
        .ok_or_else(|| format!("无效 timestamp 纳秒值: {}", value))?;
    Ok((dt.year() * 10000 + dt.month() as i32 * 100 + dt.day() as i32) as i32)
}

fn extract_date_keys(batch: &arrow::record_batch::RecordBatch, date_col_idx: usize) -> Result<Vec<i32>, String> {
    let date_column = batch.column(date_col_idx);
    if let Some(array) = date_column.as_any().downcast_ref::<Int32Array>() {
        return Ok((0..array.len()).map(|idx| array.value(idx)).collect());
    }
    if let Some(array) = date_column.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..array.len()).map(|idx| array.value(idx) as i32).collect());
    }
    if let Some(array) = date_column.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return (0..array.len())
            .map(|idx| timestamp_ns_to_date_key(array.value(idx)))
            .collect();
    }
    if let Some(array) = date_column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return (0..array.len())
            .map(|idx| timestamp_ns_to_date_key(array.value(idx) * 1_000))
            .collect();
    }
    if let Some(array) = date_column.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return (0..array.len())
            .map(|idx| timestamp_ns_to_date_key(array.value(idx) * 1_000_000))
            .collect();
    }
    Err("不支持的 date 列类型".to_string())
}

fn load_factor_to_template(
    factor_path: &str,
    template_dates: &[i32],
    template_stocks: &[String],
) -> Result<Array2<f32>, String> {
    let file = File::open(factor_path).map_err(|e| format!("打开因子文件失败 {}: {}", factor_path, e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("创建 parquet 读取器失败 {}: {}", factor_path, e))?;
    let schema = builder.schema();
    let mut date_col_idx = None;
    let stock_pos_map: HashMap<&str, usize> = template_stocks
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect();
    let mut stock_cols = Vec::<(usize, usize)>::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let name = field.name();
        if name == "date" {
            date_col_idx = Some(col_idx);
        } else if let Some(&stock_pos) = stock_pos_map.get(name.as_str()) {
            stock_cols.push((col_idx, stock_pos));
        }
    }
    let date_col_idx = date_col_idx.ok_or_else(|| format!("因子文件缺少 date 列: {}", factor_path))?;
    let date_pos_map: HashMap<i32, usize> = template_dates
        .iter()
        .enumerate()
        .map(|(idx, value)| (*value, idx))
        .collect();
    let reader = builder
        .with_batch_size(8192)
        .build()
        .map_err(|e| format!("构建 parquet 批读取器失败 {}: {}", factor_path, e))?;

    let mut output = Array2::<f32>::from_elem((template_dates.len(), template_stocks.len()), f32::NAN);
    for batch_result in reader {
        let batch = batch_result.map_err(|e| format!("读取 parquet batch 失败 {}: {}", factor_path, e))?;
        let date_keys = extract_date_keys(&batch, date_col_idx)?;
        for (row_idx, date_key) in date_keys.iter().enumerate() {
            let Some(&date_pos) = date_pos_map.get(date_key) else {
                continue;
            };
            for &(col_idx, stock_pos) in &stock_cols {
                let array = batch.column(col_idx);
                let value = if let Some(col) = array.as_any().downcast_ref::<Float64Array>() {
                    if col.is_null(row_idx) {
                        f32::NAN
                    } else {
                        col.value(row_idx) as f32
                    }
                } else if let Some(col) = array.as_any().downcast_ref::<Float32Array>() {
                    if col.is_null(row_idx) {
                        f32::NAN
                    } else {
                        col.value(row_idx)
                    }
                } else {
                    return Err(format!("股票列类型不是 float: {} / col_idx={}", factor_path, col_idx));
                };
                output[[date_pos, stock_pos]] = if value.is_finite() { value } else { f32::NAN };
            }
        }
    }
    Ok(output)
}

fn build_fold_values(raw_values: &Array2<f32>) -> Array2<f32> {
    let n_dates = raw_values.nrows();
    let n_stocks = raw_values.ncols();
    let mut folded = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    for date_idx in 0..n_dates {
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for stock_idx in 0..n_stocks {
            let value = raw_values[[date_idx, stock_idx]];
            if value.is_finite() {
                sum += value as f64;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        let mean = (sum / count as f64) as f32;
        for stock_idx in 0..n_stocks {
            let value = raw_values[[date_idx, stock_idx]];
            if value.is_finite() {
                folded[[date_idx, stock_idx]] = (value - mean).abs();
            }
        }
    }
    folded
}

fn derived_names_for_variant(source_factor: &str, windows: &[usize]) -> Vec<String> {
    let mut names = vec![format!("{}_smooth_1", source_factor)];
    for &window in windows {
        names.push(format!("{}_mean_smooth_{}", source_factor, window));
        names.push(format!("{}_max_smooth_{}", source_factor, window));
        names.push(format!("{}_min_smooth_{}", source_factor, window));
        names.push(format!("{}_std_smooth_{}", source_factor, window));
    }
    names
}

fn legacy_rank_values(values: &[f64]) -> Vec<f64> {
    let mut indexed_values = Vec::with_capacity(values.len());
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_nan() {
            indexed_values.push((idx, value));
        }
    }
    indexed_values
        .sort_unstable_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![f64::NAN; values.len()];
    for (rank, &(original_idx, _)) in indexed_values.iter().enumerate() {
        ranks[original_idx] = (rank + 1) as f64;
    }
    ranks
}

fn neutralize_block_legacy_exact(
    legacy_style_data: &IOOptimizedStyleData,
    factor: ArrayView3<'_, f32>,
    dates: &[i32],
    stocks: &[String],
    rank_before: bool,
    min_valid: usize,
) -> Result<Array3<f32>, String> {
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let n_factors = factor.shape()[2];
    if dates.len() != n_dates || stocks.len() != n_stocks {
        return Err("legacy neutralize exact 输入形状不匹配".to_string());
    }

    let mut all_style_stocks = std::collections::HashSet::new();
    for day_data in legacy_style_data.data_by_date.values() {
        for stock in day_data.stocks.iter() {
            all_style_stocks.insert(stock.clone());
        }
    }

    let mut ordered_factor_stocks: Vec<(usize, String)> = stocks
        .iter()
        .enumerate()
        .filter_map(|(stock_idx, stock)| {
            let stock_code = stock.get(..6).unwrap_or(stock.as_str()).to_string();
            if all_style_stocks.contains(&stock_code) {
                Some((stock_idx, stock_code))
            } else {
                None
            }
        })
        .collect();
    ordered_factor_stocks.sort_unstable_by(|lhs, rhs| lhs.1.cmp(&rhs.1));

    let mut output = Array3::<f32>::from_elem((n_dates, n_stocks, n_factors), f32::NAN);
    for date_idx in 0..n_dates {
        let date_key = dates[date_idx] as i64;
        let Some(day_data) = legacy_style_data.data_by_date.get(&date_key) else {
            continue;
        };
        let Some(regression_matrix) = &day_data.regression_matrix else {
            continue;
        };
        let mut template_style_positions = vec![None; n_stocks];
        for (stock_idx, stock) in stocks.iter().enumerate() {
            let stock_code = stock.get(..6).unwrap_or(stock.as_str());
            if let Some(&style_idx) = day_data.stock_index_map.get(stock_code) {
                template_style_positions[stock_idx] = Some(style_idx);
            }
        }

        let mut daily_factor_values = Vec::<f64>::with_capacity(n_stocks);
        let mut valid_stock_indices = Vec::<usize>::with_capacity(n_stocks);
        let mut valid_style_indices = Vec::<usize>::with_capacity(n_stocks);
        let mut beta_values = vec![0.0_f64; 12];

        for factor_idx in 0..n_factors {
            daily_factor_values.clear();
            valid_stock_indices.clear();
            valid_style_indices.clear();
            beta_values.fill(0.0);

            for &(stock_idx, _) in ordered_factor_stocks.iter() {
                let Some(style_idx) = template_style_positions[stock_idx] else {
                    continue;
                };
                let value = factor[[date_idx, stock_idx, factor_idx]];
                if value.is_finite() {
                    daily_factor_values.push(value as f64);
                    valid_stock_indices.push(stock_idx);
                    valid_style_indices.push(style_idx);
                }
            }
            if daily_factor_values.len() < min_valid {
                continue;
            }
            let ranked_values = if rank_before {
                legacy_rank_values(&daily_factor_values)
            } else {
                daily_factor_values.clone()
            };

            for (col_idx, &style_idx) in valid_style_indices.iter().enumerate() {
                let y_value = ranked_values[col_idx];
                for feature_idx in 0..12 {
                    beta_values[feature_idx] += regression_matrix[(feature_idx, style_idx)] * y_value;
                }
            }

            for (row_idx, &stock_idx) in valid_stock_indices.iter().enumerate() {
                let style_idx = valid_style_indices[row_idx];
                let mut predicted = 0.0_f64;
                for feature_idx in 0..12 {
                    predicted += day_data.style_matrix[(style_idx, feature_idx)] * beta_values[feature_idx];
                }
                output[[date_idx, stock_idx, factor_idx]] = (ranked_values[row_idx] - predicted) as f32;
            }
        }
    }
    Ok(output)
}

#[pyclass]
pub struct TailV4LegacyStyleData {
    style_data: Arc<IOOptimizedStyleData>,
}

#[pymethods]
impl TailV4LegacyStyleData {
    #[new]
    fn new(style_data_path: String) -> PyResult<Self> {
        let style_data = IOOptimizedStyleData::load_from_parquet_io_optimized(&style_data_path)?;
        Ok(Self {
            style_data: Arc::new(style_data),
        })
    }

    #[pyo3(signature = (dates, stocks, factor_block, rank_before=true, min_valid=12))]
    fn neutralize_block_exact<'py>(
        &self,
        py: Python<'py>,
        dates: Vec<i32>,
        stocks: Vec<String>,
        factor_block: numpy::PyReadonlyArray3<'py, f32>,
        rank_before: bool,
        min_valid: usize,
    ) -> PyResult<Py<numpy::PyArray3<f32>>> {
        let factor = factor_block.as_array();
        let output = py
            .allow_threads(|| {
                neutralize_block_legacy_exact(
                    self.style_data.as_ref(),
                    factor,
                    &dates,
                    &stocks,
                    rank_before,
                    min_valid,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(output.into_pyarray(py).to_owned())
    }
}

#[pyfunction]
#[pyo3(signature = (style_data_path, dates, stocks, factor_block, rank_before=true, min_valid=12))]
pub fn tail_v4_neutralize_block_exact<'py>(
    py: Python<'py>,
    style_data_path: String,
    dates: Vec<i32>,
    stocks: Vec<String>,
    factor_block: numpy::PyReadonlyArray3<'py, f32>,
    rank_before: bool,
    min_valid: usize,
) -> PyResult<Py<numpy::PyArray3<f32>>> {
    let factor = factor_block.as_array();
    let style_data = IOOptimizedStyleData::load_from_parquet_io_optimized(&style_data_path)?;
    let output = py
        .allow_threads(|| neutralize_block_legacy_exact(&style_data, factor, &dates, &stocks, rank_before, min_valid))
        .map_err(PyRuntimeError::new_err)?;
    Ok(output.into_pyarray(py).to_owned())
}

fn summary_from_row(
    factor_name: &str,
    stage: &str,
    gap: i32,
    source_factor: &str,
    values: &[f64],
) -> SummaryRowRecord {
    SummaryRowRecord {
        factor_name: factor_name.to_string(),
        stage: stage.to_string(),
        gap,
        source_factor: source_factor.to_string(),
        ic_mean: values[0],
        ir: values[1],
        annualized_return: values[2],
        sharpe_ratio: values[3],
        max_drawdown: values[4],
        date_size: values[5] as i32,
        ratio_mean: values[6],
        hedge_annualized_return: values[7],
        hedge_annualized_sharpe_ratio: values[8],
        hedge_max_drawdown: values[9],
    }
}

fn qualify_raw(summary: &SummaryRowRecord, gap: usize, cfg: &TailSelectionConfig) -> bool {
    if summary.ratio_mean < cfg.cover_rate {
        return false;
    }
    match gap {
        1 => summary.hedge_annualized_return >= cfg.ret_point_gap1 || summary.ic_mean.abs() >= cfg.ic_point_gap1,
        5 => summary.hedge_annualized_return >= cfg.ret_point_gap5 || summary.ic_mean.abs() >= cfg.ic_point_gap5,
        _ => false,
    }
}

fn qualify_neu(summary: &SummaryRowRecord, gap: usize, cfg: &TailSelectionConfig) -> bool {
    if summary.ratio_mean < cfg.cover_rate {
        return false;
    }
    let (ret_point, ic_point, ic_more) = match gap {
        1 => (cfg.ret_point_neu_gap1, cfg.ic_point_neu_gap1, cfg.ic_more_important_gap1),
        5 => (cfg.ret_point_neu_gap5, cfg.ic_point_neu_gap5, cfg.ic_more_important_gap5),
        _ => return false,
    };
    let mut qualifies =
        summary.hedge_annualized_return >= ret_point || summary.ic_mean.abs() >= ic_point;
    if let Some(ic_more_value) = ic_more {
        qualifies = qualifies
            || (summary.hedge_annualized_return >= ret_point && summary.ic_mean.abs() >= ic_more_value);
    }
    qualifies
}

fn process_task(task: &TailTask, shared: &SharedInputs) -> Result<TailTaskResult, String> {
    let raw_values = load_factor_to_template(&task.factor_path, shared.dates.as_slice(), shared.stocks.as_slice())?;
    let mut variants = vec![(task.source_factor.clone(), raw_values)];
    if shared.fold {
        let folded = build_fold_values(&variants[0].1);
        variants.push((format!("{}_fold", task.source_factor), folded));
    }

    let mut result = TailTaskResult {
        source_factor: task.source_factor.clone(),
        ..TailTaskResult::default()
    };

    for (variant_name, variant_values) in variants {
        let rolled_block = rank_roll_block_f32_with_parallel(&variant_values, shared.windows.as_slice(), false)?;
        let derived_names = derived_names_for_variant(&variant_name, shared.windows.as_slice());
        result.derived_factor_count += derived_names.len();

        let raw_gap1_results = legacy_backtest_block_f32(
            rolled_block.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            1,
            10,
        )?;
        let raw_gap5_results = legacy_backtest_block_f32(
            rolled_block.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            5,
            10,
        )?;

        let neutralized = neutralize_block_legacy_exact(
            shared.legacy_style_data.as_ref(),
            rolled_block.view(),
            shared.dates.as_slice(),
            shared.stocks.as_slice(),
            true,
            shared.min_valid,
        )?;
        let neu_gap1_results = legacy_backtest_block_f32(
            neutralized.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            1,
            10,
        )?;
        let neu_gap5_results = legacy_backtest_block_f32(
            neutralized.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            5,
            10,
        )?;

        for (slot_idx, derived_name) in derived_names.iter().enumerate() {
            let raw_gap1_row = summary_from_row(
                derived_name,
                "rolled",
                1,
                &variant_name,
                &raw_gap1_results[slot_idx].summary,
            );
            let raw_gap5_row = summary_from_row(
                derived_name,
                "rolled",
                5,
                &variant_name,
                &raw_gap5_results[slot_idx].summary,
            );
            let neu_gap1_row = summary_from_row(
                derived_name,
                "neu",
                1,
                &variant_name,
                &neu_gap1_results[slot_idx].summary,
            );
            let neu_gap5_row = summary_from_row(
                derived_name,
                "neu",
                5,
                &variant_name,
                &neu_gap5_results[slot_idx].summary,
            );

            let raw_gap1_keep = qualify_raw(&raw_gap1_row, 1, &shared.config);
            let raw_gap5_keep = qualify_raw(&raw_gap5_row, 5, &shared.config);
            let neu_gap1_keep = qualify_neu(&neu_gap1_row, 1, &shared.config);
            let neu_gap5_keep = qualify_neu(&neu_gap5_row, 5, &shared.config);

            if raw_gap1_keep {
                result.raw_summary_gap1.push(raw_gap1_row.clone());
                result.raw_ic_gap1.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: raw_gap1_results[slot_idx].ic_dates.clone(),
                    values: raw_gap1_results[slot_idx].ic_values.clone(),
                });
            }
            if raw_gap5_keep {
                result.raw_summary_gap5.push(raw_gap5_row.clone());
                result.raw_ic_gap5.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: raw_gap5_results[slot_idx].ic_dates.clone(),
                    values: raw_gap5_results[slot_idx].ic_values.clone(),
                });
            }
            if neu_gap1_keep {
                result.neu_summary_gap1.push(neu_gap1_row.clone());
            }
            if neu_gap5_keep {
                result.neu_summary_gap5.push(neu_gap5_row.clone());
            }
            if raw_gap1_keep || neu_gap1_keep {
                result.neu_ic_gap1.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: neu_gap1_results[slot_idx].ic_dates.clone(),
                    values: neu_gap1_results[slot_idx].ic_values.clone(),
                });
            }
            if raw_gap5_keep || neu_gap5_keep {
                result.neu_ic_gap5.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: neu_gap5_results[slot_idx].ic_dates.clone(),
                    values: neu_gap5_results[slot_idx].ic_values.clone(),
                });
            }
        }
    }

    Ok(result)
}

fn write_summary_json(path: &Path, rows: &[SummaryRowRecord]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建 metrics 目录失败: {}", e))?;
    }
    let file = File::create(path).map_err(|e| format!("创建 summary json 失败: {}", e))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, rows).map_err(|e| format!("写 summary json 失败: {}", e))
}

fn write_ic_outputs(
    matrix_path: &Path,
    names_path: &Path,
    dates_path: &Path,
    store: &HashMap<String, IcRecord>,
) -> Result<(), String> {
    if let Some(parent) = matrix_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建 ic 目录失败: {}", e))?;
    }
    let factor_names = {
        let mut names = store.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    };
    let all_dates = {
        let mut date_set = BTreeSet::<i32>::new();
        for record in store.values() {
            for &date in &record.dates {
                date_set.insert(date);
            }
        }
        date_set.into_iter().collect::<Vec<_>>()
    };
    let date_positions = all_dates
        .iter()
        .enumerate()
        .map(|(idx, date)| (*date, idx))
        .collect::<HashMap<_, _>>();
    let matrix = if factor_names.is_empty() {
        Array2::<f32>::from_shape_vec((all_dates.len(), 0), Vec::new())
            .map_err(|e| format!("构造空 ic 矩阵失败: {}", e))?
    } else {
        let mut data = vec![f32::NAN; all_dates.len() * factor_names.len()];
        for (col_idx, name) in factor_names.iter().enumerate() {
            let record = store
                .get(name)
                .ok_or_else(|| format!("缺少候选 IC 数据: {}", name))?;
            for (&date, &value) in record.dates.iter().zip(record.values.iter()) {
                if let Some(&row_idx) = date_positions.get(&date) {
                    data[row_idx * factor_names.len() + col_idx] = value;
                }
            }
        }
        Array2::<f32>::from_shape_vec((all_dates.len(), factor_names.len()), data)
            .map_err(|e| format!("构造 ic 矩阵失败: {}", e))?
    };
    write_npy(matrix_path, &matrix).map_err(|e| format!("写 ic npy 失败: {}", e))?;
    let file = File::create(names_path).map_err(|e| format!("创建 ic names json 失败: {}", e))?;
    serde_json::to_writer(BufWriter::new(file), &factor_names)
        .map_err(|e| format!("写 ic names json 失败: {}", e))?;
    let dates_array = Array1::<i32>::from_vec(all_dates);
    write_npy(dates_path, &dates_array).map_err(|e| format!("写 ic dates npy 失败: {}", e))?;
    Ok(())
}

fn write_aggregated_outputs(
    cache_root: &Path,
    aggregated: &AggregatedCandidates,
) -> Result<(), String> {
    let metrics_dir = cache_root.join("metrics");
    let ic_dir = cache_root.join("ic_ts");
    write_summary_json(
        &metrics_dir.join("summary_rolled_gap1_candidates.json"),
        &aggregated.raw_summary_gap1,
    )?;
    write_summary_json(
        &metrics_dir.join("summary_rolled_gap5_candidates.json"),
        &aggregated.raw_summary_gap5,
    )?;
    write_summary_json(
        &metrics_dir.join("summary_neu_gap1_candidates.json"),
        &aggregated.neu_summary_gap1,
    )?;
    write_summary_json(
        &metrics_dir.join("summary_neu_gap5_candidates.json"),
        &aggregated.neu_summary_gap5,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_rolled_gap1.npy"),
        &ic_dir.join("ic_rolled_gap1_names.json"),
        &ic_dir.join("ic_rolled_gap1_dates.npy"),
        &aggregated.raw_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_rolled_gap5.npy"),
        &ic_dir.join("ic_rolled_gap5_names.json"),
        &ic_dir.join("ic_rolled_gap5_dates.npy"),
        &aggregated.raw_ic_gap5,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap1.npy"),
        &ic_dir.join("ic_neu_gap1_names.json"),
        &ic_dir.join("ic_neu_gap1_dates.npy"),
        &aggregated.neu_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap5.npy"),
        &ic_dir.join("ic_neu_gap5_names.json"),
        &ic_dir.join("ic_neu_gap5_dates.npy"),
        &aggregated.neu_ic_gap5,
    )?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
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
    cover_rate=0.97,
    ret_point_neu_gap5=0.055,
    ret_point_neu_gap1=0.08,
    ic_point_neu_gap5=0.01,
    ic_point_neu_gap1=0.006,
    ret_point_gap5=0.1,
    ret_point_gap1=0.13,
    ic_point_gap5=0.03,
    ic_point_gap1=0.02,
    ic_more_important_gap5=0.01,
    ic_more_important_gap1=0.006,
))]
pub fn tail_v4_run_candidates<'py>(
    py: Python<'py>,
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
) -> PyResult<PyObject> {
    if factor_names.len() != factor_paths.len() {
        return Err(PyValueError::new_err("factor_names 和 factor_paths 长度必须一致"));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs 必须大于 0"));
    }
    if (ic_more_important_gap5.is_some()) != (ic_more_important_gap1.is_some()) {
        return Err(PyValueError::new_err(
            "ic_more_important_gap5 和 ic_more_important_gap1 必须同时有值或同时为 None",
        ));
    }

    let output = py.allow_threads(|| -> Result<(usize, usize, HashMap<String, usize>), String> {
        let started = Instant::now();
        let cache_root_path = PathBuf::from(&cache_root);
        let task_results_dir = cache_root_path.join("task_results");
        let logs_dir = cache_root_path.join("logs");
        let completed_log_path = logs_dir.join("completed_sources.txt");
        fs::create_dir_all(&task_results_dir).map_err(|e| format!("创建 task_results 目录失败: {}", e))?;
        fs::create_dir_all(&logs_dir).map_err(|e| format!("创建 logs 目录失败: {}", e))?;

        let shared = SharedInputs {
            dates: Arc::new(dates),
            stocks: Arc::new(stocks),
            windows: Arc::new(windows),
            fold,
            min_valid,
            backtest_start,
            legacy_style_data: Arc::new(
                IOOptimizedStyleData::load_from_parquet_io_optimized(&style_data_path)
                    .map_err(|e| e.to_string())?
            ),
            ret_gap1: Arc::new(read_npy(&ret_gap1_path).map_err(|e| format!("读取 ret_gap1.npy 失败: {}", e))?),
            ret_sum_gap1: Arc::new(read_npy(&ret_sum_gap1_path).map_err(|e| format!("读取 ret_sum_gap1.npy 失败: {}", e))?),
            ret_gap5: Arc::new(read_npy(&ret_gap5_path).map_err(|e| format!("读取 ret_gap5.npy 失败: {}", e))?),
            ret_sum_gap5: Arc::new(read_npy(&ret_sum_gap5_path).map_err(|e| format!("读取 ret_sum_gap5.npy 失败: {}", e))?),
            restrict: Arc::new(read_npy(&restrict_path).map_err(|e| format!("读取 restrict.npy 失败: {}", e))?),
            index_ret: Arc::new(read_npy(&index_ret_path).map_err(|e| format!("读取 index_ret.npy 失败: {}", e))?),
            config: Arc::new(TailSelectionConfig {
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
            }),
        };

        let mut aggregated = AggregatedCandidates::default();
        let mut completed_sources = HashSet::<String>::new();
        let mut restored_sources = 0usize;
        for (source_factor, factor_path) in factor_names.iter().zip(factor_paths.iter()) {
            let result_path = factor_result_path(&task_results_dir, source_factor);
            if result_path.exists() {
                if let Ok(task_result) = read_task_result(&result_path) {
                    aggregated.merge_task(task_result);
                    completed_sources.insert(source_factor.clone());
                    restored_sources += 1;
                    continue;
                }
            }
            let _ = factor_path;
        }

        let pending_tasks = factor_names
            .iter()
            .zip(factor_paths.iter())
            .filter_map(|(source_factor, factor_path)| {
                if completed_sources.contains(source_factor) {
                    None
                } else {
                    Some(TailTask {
                        source_factor: source_factor.clone(),
                        factor_path: factor_path.clone(),
                    })
                }
            })
            .collect::<Vec<_>>();

        let total_pending = pending_tasks.len();
        if total_pending > 0 {
            let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            print!(
                "\r[{}] Tail V4 启动，待处理 {}/{} 个原始因子，已恢复 {} 个",
                current_time,
                total_pending,
                factor_names.len(),
                restored_sources,
            );
            std::io::stdout().flush().map_err(|e| format!("刷新进度输出失败: {}", e))?;
        }
        let (task_sender, task_receiver): (Sender<TailTask>, Receiver<TailTask>) = unbounded();
        let (result_sender, result_receiver) = unbounded::<Result<TailTaskResult, (String, String)>>();
        for task in pending_tasks {
            task_sender.send(task).map_err(|e| format!("发送任务失败: {}", e))?;
        }
        drop(task_sender);

        let shared_arc = Arc::new(shared);
        let mut handles = Vec::with_capacity(n_jobs);
        for _ in 0..n_jobs {
            let rx = task_receiver.clone();
            let tx = result_sender.clone();
            let shared_clone = shared_arc.clone();
            handles.push(thread::spawn(move || {
                while let Ok(task) = rx.recv() {
                    let task_name = task.source_factor.clone();
                    let outcome = process_task(&task, &shared_clone).map_err(|err| (task_name, err));
                    if tx.send(outcome).is_err() {
                        break;
                    }
                }
            }));
        }
        drop(result_sender);

        let mut processed_sources = 0usize;
        while let Ok(task_outcome) = result_receiver.recv() {
            match task_outcome {
                Ok(task_result) => {
                    let result_path = factor_result_path(&task_results_dir, &task_result.source_factor);
                    write_task_result(&result_path, &task_result)?;
                    append_completed_source(&completed_log_path, &task_result.source_factor)?;
                    aggregated.merge_task(task_result);
                    processed_sources += 1;
                    if total_pending > 0 {
                        let elapsed = started.elapsed();
                        let elapsed_secs = elapsed.as_secs();
                        let progress = processed_sources as f64 / total_pending as f64;
                        let estimated_total_secs = if progress > 0.0 {
                            elapsed.as_secs_f64() / progress
                        } else {
                            elapsed.as_secs_f64()
                        };
                        let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() {
                            (estimated_total_secs - elapsed.as_secs_f64()) as u64
                        } else {
                            0
                        };
                        let (elapsed_h, elapsed_m, elapsed_s) = format_hms(elapsed_secs);
                        let (remaining_h, remaining_m, remaining_s) = format_hms(remaining_secs);
                        let progress_pct = progress * 100.0;
                        let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                        print!(
                            "\r[{}] Tail V4 进度 {}/{} ({:.1}%)，已恢复 {} 个，已用{}h{}m{}s，预计剩余{}h{}m{}s",
                            current_time,
                            processed_sources,
                            total_pending,
                            progress_pct,
                            restored_sources,
                            elapsed_h,
                            elapsed_m,
                            elapsed_s,
                            remaining_h,
                            remaining_m,
                            remaining_s,
                        );
                        std::io::stdout().flush().map_err(|e| format!("刷新进度输出失败: {}", e))?;
                    }
                }
                Err((task_name, err)) => {
                    return Err(format!("处理因子 {} 失败: {}", task_name, err));
                }
            }
        }

        if total_pending > 0 {
            println!();
        }

        for handle in handles {
            handle.join().map_err(|_| "Tail V4 worker 线程 join 失败".to_string())?;
        }

        write_aggregated_outputs(
            &cache_root_path,
            &aggregated,
        )?;

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

#[pyfunction]
#[pyo3(signature = (
    ver,
    gap5_selected,
    gap1_selected,
    cache_root,
    temp_root,
    source_dir,
    style_data_path,
    min_valid=12,
    start_date="2016-01-01",
    backtest_start_date="2016-02-01",
    end_date="2024-12-31",
    fulltest_jobs=32,
    python_path="/home/chenzongwei/.conda/envs/chenzongwei311/bin/python",
    resume=true,
    index_name="000905"
))]
pub fn tail_v4_run_fulltest_queue<'py>(
    py: Python<'py>,
    ver: String,
    gap5_selected: Vec<String>,
    gap1_selected: Vec<String>,
    cache_root: String,
    temp_root: String,
    source_dir: String,
    style_data_path: String,
    min_valid: usize,
    start_date: &str,
    backtest_start_date: &str,
    end_date: &str,
    fulltest_jobs: usize,
    python_path: &str,
    resume: bool,
    index_name: &str,
) -> PyResult<PyObject> {
    if fulltest_jobs == 0 {
        return Err(PyValueError::new_err("fulltest_jobs 必须大于 0"));
    }

    let output = py
        .allow_threads(|| -> Result<(usize, usize, HashMap<String, usize>), String> {
            let started = Instant::now();
            let cache_root_path = PathBuf::from(&cache_root);
            let postprocess_root = cache_root_path.join("postprocess_fulltest");
            let done_dir = postprocess_root.join("done");
            if !resume && postprocess_root.exists() {
                fs::remove_dir_all(&postprocess_root)
                    .map_err(|e| format!("删除旧 fulltest 恢复目录失败: {}", e))?;
            }
            fs::create_dir_all(&done_dir).map_err(|e| format!("创建 fulltest done 目录失败: {}", e))?;

            let all_tasks = build_fulltest_tasks(&gap5_selected, &gap1_selected);
            let total_tasks = all_tasks.len();

            let mut bucket_counts = HashMap::<String, usize>::new();
            let mut pending_tasks = VecDeque::<TailV4FulltestTask>::new();
            let mut restored_tasks = 0usize;
            for task in all_tasks {
                let done_path = fulltest_done_path(&done_dir, &task);
                if resume && done_path.exists() {
                    restored_tasks += 1;
                    increment_fulltest_bucket(&mut bucket_counts, &task.stage, task.gap);
                } else {
                    pending_tasks.push_back(task);
                }
            }

            let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            print!(
                "\r[{}] Fulltest 启动，待处理 {}/{} 个任务，已恢复 {} 个",
                current_time,
                pending_tasks.len(),
                total_tasks,
                restored_tasks,
            );
            std::io::stdout()
                .flush()
                .map_err(|e| format!("刷新 fulltest 启动进度失败: {}", e))?;

            if pending_tasks.is_empty() {
                render_fulltest_progress(0, restored_tasks, total_tasks, started, &bucket_counts)?;
                println!();
                return Ok((0, restored_tasks, bucket_counts));
            }

            let worker_config = TailV4FulltestWorkerConfig {
                ver: ver.clone(),
                temp_root,
                source_dir,
                factor_names: {
                    let mut factor_names = Vec::new();
                    let mut seen = HashSet::<String>::new();
                    for factor_name in gap5_selected.iter().chain(gap1_selected.iter()) {
                        if seen.insert(factor_name.clone()) {
                            factor_names.push(factor_name.clone());
                        }
                    }
                    factor_names
                },
                start_date: start_date.to_string(),
                backtest_start_date: backtest_start_date.to_string(),
                end_date: end_date.to_string(),
                style_data_path,
                min_valid,
                index_name: index_name.to_string(),
            };
            let worker_config_json =
                serde_json::to_string(&worker_config).map_err(|e| format!("序列化 worker 配置失败: {}", e))?;

            let task_queue = Arc::new(Mutex::new(pending_tasks));
            let stop_flag = Arc::new(AtomicBool::new(false));
            let (result_sender, result_receiver) =
                unbounded::<Result<TailV4FulltestWorkerResult, String>>();
            let worker_count = fulltest_jobs.min(total_tasks.max(1));
            let mut handles = Vec::with_capacity(worker_count);
            for worker_id in 0..worker_count {
                let queue_clone = Arc::clone(&task_queue);
                let sender_clone = result_sender.clone();
                let stop_clone = Arc::clone(&stop_flag);
                let python_path_owned = python_path.to_string();
                let worker_config_json_clone = worker_config_json.clone();
                handles.push(thread::spawn(move || {
                    run_tail_v4_fulltest_worker_process(
                        worker_id,
                        queue_clone,
                        sender_clone,
                        stop_clone,
                        python_path_owned,
                        worker_config_json_clone,
                    );
                }));
            }
            drop(result_sender);

            let pending_total = {
                let guard = task_queue
                    .lock()
                    .map_err(|_| "获取 fulltest 队列长度失败".to_string())?;
                guard.len()
            };
            let mut processed_tasks = 0usize;
            let mut fatal_error: Option<String> = None;
            while processed_tasks < pending_total {
                match result_receiver.recv() {
                    Ok(Ok(result)) => {
                        let task = TailV4FulltestTask {
                            task_key: result.task_key,
                            factor_name: result.factor_name,
                            stage: result.stage,
                            gap: result.gap,
                        };
                        let done_path = fulltest_done_path(&done_dir, &task);
                        write_fulltest_done(&done_path, &task)?;
                        processed_tasks += 1;
                        increment_fulltest_bucket(&mut bucket_counts, &task.stage, task.gap);
                        render_fulltest_progress(
                            processed_tasks,
                            restored_tasks,
                            total_tasks,
                            started,
                            &bucket_counts,
                        )?;
                    }
                    Ok(Err(err)) => {
                        fatal_error = Some(err);
                        stop_flag.store(true, AtomicOrdering::Relaxed);
                        break;
                    }
                    Err(_) => {
                        fatal_error = Some("fulltest worker 通道提前关闭".to_string());
                        stop_flag.store(true, AtomicOrdering::Relaxed);
                        break;
                    }
                }
            }

            for handle in handles {
                handle
                    .join()
                    .map_err(|_| "fulltest worker 线程 join 失败".to_string())?;
            }
            println!();

            if let Some(err) = fatal_error {
                return Err(err);
            }

            Ok((processed_tasks, restored_tasks, bucket_counts))
        })
        .map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_tasks", output.0)?;
    info.set_item("restored_tasks", output.1)?;
    let bucket_counts = PyDict::new(py);
    for (key, value) in output.2 {
        bucket_counts.set_item(key, value)?;
    }
    info.set_item("bucket_counts", bucket_counts)?;
    Ok(info.into())
}
