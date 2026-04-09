use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray,
};
use chrono::{Datelike, NaiveDateTime};
use crossbeam::channel::{unbounded, Receiver, Sender};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::{read_npy, write_npy};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

use crate::tail_v2_backtest_block::backtest_block_f32_with_parallel;
use crate::tail_v2_block_neutralizer::neutralize_block_f32_out_with_parallel;
use crate::tail_v2_rank_roll_factor::rank_roll_block_f32_with_parallel;

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
    style_cube: Arc<Array3<f32>>,
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
    raw_ic_gap1: HashMap<String, Vec<f32>>,
    raw_ic_gap5: HashMap<String, Vec<f32>>,
    neu_ic_gap1: HashMap<String, Vec<f32>>,
    neu_ic_gap5: HashMap<String, Vec<f32>>,
}

impl AggregatedCandidates {
    fn merge_task(&mut self, task: TailTaskResult) {
        self.raw_summary_gap1.extend(task.raw_summary_gap1);
        self.raw_summary_gap5.extend(task.raw_summary_gap5);
        self.neu_summary_gap1.extend(task.neu_summary_gap1);
        self.neu_summary_gap5.extend(task.neu_summary_gap5);
        for record in task.raw_ic_gap1 {
            self.raw_ic_gap1.insert(record.factor_name, record.values);
        }
        for record in task.raw_ic_gap5 {
            self.raw_ic_gap5.insert(record.factor_name, record.values);
        }
        for record in task.neu_ic_gap1 {
            self.neu_ic_gap1.insert(record.factor_name, record.values);
        }
        for record in task.neu_ic_gap5 {
            self.neu_ic_gap5.insert(record.factor_name, record.values);
        }
    }
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

        let (raw_summary_gap1, raw_ic_gap1) = backtest_block_f32_with_parallel(
            rolled_block.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            1,
            10,
            false,
        )?;
        let (raw_summary_gap5, raw_ic_gap5) = backtest_block_f32_with_parallel(
            rolled_block.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            5,
            10,
            false,
        )?;

        let neutralized = neutralize_block_f32_out_with_parallel(
            shared.style_cube.view(),
            rolled_block.view(),
            true,
            shared.min_valid,
            false,
        )?;
        let (neu_summary_gap1, neu_ic_gap1) = backtest_block_f32_with_parallel(
            neutralized.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            1,
            10,
            false,
        )?;
        let (neu_summary_gap5, neu_ic_gap5) = backtest_block_f32_with_parallel(
            neutralized.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            5,
            10,
            false,
        )?;

        for (slot_idx, derived_name) in derived_names.iter().enumerate() {
            let raw_gap1_row = summary_from_row(
                derived_name,
                "rolled",
                1,
                &variant_name,
                raw_summary_gap1.row(slot_idx).as_slice().unwrap(),
            );
            let raw_gap5_row = summary_from_row(
                derived_name,
                "rolled",
                5,
                &variant_name,
                raw_summary_gap5.row(slot_idx).as_slice().unwrap(),
            );
            let neu_gap1_row = summary_from_row(
                derived_name,
                "neu",
                1,
                &variant_name,
                neu_summary_gap1.row(slot_idx).as_slice().unwrap(),
            );
            let neu_gap5_row = summary_from_row(
                derived_name,
                "neu",
                5,
                &variant_name,
                neu_summary_gap5.row(slot_idx).as_slice().unwrap(),
            );

            let raw_gap1_keep = qualify_raw(&raw_gap1_row, 1, &shared.config);
            let raw_gap5_keep = qualify_raw(&raw_gap5_row, 5, &shared.config);
            let neu_gap1_keep = qualify_neu(&neu_gap1_row, 1, &shared.config);
            let neu_gap5_keep = qualify_neu(&neu_gap5_row, 5, &shared.config);

            if raw_gap1_keep {
                result.raw_summary_gap1.push(raw_gap1_row.clone());
                result.raw_ic_gap1.push(IcRecord {
                    factor_name: derived_name.clone(),
                    values: raw_ic_gap1.column(slot_idx).iter().map(|v| *v as f32).collect(),
                });
            }
            if raw_gap5_keep {
                result.raw_summary_gap5.push(raw_gap5_row.clone());
                result.raw_ic_gap5.push(IcRecord {
                    factor_name: derived_name.clone(),
                    values: raw_ic_gap5.column(slot_idx).iter().map(|v| *v as f32).collect(),
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
                    values: neu_ic_gap1.column(slot_idx).iter().map(|v| *v as f32).collect(),
                });
            }
            if raw_gap5_keep || neu_gap5_keep {
                result.neu_ic_gap5.push(IcRecord {
                    factor_name: derived_name.clone(),
                    values: neu_ic_gap5.column(slot_idx).iter().map(|v| *v as f32).collect(),
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
    date_count: usize,
    store: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    if let Some(parent) = matrix_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建 ic 目录失败: {}", e))?;
    }
    let factor_names = {
        let mut names = store.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    };
    let matrix = if factor_names.is_empty() {
        Array2::<f32>::from_shape_vec((date_count, 0), Vec::new())
            .map_err(|e| format!("构造空 ic 矩阵失败: {}", e))?
    } else {
        let mut data = vec![f32::NAN; date_count * factor_names.len()];
        for (col_idx, name) in factor_names.iter().enumerate() {
            let values = store
                .get(name)
                .ok_or_else(|| format!("缺少候选 IC 数据: {}", name))?;
            for row_idx in 0..date_count.min(values.len()) {
                data[row_idx * factor_names.len() + col_idx] = values[row_idx];
            }
        }
        Array2::<f32>::from_shape_vec((date_count, factor_names.len()), data)
            .map_err(|e| format!("构造 ic 矩阵失败: {}", e))?
    };
    write_npy(matrix_path, &matrix).map_err(|e| format!("写 ic npy 失败: {}", e))?;
    let file = File::create(names_path).map_err(|e| format!("创建 ic names json 失败: {}", e))?;
    serde_json::to_writer(BufWriter::new(file), &factor_names)
        .map_err(|e| format!("写 ic names json 失败: {}", e))?;
    Ok(())
}

fn write_aggregated_outputs(
    cache_root: &Path,
    aggregated: &AggregatedCandidates,
    gap1_ic_dates: usize,
    gap5_ic_dates: usize,
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
        gap1_ic_dates,
        &aggregated.raw_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_rolled_gap5.npy"),
        &ic_dir.join("ic_rolled_gap5_names.json"),
        gap5_ic_dates,
        &aggregated.raw_ic_gap5,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap1.npy"),
        &ic_dir.join("ic_neu_gap1_names.json"),
        gap1_ic_dates,
        &aggregated.neu_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap5.npy"),
        &ic_dir.join("ic_neu_gap5_names.json"),
        gap5_ic_dates,
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
    style_cube_path,
    ret_gap1_path,
    ret_sum_gap1_path,
    ret_gap5_path,
    ret_sum_gap5_path,
    restrict_path,
    index_ret_path,
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
    style_cube_path: String,
    ret_gap1_path: String,
    ret_sum_gap1_path: String,
    ret_gap5_path: String,
    ret_sum_gap5_path: String,
    restrict_path: String,
    index_ret_path: String,
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
            style_cube: Arc::new(read_npy(&style_cube_path).map_err(|e| format!("读取 style_cube.npy 失败: {}", e))?),
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
            shared_arc.ret_gap1.nrows(),
            shared_arc.ret_gap5.nrows() / 5,
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
