//! tail_backtest_engine：全新高性能回测引擎。
//!
//! 核心设计：回归 v4 的 thread-per-factor 模型（N 线程各自独立处理整个流水线），
//! 消灭 v5 的 IO/CPU 分离 + bounded(16) 瓶颈。直接读 colblk 列式存储（scatter_map 优化），
//! 所有 worker 共享一个 Reader + scatter_maps，从 unbounded channel 取因子独立处理。
//!
//! 与 v4 的区别：直接读 colblk（不走 parquet 中间格式）。
//! 与 v5 的区别：无 IO/CPU 分离，unbounded channel 不限制并行度。

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crossbeam::channel::{unbounded, Receiver, Sender};
use ndarray::Array2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::tail_v5_pipeline::{
    self, build_selection_config, build_shared_inputs, format_hms, init_status_line,
    read_task_result, reset_status_line, update_status_line, write_aggregated_outputs_with_stem,
    AggregatedCandidates, ProcessStats, SharedInputs, TailTask, TailTaskResult,
};

/// 单个因子的完整处理（IO + 计算 + 写结果），在一个线程内串行执行。
///
/// v8 三档融合路径（2026-09）：按日期块融合 rolling → preflight → 中性化 → 回测，
/// 不物化 12 张 (T,N) 派生面。需要 `industry_neutralize=true` + `neutralize_std_shared`
/// + `bt_pre`（O1 回测路径）三者齐备；任一不满足则回落到 v7 原路径。
/// 环境变量 `TAIL_ENGINE_V7=1` 可强制走旧路径（A/B 对账用）。
fn process_single_factor(
    task: &TailTask,
    raw_values: Array2<f32>,
    shared: &SharedInputs,
    sc: &mut crate::tail_v8_pipeline::V8Scratch,
    low_cov: Option<&mut crate::tail_v8_pipeline::LowCoverageAudit>,
) -> Result<TailTaskResult, String> {
    let force_v7 = std::env::var("TAIL_ENGINE_V7").is_ok();
    // 注：industry_neutralize=false（纯风格中性化）同样走 v8 融合路径——
    // 行业码矩阵仍是标准预处理（行业 OLS 填充 / 行业中位填充）的必需输入，
    // 该开关只决定最后一步残差回归是否含行业 one-hot。
    let v8_ok = !force_v7
        && shared.neutralize_std_shared.is_some()
        && shared.bt_pre.is_some()
        && shared.free_mask.is_some()
        // 11 风格（纯风格 + 冷门）已完整接入 v8 + v3 融合流水线（单跑/双跑均逐位一致）；
        // 仅在极罕见的「行业 one-hot + 冷门」组合下退回 v7 通用 OLS。
        && (shared.cold_style.is_none() || !shared.industry_neutralize);
    if v8_ok {
        crate::tail_v8_pipeline::process_task_with_values_v8_opts(task, raw_values, shared, sc, low_cov)
    } else if low_cov.is_some() {
        // 低覆盖放行只在 v8 融合路径实现；退回旧路径会被悄悄填中位数，必须报错而不是静默改口径。
        Err(format!(
            concat!(
                "因子 {} 命中低覆盖放行规则，但当前走的是 v7 旧路径",
                "（TAIL_ENGINE_V7 / 输入不全 / 开了冷门中性化），旧路径不支持放行。",
                "冷门中性化目前只在 v7 路径实现，与低覆盖放行互斥：",
                "要么传 cold_neutralize=False，要么先关掉低覆盖放行，",
                "别让因子被静默填中位数"
            ),
            task.source_factor
        ))
    } else {
        tail_v5_pipeline::process_task_with_values_v7(task, raw_values, shared)
    }
}

/// 低覆盖放行规则：因子名包含任一标记词（如 "cxl"），或在显式名单里，即放行。
/// 两者都为空 = 不放行任何因子（默认，行为逐位不变）。
#[derive(Debug, Clone, Default)]
struct LowCoverageRule {
    markers: Vec<String>,
    names: HashSet<String>,
}

impl LowCoverageRule {
    fn is_empty(&self) -> bool {
        self.markers.is_empty() && self.names.is_empty()
    }
    fn matches(&self, source_factor: &str) -> bool {
        self.names.contains(source_factor)
            || self.markers.iter().any(|m| source_factor.contains(m.as_str()))
    }
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
    style_vars_dir,
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
    save_all_metrics=false,
    industry_neutralize=true,
    industry_matrix=None,
    cold_style=None,
    cold_dual_pass=false,
    ic_only=false,
    io_concurrency=None, io_batch_size=None, io_batch_bytes=None,
    low_coverage_markers=None, low_coverage_names=None,
    extra_styles=None,
    cold_stem=None,
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
    style_vars_dir: String,
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
    save_all_metrics: bool,
    industry_neutralize: bool,
    industry_matrix: Option<numpy::PyReadonlyArray2<'py, f64>>,
    // 冷门（热度）中性化：单列 (T,N) f64（旧 Cold v1 兼容入口）；None = 关闭
    cold_style: Option<numpy::PyReadonlyArray2<'py, f64>>,
    // 有冷门/扩展风格时是否两套指标都算：false（推荐）= 只算扩展风格，速度与没有冷门接近
    cold_dual_pass: bool,
    ic_only: bool,
    io_concurrency: Option<usize>,
    io_batch_size: Option<usize>,
    io_batch_bytes: Option<usize>,
    low_coverage_markers: Option<Vec<String>>,
    low_coverage_names: Option<Vec<String>>,
    // 多列扩展风格中性化：(T, N, M) f64（如 Cold v2 M=3 或 Cold v2 + TradeSize M=4）；传入时优先于 cold_style
    extra_styles: Option<numpy::PyReadonlyArray3<'py, f64>>,
    cold_stem: Option<String>,
) -> PyResult<PyObject> {
    let resolved_cold_stem: String = match cold_stem {
        Some(ref s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => {
            if extra_styles.is_some() {
                "coldv2".to_string()
            } else {
                "neu11".to_string()
            }
        }
    };
    // 低覆盖放行（撤单类因子：2021-11-05 前沪市无撤单，只有深市有值）。默认 None = 关闭。
    let low_cov_rule = LowCoverageRule {
        markers: low_coverage_markers
            .unwrap_or_default()
            .into_iter()
            .filter(|m| !m.is_empty())
            .collect(),
        names: low_coverage_names.unwrap_or_default().into_iter().collect(),
    };
    let low_cov_hits: Vec<&String> =
        factor_names.iter().filter(|n| low_cov_rule.matches(n)).collect();
    if !low_cov_rule.is_empty() {
        println!(
            "  [Engine] 警告：低覆盖放行已开启（标记词 {:?}，显式名单 {} 个）。命中 {} / {} 个原始因子：\
             这些因子缺失值不填中位数、不因覆盖率或缺失占比被淘汰，各闸门读数写入 {{cache_root}}/low_coverage_audit/",
            low_cov_rule.markers,
            low_cov_rule.names.len(),
            low_cov_hits.len(),
            factor_names.len(),
        );
        if let Some(missing) = low_cov_rule
            .names
            .iter()
            .filter(|n| !factor_names.contains(n))
            .min()
        {
            return Err(PyValueError::new_err(format!(
                "low_coverage_names 里有不在本次因子列表中的名字（例如 {missing}），请检查拼写"
            )));
        }
    }
    let low_cov_rule = Arc::new(low_cov_rule);
    if factor_names.len() != factor_paths.len() {
        return Err(PyValueError::new_err(
            "factor_names 和 factor_paths 长度必须一致",
        ));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs 必须大于 0"));
    }

    let profile = crate::storage_policy::StorageProfile::detect(Path::new(&colblk_store_dir))
        .map_err(PyRuntimeError::new_err)?;
    let resolved_io = io_concurrency.unwrap_or(profile.io_concurrency(n_jobs));
    let resolved_batch = io_batch_size.unwrap_or(profile.io_batch_size());
    let resolved_batch_bytes = io_batch_bytes.unwrap_or(profile.io_batch_bytes());
    if resolved_io == 0 || resolved_batch == 0 || resolved_batch_bytes == 0 {
        return Err(PyValueError::new_err(
            "io_concurrency/io_batch_size/io_batch_bytes 必须 > 0",
        ));
    }
    let read_count = Arc::new(AtomicUsize::new(0));
    let industry = industry_matrix
        .ok_or_else(|| {
            PyValueError::new_err("tail_backtest_engine 需要 industry_matrix (模板轴行业码)")
        })?
        .as_array()
        .to_owned();

    // 冷门/扩展风格中性化矩阵必须在 allow_threads 之前转成纯 Rust C 序连续 Array3<f64>：
    // PyReadonlyArray2/3 内含 PyObject 引用，不能跨线程（不满足 Send）。
    let cold_style_arr: Option<ndarray::Array3<f64>> = match (extra_styles, cold_style) {
        (Some(a3), _) => {
            let v = a3.as_array();
            let (ct, cn, cm) = v.dim();
            if (ct, cn) != industry.dim() || cm == 0 || cm > 12 {
                return Err(PyValueError::new_err(format!(
                    "extra_styles 形状 {:?} 与模板轴 {:?} 不一致（或列数 M={} 不在 1..=12）",
                    v.dim(),
                    industry.dim(),
                    cm
                )));
            }
            Some(v.as_standard_layout().into_owned())
        }
        (None, Some(a2)) => {
            let v = a2.as_array();
            if v.dim() != industry.dim() {
                return Err(PyValueError::new_err(format!(
                    "cold_style 形状 {:?} 与模板轴 {:?} 不一致",
                    v.dim(),
                    industry.dim()
                )));
            }
            let (t_d, n_d) = v.dim();
            let owned2 = v.as_standard_layout().into_owned();
            Some(
                owned2
                    .into_shape((t_d, n_d, 1))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        }
        (None, None) => None,
    };

    let output = py.allow_threads(
        || -> Result<(usize, usize, HashMap<String, usize>, HashMap<String, (f64, f64)>), String> {
        let started = Instant::now();
        let cache_root_path = PathBuf::from(&cache_root);
        let task_results_dir = cache_root_path.join("task_results");
        let logs_dir = cache_root_path.join("logs");
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
            save_all_metrics,
            ic_only,
        );
        let mut shared = build_shared_inputs(
            dates, stocks, windows, fold, min_valid, backtest_start,
            industry_neutralize,
            Some(industry),
            cold_style_arr,
            cold_dual_pass,
            &style_vars_dir,
            &ret_gap1_path, &ret_sum_gap1_path,
            &ret_gap5_path, &ret_sum_gap5_path,
            &restrict_path, &index_ret_path,
            config,
        )?;
        shared.cold_stem = resolved_cold_stem.clone();

        // ---- 断点续算：优先从单文件批量 WAL 恢复，并兼容历史 *.msgpack 小文件 ----
        let mut aggregated = AggregatedCandidates::default();
        let mut completed_sources = HashSet::<String>::new();
        let mut prefill_coverage = HashMap::<String, (f64, f64)>::new();
        let mut stats = ProcessStats::default();
        let requested_set: HashSet<&str> = factor_names.iter().map(|s| s.as_str()).collect();

        let mut record_restored =
            |task_result: TailTaskResult, completed_sources: &mut HashSet<String>| {
                let source_factor = task_result.source_factor.clone();
                if !requested_set.contains(source_factor.as_str())
                    || completed_sources.contains(&source_factor)
                {
                    return;
                }
                let passed = task_result.passed;
                let raw_cov = task_result.eliminated_by_raw_cover;
                let any_window = task_result.any_window_passed_preflight;
                let has_data = !task_result.raw_summary_gap1.is_empty()
                    || !task_result.raw_summary_gap5.is_empty()
                    || !task_result.neu_summary_gap1.is_empty()
                    || !task_result.neu_summary_gap5.is_empty()
                    || !task_result.neu11_summary_gap1.is_empty()
                    || !task_result.neu11_summary_gap5.is_empty();
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
                prefill_coverage.insert(
                    source_factor.clone(),
                    (
                        task_result.raw_cover_before_fill,
                        task_result.raw_cover_after_fill,
                    ),
                );
                aggregated.merge_task(task_result);
                completed_sources.insert(source_factor);
            };

        let wal_path = task_results_dir.join("checkpoint_batch.bin");
        if wal_path.exists() {
            use std::io::{Read, Seek, SeekFrom};
            let mut wal_file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&wal_path)
                .map_err(|e| format!("打开批量断点文件失败: {e}"))?;
            let file_len = wal_file.metadata().map(|m| m.len()).unwrap_or(0);
            let mut valid_pos = 0u64;
            let mut hdr = [0u8; 4];
            let mut payload = Vec::new();
            while valid_pos + 4 <= file_len {
                if wal_file.read_exact(&mut hdr).is_err() {
                    break;
                }
                let frame_len = u32::from_le_bytes(hdr) as usize;
                if frame_len == 0 || valid_pos + 4 + frame_len as u64 > file_len {
                    break;
                }
                payload.resize(frame_len, 0u8);
                if wal_file.read_exact(&mut payload).is_err() {
                    break;
                }
                match rmp_serde::from_slice::<TailTaskResult>(&payload) {
                    Ok(tr) => {
                        valid_pos += 4 + frame_len as u64;
                        record_restored(tr, &mut completed_sources);
                    }
                    Err(_) => break,
                }
            }
            if valid_pos < file_len {
                let _ = wal_file.set_len(valid_pos);
                let _ = wal_file.seek(SeekFrom::End(0));
            }
        }

        // 兼容历史遗留的单因子 *.msgpack（单次 read_dir 扫描，避免无旧文件时数万次 stat）
        if let Ok(entries) = fs::read_dir(&task_results_dir) {
            let mut legacy_files: Vec<(String, PathBuf)> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let p = e.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("msgpack") {
                        let stem = p.file_stem()?.to_str()?.to_string();
                        if requested_set.contains(stem.as_str()) && !completed_sources.contains(&stem) {
                            return Some((stem, p));
                        }
                    }
                    None
                })
                .collect();
            legacy_files.sort_by(|a, b| a.0.cmp(&b.0));
            for (source_factor, result_path) in legacy_files {
                let task_result = read_task_result(&result_path)
                    .map_err(|e| format!("读取已保存因子结果 {source_factor} 失败: {e}"))?;
                record_restored(task_result, &mut completed_sources);
            }
        }
        drop(record_restored);

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

        let mut pending_tasks=pending_tasks;
        pending_tasks.sort_by_key(|task|parse_col_idx(&task.factor_path).unwrap_or(usize::MAX));
        let read_cols:Vec<_>=pending_tasks.iter().filter_map(|t|parse_col_idx(&t.factor_path)).collect();
        let total_pending = pending_tasks.len();

        // n_jobs 仍然保持 thread-per-factor 语义，但每个 factor 的内存足迹已通过流式
        // rank_roll / 单 slot 中性化大幅降低（见 process_task_with_values_v7）。
        let n_workers = n_jobs.min(total_pending);

        // 机械盘设备级排他读盘锁：防止多个大系列同时扫同一块 /hdd 机械盘阵列导致随机寻道降速 10~16 倍。
        // 仅在读盘回测循环期间持有，worker 结束后立即释放，下游筛选/Fulltest 不占锁。
        let mut hdd_io_lock_guard: Option<fs::File> = None;
        #[cfg(unix)]
        if total_pending > 0
            && !profile.network
            && (profile.btrfs || colblk_store_dir.starts_with("/hdd"))
            && std::env::var("RUST_PYFUNC_DISABLE_HDD_LOCK")
                .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
                .unwrap_or(true)
        {
            use std::os::unix::fs::MetadataExt;
            use std::os::unix::io::AsRawFd;
            if let Ok(meta) = fs::metadata(&colblk_store_dir) {
                let dev_id = meta.dev();
                let lock_path = format!("/tmp/rp_backtest_hdd_dev_{dev_id}.lock");
                if let Ok(lf) = fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(false)
                    .open(&lock_path)
                {
                    let try_rc = unsafe { libc::flock(lf.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                    if try_rc != 0 {
                        let wait_t0 = Instant::now();
                        println!(
                            "  [Engine] 检测到同一机械盘阵列 (dev={dev_id}) 有其他回测引擎正在顺序读盘，自动排队错峰等待以避免磁头寻道降速..."
                        );
                        unsafe {
                            libc::flock(lf.as_raw_fd(), libc::LOCK_EX);
                        }
                        println!(
                            "  [Engine] 已获得机械盘读盘锁（排队 {:.1}s），开始顺序连读",
                            wait_t0.elapsed().as_secs_f64()
                        );
                    }
                    hdd_io_lock_guard = Some(lf);
                }
            }
        }

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
            let l3 = format!(
                "恢复 {} 个 | 即将开始处理（n_jobs={}，实际 {} 线程）...",
                restored_sources, n_jobs, n_workers,
            );
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
        let cancelled=Arc::new(std::sync::atomic::AtomicBool::new(false));

        // 所有 worker 共享同一个 Reader 和同一份 scatter_maps。
        // 旧实现每个线程各开一个 Reader，会复制 8 个分片的全部 factor_names/dates/codes 字典；
        // n_jobs 很大时仅是这些字典就会吃掉几十 GB。pread 本身线程安全，共享无副作用。
        let shared_reader = if n_workers > 0 {
            let reader = Arc::new(
                crate::factor_store_v5::FactorStoreReader::open(&colblk_store_dir)
                    .and_then(|r|r.with_io_concurrency(resolved_io))
                    .map_err(|e| format!("打开 colblk 存储失败: {e}"))?,
            );
            let scatter_maps = Arc::new(reader.precompute_scatter_maps(
                shared_arc.dates.as_slice(),
                shared_arc.stocks.as_slice(),
            ));
            let valid_cols:Vec<_>=read_cols.iter().copied().filter(|&c|c<reader.factor_names().len()).collect();
            let cache=Arc::new(crate::storage_read_cache::FactorReadCache::new(&reader,&valid_cols,resolved_batch,shared_arc.dates.len().saturating_mul(shared_arc.stocks.len()).saturating_mul(4),resolved_batch_bytes)?);
            Some((reader, scatter_maps, cache))
        } else {
            None
        };

        // 启动 n_workers 个 worker 线程（thread-per-factor，但并发数已被内存上限约束）
        let mut handles = Vec::with_capacity(n_workers);
        if let Some((reader, scatter_maps, cache)) = shared_reader.as_ref() {
            for _ in 0..n_workers {
                let rx = task_receiver.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                let reader = reader.clone();
                let scatter_maps = scatter_maps.clone();
                let cache=cache.clone();let cancelled=cancelled.clone();
                let low_cov_rule = low_cov_rule.clone();
                let low_cov_dir = cache_root_path.join("low_coverage_audit");
                let dates_arc = shared_clone.dates.clone();
                let stocks_arc = shared_clone.stocks.clone();
                handles.push(thread::spawn(move || {
                    let n_stocks = shared_clone.restrict.ncols();
                    let mut v8sc = crate::tail_v8_pipeline::V8Scratch::new(
                        n_stocks,
                        shared_clone.windows.as_slice(),
                        crate::tail_v8_pipeline::BLOCK_ROWS,
                    );
                    while let Ok(task) = rx.recv() {
                        if cancelled.load(AtomicOrdering::Relaxed) {break;}
                        // 1. 解析 col_idx（失败 = 硬错误：带因子名返回，绝不静默跳过）
                        let col_idx = match parse_col_idx(&task.factor_path) {
                            Some(v) => v,
                            None => {
                                let _ = tx.send(Err((
                                    task.source_factor.clone(),
                                    format!("解析 col_idx 失败（factor_path 缺少 ::col_idx）: {}", task.factor_path),
                                )));
                                break;
                            }
                        };

                        // 2. 快速读因子（scatter_map 优化，共享 Reader）
                        // 读取失败 = 硬错误：带因子名返回，绝不静默遗漏（否则 processed+restored < requested）。
                        let raw_values = match cache.take(
                            &reader,col_idx,
                            dates_arc.as_slice(),
                            stocks_arc.as_slice(),
                            &scatter_maps,
                        ) {
                            Ok(m) => m,
                            Err(e) => {
                                let _ = tx.send(Err((
                                    task.source_factor.clone(),
                                    format!("读取因子失败（col_idx={col_idx}）: {e}"),
                                )));
                                break;
                            }
                        };

                        // 3. 完整计算流水线（rank_roll → preflight → backtest → neutralize → backtest）
                        let task_name = task.source_factor.clone();
                        let mut audit = if low_cov_rule.matches(&task.source_factor) {
                            Some(crate::tail_v8_pipeline::LowCoverageAudit::default())
                        } else {
                            None
                        };
                        let outcome = process_single_factor(
                            &task, raw_values, &shared_clone, &mut v8sc, audit.as_mut(),
                        )
                        .and_then(|r| {
                            if let Some(a) = audit.as_ref() {
                                write_low_coverage_audit(&low_cov_dir, a)?;
                            }
                            Ok(r)
                        })
                        .map_err(|err| (task_name, err));

                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
        }
        drop(result_sender);

        // ---- 主线程收集结果（写入单文件批量 WAL，每 256 个因子或 5 秒 sync 一次，消除 10万+ 次双 fsync） ----
        let mut wal_writer = if total_pending > 0 {
            let f = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&wal_path)
                .map_err(|e| format!("打开批量断点写入文件失败: {e}"))?;
            Some(std::io::BufWriter::with_capacity(4 * 1024 * 1024, f))
        } else {
            None
        };
        let mut unsynced_count = 0usize;
        let mut last_sync_at = Instant::now();
        let mut processed_sources = 0usize;
        while let Ok(task_outcome) = result_receiver.recv() {
            match task_outcome {
                Ok(task_result) => {
                    let is_passed = task_result.passed;
                    let is_raw_cov = task_result.eliminated_by_raw_cover;
                    let any_window = task_result.any_window_passed_preflight;
                    let preflight_maj = task_result.preflight_maj_failed_windows;
                    let preflight_zero = task_result.preflight_zero_failed_windows;
                    let preflight_nan = task_result.preflight_nan_failed_windows;
                    if let Some(ref mut w) = wal_writer {
                        use std::io::Write;
                        let payload = rmp_serde::to_vec_named(&task_result)
                            .map_err(|e| format!("序列化任务结果失败: {e}"))?;
                        let len_u32 = u32::try_from(payload.len())
                            .map_err(|_| "单因子回测结果超出 4GiB".to_string())?;
                        w.write_all(&len_u32.to_le_bytes())
                            .and_then(|_| w.write_all(&payload))
                            .map_err(|e| format!("追加断点记录失败: {e}"))?;
                        unsynced_count += 1;
                        if unsynced_count >= 256 || last_sync_at.elapsed().as_secs() >= 5 {
                            w.flush().map_err(|e| format!("刷新断点缓冲失败: {e}"))?;
                            let _ = w.get_ref().sync_data();
                            unsynced_count = 0;
                            last_sync_at = Instant::now();
                        }
                    }
                    prefill_coverage.insert(
                        task_result.source_factor.clone(),
                        (
                            task_result.raw_cover_before_fill,
                            task_result.raw_cover_after_fill,
                        ),
                    );
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
                    cancelled.store(true,AtomicOrdering::Relaxed);
                    for h in handles {let _=h.join();}
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
        // 读盘已全部结束，立即释放机械盘排他锁供其它排队任务接力读盘
        drop(hdd_io_lock_guard);

        if let Some(mut w) = wal_writer {
            use std::io::Write;
            w.flush().map_err(|e| format!("收尾刷新断点缓冲失败: {e}"))?;
            crate::storage_policy::sync_file(w.get_ref(), "回测断点WAL")?;
        }

        if let Some((_,_,cache))=&shared_reader {read_count.store(cache.reads.load(AtomicOrdering::Relaxed),AtomicOrdering::Relaxed);}
        // P0-1 读取闭环断言：走到这里时（没有收到过 Err 结果），每个因子要么被恢复、
        // 要么被成功处理。两者之和必须等于请求总数，否则说明存在静默遗漏
        // （例如 worker panic / 结果 channel 提前关闭），必须报错而不是产出残缺结果。
        let requested = factor_names.len();
        if processed_sources + restored_sources != requested {
            return Err(format!(
                "读取/处理闭环不完整: requested={requested}, processed={processed_sources}, restored={restored_sources}"
            ));
        }

        write_aggregated_outputs_with_stem(&cache_root_path, &aggregated, &resolved_cold_stem)?;
        let stamp_json = serde_json::json!({
            "version": 1,
            "requested_sources": requested,
            "processed_sources": processed_sources,
            "restored_sources": restored_sources,
            "ic_only": ic_only,
            "industry_neutralize": industry_neutralize,
            "cold_dual_pass": cold_dual_pass,
            "cold_stem": resolved_cold_stem,
        });
        if let Ok(stamp_bytes) = serde_json::to_vec_pretty(&stamp_json) {
            let _ = crate::storage_policy::atomic_write(
                &cache_root_path.join("engine_completed.json"),
                &stamp_bytes,
            );
        }

        let mut candidate_counts = HashMap::new();
        candidate_counts.insert("rolled_gap1".to_string(), aggregated.raw_summary_gap1.len());
        candidate_counts.insert("rolled_gap5".to_string(), aggregated.raw_summary_gap5.len());
        candidate_counts.insert("neu_gap1".to_string(), aggregated.neu_summary_gap1.len());
        candidate_counts.insert("neu_gap5".to_string(), aggregated.neu_summary_gap5.len());
        candidate_counts.insert(
            format!("{resolved_cold_stem}_gap1"),
            aggregated.neu11_summary_gap1.len(),
        );
        candidate_counts.insert(
            format!("{resolved_cold_stem}_gap5"),
            aggregated.neu11_summary_gap5.len(),
        );
        Ok((
            processed_sources,
            restored_sources,
            candidate_counts,
            prefill_coverage,
        ))
    })
    .map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_sources", output.0)?;
    info.set_item(
        "storage_kind",
        if profile.network {
            "nas"
        } else if profile.btrfs {
            "local_btrfs"
        } else {
            "local"
        },
    )?;
    info.set_item("io_concurrency", resolved_io)?;
    info.set_item("io_batch_size", resolved_batch)?;
    info.set_item("io_batch_bytes", resolved_batch_bytes)?;
    info.set_item("io_read_groups", read_count.load(AtomicOrdering::Relaxed))?;
    info.set_item("restored_sources", output.1)?;
    let candidate_counts = PyDict::new(py);
    for (key, value) in output.2 {
        candidate_counts.set_item(key, value)?;
    }
    info.set_item("candidate_counts", candidate_counts)?;
    let prefill_dict = PyDict::new(py);
    for (source, (before, after)) in &output.3 {
        let pair = PyDict::new(py);
        pair.set_item("raw_cover_before_fill", before)?;
        pair.set_item("raw_cover_after_fill", after)?;
        prefill_dict.set_item(source, pair)?;
    }
    info.set_item("prefill_coverage", prefill_dict)?;
    Ok(info.into())
}

/// 写一个放行因子的对账文件：{cache_root}/low_coverage_audit/{因子名}.json。
/// 内容只依赖因子数据与阈值（派生面按固定顺序），同一输入每次写出相同字节。
fn write_low_coverage_audit(
    dir: &Path,
    audit: &crate::tail_v8_pipeline::LowCoverageAudit,
) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("创建 low_coverage_audit 目录失败: {e}"))?;
    let safe: String = audit
        .source_factor
        .chars()
        .map(|c| if c == '/' || c == '\\' { '_' } else { c })
        .collect();
    let path = dir.join(format!("{safe}.json"));
    let tmp = dir.join(format!("{safe}.json.tmp"));
    let bytes = serde_json::to_vec_pretty(audit)
        .map_err(|e| format!("序列化低覆盖对账失败: {e}"))?;
    fs::write(&tmp, bytes).map_err(|e| format!("写低覆盖对账失败: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("发布低覆盖对账失败: {e}"))?;
    Ok(())
}
