//! run_factor_pipeline —— 纯 Rust 因子流水线引擎（run_factor_pipeline 优化方案 Phase 3）。
//!
//! 把 go 函数（如 hm90.go）的整条链路在 Rust 内部一气呵成，消除 Python worker、
//! pandas、pyo3 往返、msgpack 等所有流转浪费。
//!
//! 并行控制：自适应嵌套并行 + 核绑定。
//! - 单一全局 rayon 线程池 = n_jobs（总线程恒定，绝不超额）
//! - 大文件 CSV 自动拆块被池内线程偷取，小文件不拆
//! - 可选核绑定（start_handler 把线程 idx 绑到物理核 idx）
//!
//! backup 格式与现有 run_pools_queue 完全兼容（复用 backup_writer），断点续算可用。
use crate::backup_reader::{read_existing_backup_with_filter, TaskResult};
use crate::backup_writer::save_results_to_backup;
use crate::fast_csv_reader;
use crate::features;
use crate::order_pair_metrics_pipeline;
use chrono::Local;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// hm90 流水线的参数。
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Hm90Params {
    /// V1 (calculate_order_pair_metrics_more) 的 tolerance，默认 0.001
    pub tolerance_v1: f64,
    /// V2 (calculate_order_pair_metrics_more_v2) 的 tolerance，默认 0.00001
    pub tolerance_v2: f64,
}

impl Default for Hm90Params {
    fn default() -> Self {
        Self {
            tolerance_v1: 0.001,
            tolerance_v2: 0.00001,
        }
    }
}

/// 从 Python dict 解析 hm90 参数。
fn parse_hm90_params(py: Python, params: &PyObject) -> PyResult<Hm90Params> {
    let mut p = Hm90Params::default();
    if let Ok(dict) = params.extract::<&PyDict>(py) {
        if let Some(v) = dict.get_item("tolerance_v1") {
            if !v.is_none() {
                p.tolerance_v1 = v.extract()?;
            }
        }
        if let Some(v) = dict.get_item("tolerance_v2") {
            if !v.is_none() {
                p.tolerance_v2 = v.extract()?;
            }
        }
    }
    Ok(p)
}

/// 从已加载的交易日历数组中，用二分查找返回严格小于 date 的最大交易日。
/// 对齐 Python 的 rp.td.last_trading_day(date)。
fn last_trading_day(trading_days: &[i64], date: i64) -> Option<i64> {
    // trading_days 已排序，找最后一个 < date 的元素
    let idx = trading_days.partition_point(|&d| d < date);
    if idx == 0 {
        None
    } else {
        Some(trading_days[idx - 1])
    }
}

/// hm90 流水线的单任务计算（纯 Rust，不经 Python）。
///
/// 输入 (date, code)，输出 Vec<f64> 因子值（vals1 + vals2 拼接）。
/// 失败时返回 NaN 填充向量（与 run_pools_queue 的错误处理一致）。
pub fn pipeline_order_pair_hm90(
    date: i64,
    code: &str,
    params: &Hm90Params,
    trading_days: &[i64],
    expected_len: usize,
) -> Vec<f32> {
    // 1. 读取今日 + 昨日数据（CSV 解析，含 afternoon 调整）。
    // 注意：parallel_threshold 设为 usize::MAX 禁用内部 rayon 并行，
    // 因为外层 run_factor_pipeline 已用自定义 pool 做任务级并行（n_jobs=200）。
    // 若内部再用全局 rayon 池，会导致嵌套池冲突、线程互相等待（实测 CPU 利用率暴跌）。
    // 任务内串行读取，依赖外层 200 个任务并行即可充分用满 CPU。
    let trade2 = match fast_csv_reader::read_trade_fast_inner(code, date, false, true, usize::MAX) {
        Ok(r) => r,
        Err(_) => return nan_vec(expected_len),
    };
    let prev_date = match last_trading_day(trading_days, date) {
        Some(d) => d,
        None => return nan_vec(expected_len),
    };
    let trade1 =
        match fast_csv_reader::read_trade_fast_inner(code, prev_date, false, true, usize::MAX) {
            Ok(r) => r,
            Err(_) => return nan_vec(expected_len),
        };

    // 2. order_pair_metrics（纯 Rust inner，零 pyo3 边界）
    let (result1, cols1) = order_pair_metrics_pipeline::calculate_order_pair_metrics_more_inner(
        &trade2,
        &trade1,
        params.tolerance_v1,
    );
    let (result2, cols2) = order_pair_metrics_pipeline::calculate_order_pair_metrics_more_v2_inner(
        &trade2,
        &trade1,
        params.tolerance_v2,
    );

    // 3. get_features_factors（纯 Rust，rayon 列级并行，关闭 lyapunov）
    // 空结果矩阵（0 配对）时返回半量 NaN，保证总长度 = expected_len
    let half = expected_len / 2;
    let result1_f32 = result1.mapv(|v| v as f32);
    let result2_f32 = result2.mapv(|v| v as f32);
    let (vals1, _) = if result1_f32.nrows() == 0 {
        (nan_vec(half), vec![])
    } else {
        features::get_features_factors_rust(&result1_f32.view(), &cols1)
    };
    let (vals2, _) = if result2_f32.nrows() == 0 {
        (nan_vec(half), vec![])
    } else {
        features::get_features_factors_rust(&result2_f32.view(), &cols2)
    };

    // 4. 拼接返回
    let mut res = vals1;
    res.extend(vals2);
    res
}
// ============================================================================
// observable_order pipeline
// ============================================================================

use crate::observable_order_metrics;

/// observable_order 流水线的参数。
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ObservableOrderParams {
    pub q: f64,
    pub drop_min: usize,
    pub forward_sec: f64,
    pub pre_minutes: f64,
    pub rolling_n: usize,
    pub autocorr_lag: usize,
}

impl Default for ObservableOrderParams {
    fn default() -> Self {
        Self {
            q: 0.10,
            drop_min: 10,
            forward_sec: 60.0,
            pre_minutes: 5.0,
            rolling_n: 20,
            autocorr_lag: 1,
        }
    }
}

/// observable_order 流水线的单任务计算。
/// 翻译 observable_order_go.py 的 go 函数：4 套配置 × (seg + pre5 + diff + abs_diff) = 16 组，
/// 每组调 get_features_factors_rust_full(with_threshold_counts=false) 生成特征。
pub fn pipeline_observable_order(
    date: i64,
    code: &str,
    ooparams: &ObservableOrderParams,
    _trading_days: &[i64],
    expected_len: usize,
) -> Vec<f32> {
    // 1. 调核心计算（4 套配置）
    let obs_params = observable_order_metrics::ObsOrderParams {
        q: ooparams.q as f32,
        drop_min: ooparams.drop_min,
        forward_sec: ooparams.forward_sec as f32,
        pre_minutes: ooparams.pre_minutes as f32,
        rolling_n: ooparams.rolling_n,
        autocorr_lag: ooparams.autocorr_lag,
    };
    let results =
        match observable_order_metrics::compute_observable_order_metrics(code, date, &obs_params) {
            Ok(r) => r,
            Err(_) => return nan_vec(expected_len),
        };

    // 2. 生成列名（过滤后 114 列，与 go 函数一致）
    let keep = observable_order_metrics::keep_cols();
    let col_names: Vec<String> = keep.iter().map(|&i| build_col_names().remove(i)).collect();
    // remove 会移动，改用索引
    let full_names = build_col_names();
    let col_names: Vec<String> = keep.iter().map(|&i| full_names[i].clone()).collect();
    let ncols_out = col_names.len();

    // 3. 对 4 套配置 × 4 种数组(seg/pre5/diff/abs_diff) = 16 组各算 get_features_factors
    let keys = ["bid_A", "bid_B", "ask_A", "ask_B"];
    let mut all_factors: Vec<f32> = Vec::with_capacity(expected_len);

    for (ci, _key) in keys.iter().enumerate() {
        let (seg_flat, seg_rows, pre5_flat, pre5_rows) = &results[ci];
        // 内部 NCOLS 列，过滤为 ncols_out 列
        let seg = filter_array2(seg_flat, *seg_rows, &keep);
        let pre5 = filter_array2(pre5_flat, *pre5_rows, &keep);
        // diff / abs_diff（过滤后的列上做差）
        let diff = if seg.nrows() > 0 && pre5.nrows() > 0 && seg.nrows() == pre5.nrows() {
            let mut d = ndarray::Array2::zeros(seg.dim());
            for r in 0..seg.nrows() {
                for c in 0..ncols_out {
                    d[[r, c]] = seg[[r, c]] - pre5[[r, c]];
                }
            }
            d
        } else {
            ndarray::Array2::zeros((0, ncols_out))
        };
        let abs_diff = {
            let mut d = ndarray::Array2::zeros(diff.dim());
            for r in 0..diff.nrows() {
                for c in 0..diff.ncols() {
                    d[[r, c]] = diff[[r, c]].abs();
                }
            }
            d
        };

        for arr in [seg.view(), pre5.view(), diff.view(), abs_diff.view()] {
            let (vals, _) = if arr.nrows() == 0 {
                (nan_vec(features_per_group(&col_names)), vec![])
            } else {
                features::get_features_factors_rust_full(&arr, &col_names, false)
            };
            all_factors.extend_from_slice(&vals);
        }
    }

    // 长度校准：若实际长度与 expected_len 不符，用 NaN 填充或截断
    if all_factors.len() < expected_len {
        all_factors.resize(expected_len, f32::NAN);
    } else if all_factors.len() > expected_len {
        all_factors.truncate(expected_len);
    }
    all_factors
}

/// 把展平的 Vec<f64> 转为 ndarray::Array2。
fn slice_to_array2(data: &[f32], rows: usize, cols: usize) -> ndarray::Array2<f32> {
    if rows == 0 || cols == 0 {
        return ndarray::Array2::zeros((0, cols));
    }
    ndarray::Array2::from_shape_vec((rows, cols), data.to_vec())
        .unwrap_or_else(|_| ndarray::Array2::zeros((0, cols)))
}

/// 把 (n_rows × NCOLS) 的展平数据按 keep 索引过滤为 (n_rows × keep.len()) 的 Array2。
fn filter_array2(data: &[f32], n_rows: usize, keep: &[usize]) -> ndarray::Array2<f32> {
    let ncols = observable_order_metrics::NCOLS;
    if n_rows == 0 || keep.is_empty() {
        return ndarray::Array2::zeros((0, keep.len()));
    }
    let mut out = vec![0.0f32; n_rows * keep.len()];
    for r in 0..n_rows {
        for (j, &c) in keep.iter().enumerate() {
            out[r * keep.len() + j] = data[r * ncols + c];
        }
    }
    ndarray::Array2::from_shape_vec((n_rows, keep.len()), out)
        .unwrap_or_else(|_| ndarray::Array2::zeros((0, keep.len())))
}

/// 计算 get_features_factors_rust_full(with_threshold_counts=false) 在 n_cols 列时的输出长度。
fn features_per_group(col_names: &[String]) -> usize {
    let n = col_names.len();
    // mean/median/std/skew/kurt + p5/p25/p75/p95/iqr/cv + autocorr1/abs + trend + period_diff/ratio + lz/entropy/max_range
    // = (5+6+2+1+2+3)*n + C(n,2)
    19 * n + n * (n - 1) / 2
}

/// 生成 147 列的列名（与 go 函数 observable_order_go.py 一致）。
fn build_col_names() -> Vec<String> {
    let mut names = Vec::with_capacity(observable_order_metrics::NCOLS);
    let aggs = ["mean", "std", "skew", "ac1", "trend"];
    // P1 盘口失衡 4 档 ×5 = 20
    for imb in ["imb1", "imb1_5", "imb1_10", "imb6_10"] {
        for a in &aggs {
            names.push(format!("{}_{}", imb, a));
        }
    }
    // P2 10档trend 同侧/对手 ×5 = 10
    for v in ["vt_same", "vt_opp"] {
        for a in &aggs {
            names.push(format!("{}_{}", v, a));
        }
    }
    // P3 10档std 同侧/对手 ×5 = 10
    for v in ["vs_same", "vs_opp"] {
        for a in &aggs {
            names.push(format!("{}_{}", v, a));
        }
    }
    // P7 rolling10 同侧 det/|det| ×5 = 10
    for r in ["r10_det", "r10_abs"] {
        for a in &aggs {
            names.push(format!("{}_{}", r, a));
        }
    }
    // P8 rolling20 双方 det/|det| ×5 = 10
    for r in ["r20_det", "r20_abs"] {
        for a in &aggs {
            names.push(format!("{}_{}", r, a));
        }
    }
    // 盘口标量 13（与 compute_segment_features 的 push 顺序严格对齐）
    // P4 价格波动率, P5 间隔成交量std, P6 间隔成交量trend
    // S1 段内10档归一化协方差det 8版本: 同侧det/|同侧|/对手det/|对手|/差/|差|/|同侧|-|对手|/||差||
    // S2 段内双方20档归一化协方差det 2版本: 双方det/|双方|
    for s in [
        "pvol",
        "ivstd",
        "ivtrend",
        "sd10_same",
        "sd10_same_abs",
        "sd10_opp",
        "sd10_opp_abs",
        "sd10_diff",
        "sd10_diff_abs",
        "sd10_absdiff",
        "sd10_absdiff_abs",
        "sd20_both",
        "sd20_both_abs",
    ] {
        names.push(s.to_string());
    }
    // T1 订单差值 原始/|·| ×5 = 10
    for od in ["od_raw", "od_abs"] {
        for a in &aggs {
            names.push(format!("{}_{}", od, a));
        }
    }
    // T2 主买占比 ×5 = 5
    for a in &aggs {
        names.push(format!("br_{}", a));
    }
    // T3 按秒成交笔数 买/卖/差/|差|/全量 ×5 = 25
    for ct in ["ct_b", "ct_s", "ct_d", "ct_da", "ct_a"] {
        for a in &aggs {
            names.push(format!("{}_{}", ct, a));
        }
    }
    // T4 按秒每笔均额 买/卖/差/|差|/全量 ×5 = 25
    for am in ["am_b", "am_s", "am_d", "am_da", "am_a"] {
        for a in &aggs {
            names.push(format!("{}_{}", am, a));
        }
    }
    // 逐笔标量 9
    for s in [
        "tr_d", "tr_ad", "bu_d", "bu_ad", "se_d", "se_ad", "bs_dd", "bs_dda", "btr",
    ] {
        names.push(s.to_string());
    }
    assert_eq!(names.len(), observable_order_metrics::NCOLS);
    names
}

/// 生成 NaN 填充向量（错误回退）。
fn nan_vec(len: usize) -> Vec<f32> {
    vec![f32::NAN; len]
}

/// Python 入口：run_factor_pipeline。
///
/// 参数：
/// - pipeline: 流水线标识，目前支持 "order_pair_hm90"
/// 从 Python dict 解析 observable_order 参数。
fn parse_observable_order_params(py: Python, params: &PyObject) -> PyResult<ObservableOrderParams> {
    let mut p = ObservableOrderParams::default();
    if let Ok(dict) = params.extract::<&PyDict>(py) {
        if let Some(v) = dict.get_item("q") {
            if !v.is_none() {
                p.q = v.extract()?;
            }
        }
        if let Some(v) = dict.get_item("drop_min") {
            if !v.is_none() {
                p.drop_min = v.extract()?;
            }
        }
        if let Some(v) = dict.get_item("forward_sec") {
            if !v.is_none() {
                p.forward_sec = v.extract()?;
            }
        }
        if let Some(v) = dict.get_item("pre_minutes") {
            if !v.is_none() {
                p.pre_minutes = v.extract()?;
            }
        }
        if let Some(v) = dict.get_item("rolling_n") {
            if !v.is_none() {
                p.rolling_n = v.extract()?;
            }
        }
        if let Some(v) = dict.get_item("autocorr_lag") {
            if !v.is_none() {
                p.autocorr_lag = v.extract()?;
            }
        }
    }
    Ok(p)
}
/// - tasks: [[date, code], ...]
/// - n_jobs: 并行线程数（默认 200）
/// - backup_file: 备份文件路径
/// - expected_result_length: 每个任务的预期结果长度
/// - trading_days: 交易日历数组（由 rp.td.trading_days 提供）
/// - params: 可选参数 dict（tolerance_v1, tolerance_v2）
/// - update_mode: 断点续算（默认 False）
/// - bind_cores: 是否核绑定（默认 True）
#[pyfunction]
#[pyo3(signature = (
    pipeline, tasks, n_jobs, backup_file, expected_result_length, trading_days,
    params=None, update_mode=None, bind_cores=true, backup_batch_size=None, progress_log=None, mode=None,
    export_names=None, export_dir=None, export_n_jobs=80,
    store_dir=None, store_factor_names=None
))]
#[allow(clippy::too_many_arguments)]
pub fn run_factor_pipeline(
    pipeline: &str,
    tasks: &PyList,
    n_jobs: usize,
    backup_file: String,
    expected_result_length: usize,
    trading_days: Vec<i64>,
    params: Option<PyObject>,
    update_mode: Option<bool>,
    bind_cores: bool,
    backup_batch_size: Option<usize>,
    progress_log: Option<bool>,
    mode: Option<String>,
    export_names: Option<Vec<String>>,
    export_dir: Option<String>,
    export_n_jobs: usize,
    store_dir: Option<String>,
    store_factor_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let py = unsafe { Python::assume_gil_acquired() };

    let pipeline_name = pipeline.to_string();
    let known = ["order_pair_hm90", "observable_order"];
    if !known.contains(&pipeline_name.as_str()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "未知流水线: {}（支持: {:?}）",
            pipeline, known
        )));
    }

    let update_mode_enabled = update_mode.unwrap_or(false);
    let progress_log_enabled = progress_log.unwrap_or(false);
    let batch_size = backup_batch_size.unwrap_or(500);
    let mode = mode.unwrap_or_else(|| "multiprocess".to_string());
    let n_shards = 8;
    let sharded_sink: Option<crate::factor_store_v5::ShardedBackupSink> = if let Some(ref sdir) =
        store_dir
    {
        let snames = store_factor_names.clone().unwrap_or_else(|| {
            (0..expected_result_length)
                .map(|i| format!("factor_{i}"))
                .collect()
        });
        Some(
            crate::factor_store_v5::ShardedBackupSink::new_colblk_sharded(sdir, &snames, n_shards)
                .map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("打开 colblk 存储失败: {}", e))
                })?,
        )
    } else {
        None
    };
    let sink: crate::factor_store_v5::BackupSink =
        crate::factor_store_v5::BackupSink::new_bin(backup_file.clone(), expected_result_length);

    // 解析参数（按 pipeline 名选不同 Params）
    let hm90_params = if pipeline_name == "order_pair_hm90" {
        if let Some(p) = &params {
            parse_hm90_params(py, p)?
        } else {
            Hm90Params::default()
        }
    } else {
        Hm90Params::default()
    };
    let oo_params = if pipeline_name == "observable_order" {
        if let Some(p) = &params {
            parse_observable_order_params(py, p)?
        } else {
            ObservableOrderParams::default()
        }
    } else {
        ObservableOrderParams::default()
    };

    // 解析任务列表
    let mut all_tasks: Vec<(i64, String)> = Vec::with_capacity(tasks.len());
    for item in tasks.iter() {
        let pair: &PyList = item.extract()?;
        if pair.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "每个任务应为 [date, code]",
            ));
        }
        let date: i64 = pair.get_item(0)?.extract()?;
        let code: String = pair.get_item(1)?.extract()?;
        all_tasks.push((date, code));
    }

    // 断点续算：过滤已完成任务
    let pending: Vec<(i64, String)> = if update_mode_enabled {
        let task_dates: std::collections::HashSet<i64> =
            all_tasks.iter().map(|(d, _)| *d).collect();
        let existing = if let Some(ref ss) = sharded_sink {
            ss.check_completed()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("读取备份失败: {}", e)))?
        } else {
            sink.check_completed()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("读取备份失败: {}", e)))?
        };
        all_tasks
            .into_iter()
            .filter(|(d, c)| !existing.contains(&(*d, c.clone())))
            .collect()
    } else {
        all_tasks
    };

    let total = pending.len();
    if total == 0 {
        println!("✅ 所有任务都已完成");
        // 即使无新任务，投影仍可能未做（断点续算恢复时）。
        // 只对需要投影的后端（store_dir 模式）执行，且仅当未投影时。
        if let Some(ref ss) = sharded_sink {
            println!("🏗️ 检查/执行投影...");
            ss.finish_and_project(export_n_jobs)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("投影失败: {}", e)))?;
        }
        return Ok(Python::with_gil(|py| py.None()));
    }
    println!("📋 待处理任务: {}, n_jobs={}", total, n_jobs);
    // mode 路由：多进程（默认，L3缓存隔离）或线程（回退）
    if mode == "multiprocess" {
        if let Some(worker_bin) = locate_worker_binary() {
            run_multiprocess_v2(
                py,
                pending,
                n_jobs,
                sink.clone_handle(),
                sharded_sink.clone(),
                expected_result_length,
                trading_days,
                hm90_params,
                oo_params,
                pipeline_name,
                bind_cores,
                batch_size,
                progress_log_enabled,
                &worker_bin,
            )?;
            if let Some(ref ss) = sharded_sink {
                ss.finish_and_project(export_n_jobs).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("投影失败: {}", e))
                })?;
            } else if let Some(ref names) = export_names {
                let dir = export_dir.clone().unwrap_or_else(|| {
                    let ver = std::path::Path::new(&backup_file)
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "factors".to_string())
                        .strip_prefix("backup_")
                        .unwrap_or("factors")
                        .to_string();
                    format!("/nas197/user_home_unsafe/chenzongwei/factor_data/{}", ver)
                });
                let py2 = unsafe { Python::assume_gil_acquired() };
                export_backup_to_parquet_py(py2, &backup_file, names, &dir, export_n_jobs)?;
            }
            return Ok(Python::with_gil(|py| py.None()));
        } else {
            println!(
                "⚠️ 找不到 rust_pyfunc_worker 二进制，回退到线程模式。可用 mode=thread 显式指定。"
            );
        }
    }

    // 释放 GIL，在 rayon 池中执行（Python 对象不再被访问）
    let backup_file_cloned = backup_file.clone();
    let trading_days_arc = std::sync::Arc::new(trading_days);
    let completed = std::sync::Arc::new(AtomicUsize::new(0));
    let start = std::time::Instant::now();

    let results_buf: std::sync::Arc<Mutex<Vec<TaskResult>>> =
        std::sync::Arc::new(Mutex::new(Vec::with_capacity(batch_size)));
    // 备份间隔记录（供进度监控线程算最近5批预估）
    let backup_intervals: std::sync::Arc<Mutex<Vec<f64>>> =
        std::sync::Arc::new(Mutex::new(Vec::new()));
    py.allow_threads(|| -> PyResult<()> {
        // 用 std::thread + crossbeam channel（无锁多消费者）替代 rayon。
        // rayon 工作窃取在本场景（长尾大文件任务）并行效率低（实测仅 ~45%）。
        // crossbeam channel 的多消费者竞争接收用原子操作，无 Mutex 锁竞争，
        // 200 线程可高效并行取任务。每个线程内部完全串行计算，等价于进程模型。
        let (task_tx, task_rx) = crossbeam::channel::unbounded::<(i64, String)>();

        // 核绑定：获取核 ID 列表
        let core_ids = if bind_cores {
            core_affinity::get_core_ids()
        } else {
            None
        };
        let cores_arc = core_ids.map(|c| std::sync::Arc::new(c));

        // 启动 worker 线程
        let mut handles = Vec::with_capacity(n_jobs);
        for worker_idx in 0..n_jobs {
            let task_rx = task_rx.clone();
            let params = hm90_params.clone();
            let oo_params_t = oo_params.clone();
            let pipeline_name_t = pipeline_name.clone();
            let trading_days = trading_days_arc.clone();
            let results_buf = results_buf.clone();
            let backup_intervals_t = backup_intervals.clone();
            let backup_file = backup_file_cloned.clone();
            let completed = completed.clone();
            let cores = cores_arc.clone();
            let mut last_flush_t = std::time::Instant::now();

            let handle = std::thread::spawn(move || {
                // 核绑定：worker_idx → 物理核 worker_idx
                if let Some(ref cores) = cores {
                    let target = worker_idx % cores.len();
                    let _ = core_affinity::set_for_current(cores[target].clone());
                }

                // 循环取任务并计算（完全串行，无内部并行）
                while let Ok((date, code)) = task_rx.recv() {
                    let vals = if pipeline_name_t == "observable_order" {
                        pipeline_observable_order(
                            date,
                            &code,
                            &oo_params_t,
                            &trading_days,
                            expected_result_length,
                        )
                    } else {
                        pipeline_order_pair_hm90(
                            date,
                            &code,
                            &params,
                            &trading_days,
                            expected_result_length,
                        )
                    };

                    let result = TaskResult {
                        date,
                        code,
                        timestamp: 0,
                        facs: vals.clone(),
                    };

                    // 累积结果，达 batch_size 时批量写 backup
                    let should_flush = {
                        let mut buf = results_buf.lock().unwrap();
                        buf.push(result);
                        buf.len() >= batch_size
                    };
                    if should_flush {
                        let batch: Vec<TaskResult> = {
                            let mut buf = results_buf.lock().unwrap();
                            buf.drain(..).collect()
                        };
                        let _ =
                            save_results_to_backup(&batch, &backup_file, expected_result_length);
                        // 记录备份间隔（供监控线程估算）
                        let mut intervals = backup_intervals_t.lock().unwrap();
                        intervals.push(last_flush_t.elapsed().as_secs_f64());
                        last_flush_t = std::time::Instant::now();
                    }

                    completed.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(handle);
        }
        // 启动独立进度监控线程（不影响计算）
        let monitor_handle = spawn_progress_monitor(
            completed.clone(),
            total,
            start,
            batch_size,
            backup_intervals.clone(),
        );

        // 分发所有任务到队列
        for task in pending {
            let _ = task_tx.send(task);
        }
        drop(task_tx); // 关闭发送端，worker recv 返回 Err 后退出

        // 等待所有 worker 完成
        for h in handles {
            let _ = h.join();
        }
        let _ = monitor_handle.join();

        // 写入剩余结果
        let remaining: Vec<TaskResult> = results_buf.lock().unwrap().drain(..).collect();
        if !remaining.is_empty() {
            let _ = save_results_to_backup(&remaining, &backup_file_cloned, expected_result_length);
        }

        Ok(())
    })?;

    let done = completed.load(Ordering::Relaxed);
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "✅ 完成 {}/{} 任务, 总用时 {:.1}s, 平均速度 {:.1} 任务/秒",
        done,
        total,
        elapsed,
        done as f64 / elapsed
    );

    // 可选：计算完成后自动导出 backup → parquet（每个因子一个文件）
    if let Some(names) = export_names {
        let dir = export_dir.unwrap_or_else(|| {
            // 默认输出到 /nas197/user_home_unsafe/chenzongwei/factor_data/<ver>/
            let ver = std::path::Path::new(&backup_file)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "factors".to_string())
                .strip_prefix("backup_")
                .unwrap_or("factors")
                .to_string();
            format!("/nas197/user_home_unsafe/chenzongwei/factor_data/{}", ver)
        });
        let py = unsafe { Python::assume_gil_acquired() };
        export_backup_to_parquet_py(py, &backup_file, &names, &dir, export_n_jobs)?;
    }

    Ok(Python::with_gil(|py| py.None()))
}

/// 在 Rust 内部完成 backup→parquet 导出（不绕 Python）。
/// 调 backup_column_cache::export_backup_to_parquet_rust，用 rayon 并行写 parquet。
fn export_backup_to_parquet_py(
    _py: Python<'_>,
    backup_file: &str,
    names: &[String],
    output_dir: &str,
    n_jobs: usize,
) -> PyResult<()> {
    match crate::backup_column_cache::export_backup_to_parquet_rust(
        backup_file,
        names,
        output_dir,
        n_jobs,
    ) {
        Ok(n) => {
            if n == 0 {
                println!("📋 全部 {} 个因子已存在，跳过导出", names.len());
            }
            Ok(())
        }
        Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e)),
    }
}

// ============================================================================
// 多进程模式（run_factor_pipeline 的 mode="multiprocess"）
//
// 核心思路：worker 从共享地址空间的线程改为独立地址空间的 Rust 进程，
// 获得进程级 L3 缓存隔离（解决线程版 100 并发下缓存争用的问题），
// 同时保留零 Python/pandas/pyo3 开销的单任务速度优势。
//
// 架构：主进程 spawn N 个 worker_pipeline 二进制进程，
//   通过 stdin/stdout pipe + 4字节长度前缀 + bincode 通信。
// ============================================================================

/// IPC 消息：主进程 → worker
#[derive(serde::Serialize, serde::Deserialize)]
pub enum TaskMessage {
    /// 初始化（worker 启动时一次性发送，传入共享的交易日历和参数）
    Init {
        pipeline_name: String,
        params: Hm90Params,
        oo_params: ObservableOrderParams,
        trading_days: Vec<i64>,
        expected_len: usize,
    },
    /// 单个任务
    Task { date: i64, code: String },
    /// 关闭信号（长度0也可，这里用显式 Shutdown 更清晰）
    Shutdown,
}

/// IPC 消息：worker → 主进程
#[derive(serde::Serialize, serde::Deserialize)]
pub enum ResultMessage {
    /// 计算结果
    Result(TaskResult),
    /// worker 内错误（不 panic，回传错误避免进程崩溃重启）
    Error {
        date: i64,
        code: String,
        msg: String,
    },
    /// worker 就绪确认（收到 Init 后回复）
    Ready,
}

/// 写一个 IPC 消息到 writer（4字节长度前缀 + bincode）。
pub fn ipc_write<W: std::io::Write>(writer: &mut W, msg: &TaskMessage) -> std::io::Result<()> {
    let data =
        bincode::serialize(msg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let len = data.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&data)?;
    writer.flush()?;
    Ok(())
}

/// 写一个结果消息到 writer。
pub fn ipc_write_result<W: std::io::Write>(
    writer: &mut W,
    msg: &ResultMessage,
) -> std::io::Result<()> {
    let data =
        bincode::serialize(msg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let len = data.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&data)?;
    writer.flush()?;
    Ok(())
}

/// 读一个 IPC 消息（4字节长度前缀 + bincode）。长度=0 表示 Shutdown。
pub fn ipc_read_task<R: std::io::Read>(reader: &mut R) -> std::io::Result<Option<TaskMessage>> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len == 0 {
        return Ok(None); // Shutdown
    }
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data)?;
    let msg = bincode::deserialize(&data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(Some(msg))
}

/// 读一个结果消息。
pub fn ipc_read_result<R: std::io::Read>(reader: &mut R) -> std::io::Result<ResultMessage> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "收到长度0的结果（异常）",
        ));
    }
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data)?;
    let msg = bincode::deserialize(&data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(msg)
}

/// 定位 worker 二进制路径。
/// 搜索顺序：环境变量 RUST_PYFUNC_WORKER_BIN → Python 包目录 → cargo target 目录。
fn locate_worker_binary() -> Option<String> {
    // 1. 环境变量
    if let Ok(path) = std::env::var("RUST_PYFUNC_WORKER_BIN") {
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    // 2. Python 包目录（alter.sh 会把 worker bin 复制到 python/rust_pyfunc/）
    //    通过当前 .so 的路径推断包目录
    if let Ok(so_path) = std::env::current_exe() {
        // so_path 是 python 解释器路径，不可靠。改为搜索常见 conda/site-packages 路径
        for prefix in &[
            "/home/chenzongwei/.conda/envs/chenzongwei311/lib/python3.11/site-packages/rust_pyfunc",
            "/home/chenzongwei/rust_pyfunc/python/rust_pyfunc",
        ] {
            let p = format!("{}/rust_pyfunc_worker", prefix);
            if std::path::Path::new(&p).exists() {
                return Some(p);
            }
        }
    }
    // 3. cargo target 目录（开发时）
    let dev_path = "/home/chenzongwei/rust_pyfunc/target/release/rust_pyfunc_worker";
    if std::path::Path::new(dev_path).exists() {
        return Some(dev_path.to_string());
    }
    None
}
/// 独立进度监控线程：每 10 秒打印进度 + 两种剩余时长预估。
/// 通过 Arc<AtomicUsize> 读 completed，不干扰计算线程。
fn spawn_progress_monitor(
    completed: std::sync::Arc<AtomicUsize>,
    total: usize,
    start: std::time::Instant,
    batch_size: usize,
    backup_intervals: std::sync::Arc<Mutex<Vec<f64>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut tick = 0u32;
        loop {
            // 1 秒间隔轮询，每 10 秒打印一次（避免 join 时等 10 秒）
            std::thread::sleep(std::time::Duration::from_secs(1));
            tick += 1;
            let done = completed.load(Ordering::Relaxed);
            if done >= total {
                break;
            }
            if tick % 10 != 0 {
                continue;
            }

            let elapsed = start.elapsed().as_secs_f64();
            let pct = done as f64 / total as f64 * 100.0;

            // 全局预估：剩余 / 全局速度
            let global_speed = done as f64 / elapsed;
            let global_eta = if global_speed > 0.0 {
                (total - done) as f64 / global_speed
            } else {
                f64::NAN
            };

            // 最近5次备份预估：用最近5次备份的平均间隔算速度
            let recent_eta = {
                let intervals = backup_intervals.lock().unwrap();
                if intervals.len() >= 2 {
                    let n_take = intervals.len().min(5);
                    let recent: Vec<f64> = intervals.iter().rev().take(n_take).copied().collect();
                    let avg_interval = recent.iter().sum::<f64>() / recent.len() as f64;
                    let batch_speed = batch_size as f64 / avg_interval;
                    if batch_speed > 0.0 {
                        (total - done) as f64 / batch_speed
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                }
            };

            fn fmt_eta(secs: f64) -> String {
                if secs.is_nan() {
                    return "N/A".to_string();
                }
                if secs < 60.0 {
                    return format!("{:.0}s", secs);
                }
                if secs < 3600.0 {
                    return format!("{:.0}m", secs / 60.0);
                }
                format!("{:.1}h", secs / 3600.0)
            }

            println!(
                "[{}] 📊 进度: {}/{} ({:.1}%), 速度: {:.0}/s, 已用: {}, 预估剩余(全局): {}, 预估剩余(近5批): {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                done, total, pct, global_speed, fmt_eta(elapsed), fmt_eta(global_eta), fmt_eta(recent_eta)
            );
        }
    })
}
#[allow(clippy::too_many_arguments)]
fn run_multiprocess_v2(
    py: Python<'_>,
    pending: Vec<(i64, String)>,
    n_jobs: usize,
    sink: crate::factor_store_v5::BackupSink,
    sharded_sink: Option<crate::factor_store_v5::ShardedBackupSink>,
    expected_result_length: usize,
    trading_days: Vec<i64>,
    params: Hm90Params,
    oo_params: ObservableOrderParams,
    pipeline_name: String,
    bind_cores: bool,
    batch_size: usize,
    progress_log_enabled: bool,
    worker_bin: &str,
) -> PyResult<()> {
    use std::io::{BufReader, BufWriter, Write};
    use std::process::{Command, Stdio};

    let total = pending.len();
    let trading_days_arc = std::sync::Arc::new(trading_days);
    let completed = std::sync::Arc::new(AtomicUsize::new(0));
    let start = std::time::Instant::now();

    py.allow_threads(|| -> PyResult<()> {
        // 任务队列（crossbeam MPMC，支持多消费者竞争 recv）
        let (task_tx, task_rx) = crossbeam::channel::unbounded::<(i64, String)>();
        for task in pending {
            let _ = task_tx.send(task);
        }
        drop(task_tx);

        // 结果 channel（worker 线程 → collector 线程）
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<TaskResult>(n_jobs * 2);

        // 核绑定信息
        let core_ids = if bind_cores {
            core_affinity::get_core_ids()
        } else {
            None
        };

        // spawn N 个 worker 管理线程
        let mut handles = Vec::with_capacity(n_jobs);
        for worker_idx in 0..n_jobs {
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
            let params = params.clone();
            let oo_params = oo_params.clone();
            let pipeline_name = pipeline_name.clone();
            let trading_days = trading_days_arc.clone();
            let worker_bin = worker_bin.to_string();
            let core_affinity_idx = if let Some(ref cores) = core_ids {
                Some(worker_idx % cores.len())
            } else {
                None
            };

            let handle = std::thread::spawn(move || {
                run_single_worker_manager(
                    &worker_bin,
                    worker_idx,
                    task_rx,
                    result_tx,
                    &params,
                    &oo_params,
                    &pipeline_name,
                    &trading_days,
                    expected_result_length,
                    core_affinity_idx,
                );
            });
            handles.push(handle);
        }
        drop(result_tx);

        // 备份间隔记录（供进度监控线程算最近5批预估）
        let backup_intervals: std::sync::Arc<Mutex<Vec<f64>>> =
            std::sync::Arc::new(Mutex::new(Vec::new()));

        // 启动独立进度监控线程（不影响计算）
        let monitor_handle = spawn_progress_monitor(
            completed.clone(),
            total,
            start,
            batch_size,
            backup_intervals.clone(),
        );

        // collector 线程：收集结果，批量写 backup
        let collector_handle = {
            let sink_c = sink.clone_handle();
            let sharded_c = sharded_sink.clone();
            let completed_cloned = completed.clone();
            let backup_intervals_cloned = backup_intervals.clone();
            std::thread::spawn(move || {
                let mut buf: Vec<TaskResult> = Vec::with_capacity(batch_size);
                let mut last_flush = std::time::Instant::now();
                while let Ok(result) = result_rx.recv() {
                    buf.push(result);
                    let _done = completed_cloned.fetch_add(1, Ordering::Relaxed) + 1;
                    if buf.len() >= batch_size {
                        let batch: Vec<TaskResult> = buf.drain(..).collect();
                        let _ = if let Some(ref ss) = sharded_c {
                            ss.append_batch(&batch)
                        } else {
                            sink_c.append_batch(&batch)
                        };
                        // 记录本次备份间隔
                        let interval = last_flush.elapsed().as_secs_f64();
                        last_flush = std::time::Instant::now();
                        backup_intervals_cloned.lock().unwrap().push(interval);
                    }
                }
                if !buf.is_empty() {
                    let _ = if let Some(ref ss) = sharded_c {
                        ss.append_batch(&buf)
                    } else {
                        sink_c.append_batch(&buf)
                    };
                }
            })
        };

        // 等待所有 worker 管理线程完成
        for h in handles {
            let _ = h.join();
        }
        let _ = collector_handle.join();
        // 让监控线程退出（completed 已达 total）
        let _ = monitor_handle.join();

        Ok(())
    })?;

    let done = completed.load(Ordering::Relaxed);
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "✅ [多进程] 完成 {} 任务, 总用时 {:.1}s, 平均速度 {:.1} 任务/秒",
        done,
        elapsed,
        done as f64 / elapsed
    );

    Ok(())
}

/// 单个 worker 进程的管理器（在一个线程内运行）。
/// 负责：spawn worker 进程 → 发 Init → 循环取任务/发任务/收结果 → IPC 错误时重启。
#[allow(clippy::too_many_arguments)]
fn run_single_worker_manager(
    worker_bin: &str,
    worker_idx: usize,
    task_rx: crossbeam::channel::Receiver<(i64, String)>,
    result_tx: std::sync::mpsc::SyncSender<TaskResult>,
    params: &Hm90Params,
    oo_params: &ObservableOrderParams,
    pipeline_name: &str,
    trading_days: &[i64],
    expected_len: usize,
    core_affinity_idx: Option<usize>,
) {
    use std::io::{BufReader, BufWriter, Write};
    use std::process::{Child, Command, Stdio};

    // 外层循环：管理 worker 进程的生命周期（崩溃后重启）
    'outer: loop {
        // spawn worker 进程
        let mut cmd = Command::new(worker_bin);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        // 设置环境变量：RAYON_NUM_THREADS=1（worker 内部单线程）
        cmd.env("RAYON_NUM_THREADS", "1");
        cmd.env("OMP_NUM_THREADS", "1");
        cmd.env("OPENBLAS_NUM_THREADS", "1");
        cmd.env("MKL_NUM_THREADS", "1");
        if let Some(idx) = core_affinity_idx {
            cmd.env("RUST_PYFUNC_CORE_AFFINITY_IDX", idx.to_string());
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("⚠️ worker{} spawn 失败: {}, 3秒后重试", worker_idx, e);
                std::thread::sleep(std::time::Duration::from_secs(3));
                continue;
            }
        };

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let mut writer = BufWriter::with_capacity(1 << 20, stdin);
        let mut reader = BufReader::with_capacity(1 << 20, stdout);

        // 发送 Init 消息
        let init_msg = TaskMessage::Init {
            pipeline_name: pipeline_name.to_string(),
            params: params.clone(),
            oo_params: oo_params.clone(),
            trading_days: trading_days.to_vec(),
            expected_len,
        };
        if ipc_write(&mut writer, &init_msg).is_err() {
            let _ = child.kill();
            continue;
        }
        // 等待 Ready 确认
        match ipc_read_result(&mut reader) {
            Ok(ResultMessage::Ready) => {}
            _ => {
                let _ = child.kill();
                continue;
            }
        }

        // 内层循环：取任务 → 发任务 → 收结果
        loop {
            let (date, code) = match task_rx.recv() {
                Ok(t) => t,
                Err(_) => {
                    // 无任务，发 Shutdown 并退出
                    let _ = ipc_write(&mut writer, &TaskMessage::Shutdown);
                    // 也发长度0作为备用关闭信号
                    let _ = writer.write_all(&[0u8; 4]);
                    let _ = child.wait();
                    break 'outer;
                }
            };

            // 发任务
            let task_msg = TaskMessage::Task {
                date,
                code: code.clone(),
            };
            if ipc_write(&mut writer, &task_msg).is_err() {
                // IPC 错误：该任务记 NaN，重启 worker
                let _ = result_tx.send(TaskResult {
                    date,
                    code,
                    timestamp: 0,
                    facs: vec![f32::NAN],
                });
                let _ = child.kill();
                continue 'outer;
            }

            // 收结果
            match ipc_read_result(&mut reader) {
                Ok(ResultMessage::Result(r)) => {
                    let _ = result_tx.send(r);
                }
                Ok(ResultMessage::Error { date, code, msg }) => {
                    eprintln!(
                        "⚠️ worker{} 计算错误 [{},{}]: {}",
                        worker_idx, date, code, msg
                    );
                    let _ = result_tx.send(TaskResult {
                        date,
                        code,
                        timestamp: 0,
                        facs: vec![f32::NAN],
                    });
                }
                Ok(ResultMessage::Ready) => {
                    // 忽略意外的 Ready
                }
                Err(_) => {
                    // IPC 错误，重启 worker
                    eprintln!("⚠️ worker{} IPC 错误，重启中...", worker_idx);
                    let _ = result_tx.send(TaskResult {
                        date,
                        code,
                        timestamp: 0,
                        facs: vec![f32::NAN],
                    });
                    let _ = child.kill();
                    continue 'outer;
                }
            }
        }
    }
}
