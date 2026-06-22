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
) -> Vec<f64> {
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
    let trade1 = match fast_csv_reader::read_trade_fast_inner(code, prev_date, false, true, usize::MAX) {
        Ok(r) => r,
        Err(_) => return nan_vec(expected_len),
    };

    // 2. order_pair_metrics（纯 Rust inner，零 pyo3 边界）
    let (result1, cols1) = order_pair_metrics_pipeline::calculate_order_pair_metrics_more_inner(
        &trade2, &trade1, params.tolerance_v1,
    );
    let (result2, cols2) =
        order_pair_metrics_pipeline::calculate_order_pair_metrics_more_v2_inner(
            &trade2, &trade1, params.tolerance_v2,
        );

    // 3. get_features_factors（纯 Rust，rayon 列级并行，关闭 lyapunov）
    // 空结果矩阵（0 配对）时返回半量 NaN，保证总长度 = expected_len
    let half = expected_len / 2;
    let (vals1, _) = if result1.nrows() == 0 {
        (nan_vec(half), vec![])
    } else {
        features::get_features_factors_rust(&result1.view(), &cols1)
    };
    let (vals2, _) = if result2.nrows() == 0 {
        (nan_vec(half), vec![])
    } else {
        features::get_features_factors_rust(&result2.view(), &cols2)
    };

    // 4. 拼接返回
    let mut res = vals1;
    res.extend(vals2);
    res
}

/// 生成 NaN 填充向量（错误回退）。
fn nan_vec(len: usize) -> Vec<f64> {
    vec![f64::NAN; len]
}

/// Python 入口：run_factor_pipeline。
///
/// 参数：
/// - pipeline: 流水线标识，目前支持 "order_pair_hm90"
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
    params=None, update_mode=None, bind_cores=true, backup_batch_size=None, progress_log=None, mode=None
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
) -> PyResult<PyObject> {
    let py = unsafe { Python::assume_gil_acquired() };

    if pipeline != "order_pair_hm90" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "未知流水线: {}（目前仅支持 'order_pair_hm90'）", pipeline
        )));
    }

    let update_mode_enabled = update_mode.unwrap_or(false);
    let progress_log_enabled = progress_log.unwrap_or(false);
    let batch_size = backup_batch_size.unwrap_or(2000);
    let mode = mode.unwrap_or_else(|| "multiprocess".to_string());

    // 解析参数
    let hm90_params = if let Some(p) = &params {
        parse_hm90_params(py, p)?
    } else {
        Hm90Params::default()
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
        let existing = read_existing_backup_with_filter(&backup_file, Some(&task_dates))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("读取备份失败: {}", e)))?;
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
        return Ok(Python::with_gil(|py| py.None()));
    }
    println!("📋 待处理任务: {}, n_jobs={}", total, n_jobs);
    // mode 路由：多进程（默认，L3缓存隔离）或线程（回退）
    if mode == "multiprocess" {
        if let Some(worker_bin) = locate_worker_binary() {
            return run_multiprocess_v2(
                py, pending, n_jobs, backup_file, expected_result_length,
                trading_days, hm90_params, bind_cores, batch_size, progress_log_enabled,
                &worker_bin,
            ).map(|_| Python::with_gil(|py| py.None()));
        } else {
            println!("⚠️ 找不到 rust_pyfunc_worker 二进制，回退到线程模式。可用 mode=thread 显式指定。");
        }
    }

    // 释放 GIL，在 rayon 池中执行（Python 对象不再被访问）
    let backup_file_cloned = backup_file.clone();
    let trading_days_arc = std::sync::Arc::new(trading_days);
    let completed = std::sync::Arc::new(AtomicUsize::new(0));
    let start = std::time::Instant::now();

    // 结果累积器（线程安全，批量写 backup）
    let results_buf: std::sync::Arc<Mutex<Vec<TaskResult>>> =
        std::sync::Arc::new(Mutex::new(Vec::with_capacity(batch_size)));

    py.allow_threads(|| -> PyResult<()> {
        // 用 std::thread + crossbeam channel（无锁多消费者）替代 rayon。
        // rayon 工作窃取在本场景（长尾大文件任务）并行效率低（实测仅 ~45%）。
        // crossbeam channel 的多消费者竞争接收用原子操作，无 Mutex 锁竞争，
        // 200 线程可高效并行取任务。每个线程内部完全串行计算，等价于进程模型。
        let (task_tx, task_rx) = crossbeam::channel::unbounded::<(i64, String)>();

        // 核绑定：获取核 ID 列表
        let core_ids = if bind_cores { core_affinity::get_core_ids() } else { None };
        let cores_arc = core_ids.map(|c| std::sync::Arc::new(c));

        // 启动 worker 线程
        let mut handles = Vec::with_capacity(n_jobs);
        for worker_idx in 0..n_jobs {
            let task_rx = task_rx.clone();
            let params = hm90_params.clone();
            let trading_days = trading_days_arc.clone();
            let results_buf = results_buf.clone();
            let backup_file = backup_file_cloned.clone();
            let completed = completed.clone();
            let cores = cores_arc.clone();

            let handle = std::thread::spawn(move || {
                // 核绑定：worker_idx → 物理核 worker_idx
                if let Some(ref cores) = cores {
                    let target = worker_idx % cores.len();
                    let _ = core_affinity::set_for_current(cores[target].clone());
                }

                // 循环取任务并计算（完全串行，无内部并行）
                // crossbeam channel 的 recv 在关闭后返回 Err，自然退出循环
                while let Ok((date, code)) = task_rx.recv() {
                    let vals = pipeline_order_pair_hm90(
                        date,
                        &code,
                        &params,
                        &trading_days,
                        expected_result_length,
                    );

                    let result = TaskResult {
                        date,
                        code,
                        timestamp: 0,
                        facs: vals,
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
                        let _ = save_results_to_backup(&batch, &backup_file, expected_result_length);
                    }

                    // 进度日志
                    if progress_log_enabled {
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 500 == 0 || done == total {
                            let elapsed = start.elapsed().as_secs_f64();
                            let speed = done as f64 / elapsed;
                            println!(
                                "📊 进度: {}/{} ({:.1}%), 速度: {:.1} 任务/秒, 用时: {:.0}s",
                                done, total,
                                done as f64 / total as f64 * 100.0,
                                speed, elapsed
                            );
                        }
                    } else {
                        completed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            handles.push(handle);
        }

        // 分发所有任务到队列
        for task in pending {
            let _ = task_tx.send(task);
        }
        drop(task_tx); // 关闭发送端，worker recv 返回 Err 后退出

        // 等待所有 worker 完成
        for h in handles {
            let _ = h.join();
        }

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
        done, total, elapsed, done as f64 / elapsed
    );

    Ok(Python::with_gil(|py| py.None()))
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
        params: Hm90Params,
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
    Error { date: i64, code: String, msg: String },
    /// worker 就绪确认（收到 Init 后回复）
    Ready,
}

/// 写一个 IPC 消息到 writer（4字节长度前缀 + bincode）。
pub fn ipc_write<W: std::io::Write>(writer: &mut W, msg: &TaskMessage) -> std::io::Result<()> {
    let data = bincode::serialize(msg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let len = data.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&data)?;
    writer.flush()?;
    Ok(())
}

/// 写一个结果消息到 writer。
pub fn ipc_write_result<W: std::io::Write>(writer: &mut W, msg: &ResultMessage) -> std::io::Result<()> {
    let data = bincode::serialize(msg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
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
    let msg = bincode::deserialize(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(Some(msg))
}

/// 读一个结果消息。
pub fn ipc_read_result<R: std::io::Read>(reader: &mut R) -> std::io::Result<ResultMessage> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len == 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "收到长度0的结果（异常）"));
    }
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data)?;
    let msg = bincode::deserialize(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
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
#[allow(clippy::too_many_arguments)]
fn run_multiprocess_v2(
    py: Python<'_>,
    pending: Vec<(i64, String)>,
    n_jobs: usize,
    backup_file: String,
    expected_result_length: usize,
    trading_days: Vec<i64>,
    params: Hm90Params,
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
    let backup_file_arc = std::sync::Arc::new(backup_file.clone());

    py.allow_threads(|| -> PyResult<()> {
        // 任务队列（crossbeam MPMC，支持多消费者竞争 recv）
        let (task_tx, task_rx) = crossbeam::channel::unbounded::<(i64, String)>();
        for task in pending {
            let _ = task_tx.send(task);
        }
        drop(task_tx);

        // 结果 channel（worker 线程 → collector 线程）
        let (result_tx, result_rx) = std::sync::mpsc::channel::<TaskResult>();

        // 核绑定信息
        let core_ids = if bind_cores { core_affinity::get_core_ids() } else { None };

        // spawn N 个 worker 管理线程
        let mut handles = Vec::with_capacity(n_jobs);
        for worker_idx in 0..n_jobs {
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
            let params = params.clone();
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
                    &trading_days,
                    core_affinity_idx,
                );
            });
            handles.push(handle);
        }
        drop(result_tx); // 关闭后 collector 在所有 worker 线程结束后能退出

        // collector 线程：收集结果，批量写 backup
        let collector_handle = {
            let backup_file = backup_file_arc.clone();
            let expected_len = expected_result_length;
            let completed_cloned = completed.clone();
            std::thread::spawn(move || {
                let mut buf: Vec<TaskResult> = Vec::with_capacity(batch_size);
                while let Ok(result) = result_rx.recv() {
                    buf.push(result);
                    let _done = completed_cloned.fetch_add(1, Ordering::Relaxed) + 1;
                    if buf.len() >= batch_size {
                        let batch: Vec<TaskResult> = buf.drain(..).collect();
                        let _ = save_results_to_backup(&batch, &backup_file, expected_len);
                    }
                }
                // 写入剩余
                if !buf.is_empty() {
                    let _ = save_results_to_backup(&buf, &backup_file, expected_len);
                }
            })
        };

        // 等待所有 worker 管理线程完成
        for h in handles {
            let _ = h.join();
        }
        // result_tx 全部 drop 后 collector 的 recv 会返回 Err 并退出
        let _ = collector_handle.join();

        Ok(())
    })?;

    let done = completed.load(Ordering::Relaxed);
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "✅ [多进程] 完成 {} 任务, 总用时 {:.1}s, 平均速度 {:.1} 任务/秒",
        done, elapsed, done as f64 / elapsed
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
    result_tx: std::sync::mpsc::Sender<TaskResult>,
    params: &Hm90Params,
    trading_days: &[i64],
    core_affinity_idx: Option<usize>,
) {
    use std::io::{BufReader, BufWriter, Write};
    use std::process::{Child, Command, Stdio};

    // 外层循环：管理 worker 进程的生命周期（崩溃后重启）
    'outer: loop {
        // spawn worker 进程
        let mut cmd = Command::new(worker_bin);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
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
        let mut writer = BufWriter::new(stdin);
        let mut reader = BufReader::new(stdout);

        // 发送 Init 消息
        let init_msg = TaskMessage::Init {
            params: params.clone(),
            trading_days: trading_days.to_vec(),
            expected_len: 0, // 由 pipeline_order_pair_hm90 内部决定
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
            let task_msg = TaskMessage::Task { date, code: code.clone() };
            if ipc_write(&mut writer, &task_msg).is_err() {
                // IPC 错误：该任务记 NaN，重启 worker
                let _ = result_tx.send(TaskResult { date, code, timestamp: 0, facs: vec![f64::NAN] });
                let _ = child.kill();
                continue 'outer;
            }

            // 收结果
            match ipc_read_result(&mut reader) {
                Ok(ResultMessage::Result(r)) => {
                    let _ = result_tx.send(r);
                }
                Ok(ResultMessage::Error { date, code, msg }) => {
                    eprintln!("⚠️ worker{} 计算错误 [{},{}]: {}", worker_idx, date, code, msg);
                    let _ = result_tx.send(TaskResult { date, code, timestamp: 0, facs: vec![f64::NAN] });
                }
                Ok(ResultMessage::Ready) => {
                    // 忽略意外的 Ready
                }
                Err(_) => {
                    // IPC 错误，重启 worker
                    eprintln!("⚠️ worker{} IPC 错误，重启中...", worker_idx);
                    let _ = result_tx.send(TaskResult { date, code, timestamp: 0, facs: vec![f64::NAN] });
                    let _ = child.kill();
                    continue 'outer;
                }
            }
        }
    }
}
