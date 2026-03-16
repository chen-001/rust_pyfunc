use chrono::Local;
use crossbeam::channel::{unbounded, Receiver, Sender};
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[cfg(target_family = "unix")]
use nix::errno::Errno;
#[cfg(target_family = "unix")]
use nix::sys::signal::{kill, Signal};
#[cfg(target_family = "unix")]
use nix::unistd::Pid;

use crate::backup_reader::{
    read_existing_backup, read_existing_backup_with_filter, TaskResult,
};
use crate::parallel_computing::{
    DebugLogger, detect_python_interpreter, ensure_fd_limit, extract_python_function_code,
};

// ==================== 数据结构 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DateOnlyTaskParam {
    date: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct SingleDateOnlyTask {
    python_code: String,
    task: DateOnlyTaskParam,
    expected_result_length: usize,
}

// worker 返回的原始结果：一个date产出多条(code, facs)记录
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DateOnlyWorkerResult {
    date: i64,
    timestamp: i64,
    records: Vec<CodeFacs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodeFacs {
    code: String,
    facs: Vec<f64>,
}

// ==================== 简化版监控管理器 ====================

#[derive(Debug, Clone)]
struct DateOnlyWorkerMonitor {
    #[allow(dead_code)]
    worker_id: usize,
    last_heartbeat: Instant,
    current_date: Option<i64>,
    task_start_time: Option<Instant>,
    is_alive: bool,
    process_id: Option<u32>,
}

impl DateOnlyWorkerMonitor {
    fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            last_heartbeat: Instant::now(),
            current_date: None,
            task_start_time: None,
            is_alive: true,
            process_id: None,
        }
    }

    fn is_process_alive(&self) -> bool {
        if let Some(pid) = self.process_id {
            Path::new(&format!("/proc/{}", pid)).exists()
        } else {
            true
        }
    }

    fn is_stuck(&self, task_timeout: Duration, heartbeat_timeout: Duration) -> Option<&'static str> {
        if !self.is_process_alive() {
            return Some("process_death");
        }
        if self.last_heartbeat.elapsed() > heartbeat_timeout {
            return Some("heartbeat_timeout");
        }
        if let Some(start_time) = self.task_start_time {
            if start_time.elapsed() > task_timeout {
                return Some("task_timeout");
            }
        }
        None
    }
}

#[derive(Debug)]
struct DateOnlyMonitorManager {
    monitors: Arc<Mutex<HashMap<usize, DateOnlyWorkerMonitor>>>,
    task_timeout: Duration,
    health_check_interval: Duration,
    debug_monitor: bool,
    should_stop: Arc<AtomicBool>,
    stuck_dates: Arc<Mutex<Vec<(i64, usize, Duration, String)>>>,
    // 新增：发送失败的任务列表
    failed_dates: Arc<Mutex<Vec<i64>>>,
}

impl DateOnlyMonitorManager {
    fn new(task_timeout: Duration, health_check_interval: Duration, debug_monitor: bool) -> Self {
        Self {
            monitors: Arc::new(Mutex::new(HashMap::new())),
            task_timeout,
            health_check_interval,
            debug_monitor,
            should_stop: Arc::new(AtomicBool::new(false)),
            stuck_dates: Arc::new(Mutex::new(Vec::new())),
            failed_dates: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn add_worker(&self, worker_id: usize) {
        if let Ok(mut monitors) = self.monitors.lock() {
            monitors.insert(worker_id, DateOnlyWorkerMonitor::new(worker_id));
        }
    }

    fn set_worker_process_id(&self, worker_id: usize, pid: u32) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                monitor.process_id = Some(pid);
            }
        }
    }

    fn start_task(&self, worker_id: usize, date: i64) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                monitor.current_date = Some(date);
                monitor.task_start_time = Some(Instant::now());
            }
        }
    }

    fn finish_task(&self, worker_id: usize) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                monitor.current_date = None;
                monitor.task_start_time = None;
            }
        }
    }

    fn update_heartbeat(&self, worker_id: usize) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                monitor.last_heartbeat = Instant::now();
                monitor.is_alive = true;
            }
        }
    }

    fn check_stuck_workers(&self) -> Vec<(usize, &'static str)> {
        let heartbeat_timeout = self.health_check_interval * 3;
        let mut stuck_workers = Vec::new();

        if let Ok(monitors) = self.monitors.lock() {
            for (worker_id, monitor) in monitors.iter() {
                if !monitor.is_alive || monitor.process_id.is_none() {
                    continue;
                }
                if let Some(reason) = monitor.is_stuck(self.task_timeout, heartbeat_timeout) {
                    stuck_workers.push((*worker_id, reason));
                }
            }
        }

        stuck_workers
    }

    fn log_stuck_worker(&self, worker_id: usize, reason: &str) {
        if let Ok(monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get(&worker_id) {
                if let Some(date) = monitor.current_date {
                    let runtime = monitor
                        .task_start_time
                        .map(|t| t.elapsed())
                        .unwrap_or(Duration::ZERO);

                    if let Ok(mut stuck) = self.stuck_dates.lock() {
                        stuck.push((date, worker_id, runtime, reason.to_string()));
                    }
                }

                if self.debug_monitor {
                    println!(
                        "⚠️ Worker {} 卡死 (原因: {}), date={:?}",
                        worker_id, reason, monitor.current_date
                    );
                }
            }
        }
    }

    fn force_kill_worker(&self, worker_id: usize) -> bool {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                if let Some(pid) = monitor.process_id {
                    if !monitor.is_process_alive() {
                        drop(monitors);
                        self.remove_worker(worker_id);
                        return true;
                    }

                    #[cfg(target_family = "unix")]
                    {
                        match kill(Pid::from_raw(pid as i32), Signal::SIGKILL) {
                            Ok(()) => {
                                crate::parallel_computing::reap_process(pid);
                                monitor.process_id = None;
                                return true;
                            }
                            Err(err) => {
                                if err == Errno::ESRCH {
                                    drop(monitors);
                                    self.remove_worker(worker_id);
                                    return true;
                                }
                            }
                        }
                    }

                    #[cfg(not(target_family = "unix"))]
                    {
                        monitor.process_id = None;
                        return true;
                    }
                }
            }
        }
        false
    }

    fn remove_worker(&self, worker_id: usize) {
        if let Ok(mut monitors) = self.monitors.lock() {
            monitors.remove(&worker_id);
        }
    }

    fn stop_monitoring(&self) {
        self.should_stop.store(true, Ordering::SeqCst);
    }

    fn should_stop_monitoring(&self) -> bool {
        self.should_stop.load(Ordering::SeqCst)
    }

    fn terminate_all_workers(&self, graceful_timeout: Duration) {
        #[cfg(target_family = "unix")]
        {
            let targets: Vec<(usize, u32)> = match self.monitors.lock() {
                Ok(monitors) => monitors
                    .iter()
                    .filter_map(|(id, m)| m.process_id.map(|pid| (*id, pid)))
                    .collect(),
                Err(_) => Vec::new(),
            };

            for (_worker_id, pid) in targets {
                crate::parallel_computing::terminate_process(pid, graceful_timeout);
            }
        }

        #[cfg(not(target_family = "unix"))]
        {
            let _ = graceful_timeout;
        }
    }

    fn print_stuck_tasks_table(&self) {
        match self.stuck_dates.try_lock() {
            Ok(stuck) => {
                if stuck.is_empty() {
                    println!("\n✅ 没有任务因超时被跳过");
                } else {
                    println!("\n📋 卡死任务统计表");
                    println!("┌──────────┬─────────┬──────────────┬──────────────┐");
                    println!("│   Date   │ Worker  │   Runtime    │    Reason    │");
                    println!("├──────────┼─────────┼──────────────┼──────────────┤");
                    for (date, worker_id, runtime, reason) in stuck.iter() {
                        let runtime_str = if runtime.as_secs() > 0 {
                            format!("{:.1}s", runtime.as_secs_f64())
                        } else {
                            format!("{}ms", runtime.as_millis())
                        };
                        let reason_str = match reason.as_str() {
                            "task_timeout" => "任务超时",
                            "heartbeat_timeout" => "心跳超时",
                            "process_death" => "进程死亡",
                            _ => reason,
                        };
                        println!(
                            "│ {:8} │ {:7} │ {:12} │ {:12} │",
                            date, worker_id, runtime_str, reason_str
                        );
                    }
                    println!("└──────────┴─────────┴──────────────┴──────────────┘");
                    println!("共 {} 个任务因超时被跳过", stuck.len());
                }
            }
            Err(_) => {}
        }
    }

    // 记录发送失败的任务
    fn record_failed_date(&self, date: i64) {
        if let Ok(mut failed) = self.failed_dates.lock() {
            failed.push(date);
        }
    }

    // 获取所有失败的任务（包括卡死和发送失败）
    fn get_all_failed_dates(&self) -> Vec<i64> {
        let mut all_failed: Vec<i64> = Vec::new();
        
        // 从 stuck_dates 获取卡死的任务
        if let Ok(stuck) = self.stuck_dates.lock() {
            for (date, _, _, _) in stuck.iter() {
                all_failed.push(*date);
            }
        }
        
        // 从 failed_dates 获取发送失败的任务
        if let Ok(failed) = self.failed_dates.lock() {
            for date in failed.iter() {
                all_failed.push(*date);
            }
        }
        
        all_failed
    }

    fn cleanup(&self) {
        if let Ok(mut monitors) = self.monitors.try_lock() {
            monitors.clear();
        }
        if let Ok(mut stuck) = self.stuck_dates.try_lock() {
            stuck.clear();
        }
        if let Ok(mut failed) = self.failed_dates.try_lock() {
            failed.clear();
        }
    }
}

// ==================== Worker 脚本 ====================
// Python函数: func(date) -> list[pd.Series]
// Worker将其转置为: list[{code, facs}]
// 即：对所有Series取并集index，对每个code取各Series的值组成facs

fn create_date_only_worker_script() -> String {
    format!(
        r#"#!/usr/bin/env python3
import sys
import msgpack
import time
import struct
import math
import signal
import os
import traceback
import pandas as pd
import numpy as np

class WorkerHealthManager:
    def __init__(self):
        self.task_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.start_time = time.time()
        self.max_consecutive_errors = 5
        self.max_errors = 100
        self.max_memory_mb = 1024

    def record_task_success(self):
        self.task_count += 1
        self.consecutive_errors = 0

    def record_task_error(self):
        self.error_count += 1
        self.consecutive_errors += 1

    def should_restart(self):
        if self.consecutive_errors >= self.max_consecutive_errors:
            return True
        if self.error_count >= self.max_errors:
            return True
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > self.max_memory_mb:
                return True
        except ImportError:
            pass
        return False

health_manager = WorkerHealthManager()

def signal_handler(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def read_message_with_timeout(timeout=30):
    import select
    if not select.select([sys.stdin.buffer], [], [], timeout)[0]:
        return None
    length_bytes = sys.stdin.buffer.read(4)
    if len(length_bytes) != 4:
        return None
    length = struct.unpack('<I', length_bytes)[0]
    if length == 0:
        return None
    if length > 100 * 1024 * 1024:
        return None
    data = sys.stdin.buffer.read(length)
    if len(data) != length:
        return None
    return data

def write_message(data):
    length = len(data)
    length_bytes = struct.pack('<I', length)
    sys.stdout.buffer.write(length_bytes)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

def normalize_value(x):
    if x is None:
        return float('nan')
    try:
        val = float(x)
        if math.isinf(val) or math.isnan(val):
            return float('nan')
        return val
    except (ValueError, TypeError):
        return float('nan')

def transpose_series_list(series_list, expected_length):
    """将 list[pd.Series] 转置为 list[dict(code, facs)]
    
    series_list: 每个Series的index是股票代码, values是因子值
    expected_length: 期望的因子数量(=Series数量)
    
    对所有Series取并集index, 对每个code:
      facs[i] = series_list[i].get(code, nan)
    """
    # 收集所有code（按第一个Series的index顺序为主）
    all_codes = []
    seen = set()
    for s in series_list:
        if isinstance(s, pd.Series):
            for c in s.index:
                c_str = str(c)
                if c_str not in seen:
                    seen.add(c_str)
                    all_codes.append(c_str)
    
    records = []
    for code in all_codes:
        facs = []
        for i in range(expected_length):
            if i < len(series_list) and isinstance(series_list[i], pd.Series):
                s = series_list[i]
                if code in s.index:
                    facs.append(normalize_value(s.loc[code]))
                elif int(code) in s.index if code.isdigit() else False:
                    facs.append(normalize_value(s.loc[int(code)]))
                else:
                    facs.append(float('nan'))
            else:
                facs.append(float('nan'))
        records.append({{'code': code, 'facs': facs}})
    
    return records

def main():
    while True:
        try:
            if health_manager.should_restart():
                sys.exit(1)

            message_data = read_message_with_timeout(timeout=30)
            if message_data is None:
                break

            try:
                task_data = msgpack.unpackb(message_data, raw=False)
            except Exception as e:
                print(f"Failed to unpack message: {{e}}", file=sys.stderr)
                continue

            if not isinstance(task_data, dict):
                continue

            func_code = task_data['python_code']
            task = task_data['task']
            expected_length = task_data['expected_result_length']
            date = task['date']
            timestamp = int(time.time() * 1000)

            try:
                namespace = {{'__builtins__': __builtins__}}
                exec(func_code, namespace)

                user_functions = [name for name, obj in namespace.items()
                                 if callable(obj) and not name.startswith('_')]
                if not user_functions:
                    raise ValueError("No user function found")

                func = namespace[user_functions[0]]
                output = func(date)

                if not isinstance(output, list):
                    raise ValueError(f"Expected list[pd.Series], got {{type(output)}}")

                records = transpose_series_list(output, expected_length)
                health_manager.record_task_success()

            except Exception as e:
                print(f"Task error for date={{date}}: {{e}}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                records = []
                health_manager.record_task_error()

            result = {{
                'date': date,
                'timestamp': timestamp,
                'records': records,
            }}

            packed_output = msgpack.packb({{'result': result}}, use_bin_type=True)
            write_message(packed_output)

        except KeyboardInterrupt:
            break
        except IOError:
            break
        except Exception as e:
            print(f"Worker error: {{e}}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            health_manager.record_task_error()

            error_result = {{
                'result': {{
                    'date': 0,
                    'timestamp': int(time.time() * 1000),
                    'records': [],
                }}
            }}
            try:
                packed_error = msgpack.packb(error_result, use_bin_type=True)
                write_message(packed_error)
            except Exception:
                break

if __name__ == '__main__':
    main()
"#
    )
}

// ==================== Worker 函数 ====================

fn run_date_only_worker(
    worker_id: usize,
    task_queue: Receiver<DateOnlyTaskParam>,
    python_code: String,
    expected_result_length: usize,
    python_path: String,
    result_sender: Sender<TaskResult>,
    restart_flag: Arc<AtomicBool>,
    monitor_manager: Arc<DateOnlyMonitorManager>,
    debug_logger: DebugLogger,
) {
    monitor_manager.add_worker(worker_id);

    loop {
        if restart_flag
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            debug_logger.log_info(Some(worker_id), "RESTART", "Worker检测到重启信号");
        }

        let script_content = create_date_only_worker_script();
        let script_path = format!("/tmp/date_only_worker_{}.py", worker_id);

        if let Err(e) = std::fs::write(&script_path, &script_content) {
            debug_logger.log_error(Some(worker_id), "SCRIPT_CREATE", &format!("创建脚本失败: {}", e));
            continue;
        }

        let mut child = match Command::new(&python_path)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                debug_logger.log_error(Some(worker_id), "PROCESS_START", &format!("启动失败: {}", e));
                continue;
            }
        };

        let pid = child.id();
        monitor_manager.set_worker_process_id(worker_id, pid);
        monitor_manager.update_heartbeat(worker_id);

        let mut stdin = child.stdin.take().expect("Failed to get stdin");
        let mut stdout = child.stdout.take().expect("Failed to get stdout");
        let stderr = child.stderr.take().expect("Failed to get stderr");

        let stderr_logger = debug_logger.clone();
        let stderr_worker_id = worker_id;
        let stderr_handle = thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        stderr_logger.log_error(Some(stderr_worker_id), "PYTHON_STDERR", &text);
                    }
                    Err(_) => break,
                }
            }
        });

        let mut task_count = 0;
        let mut needs_restart = false;

        while let Ok(task) = task_queue.recv() {
            if restart_flag.load(Ordering::Relaxed) {
                needs_restart = true;
                break;
            }

            task_count += 1;
            monitor_manager.start_task(worker_id, task.date);
            monitor_manager.update_heartbeat(worker_id);

            let single_task = SingleDateOnlyTask {
                python_code: python_code.clone(),
                task: task.clone(),
                expected_result_length,
            };

            let packed_data = match rmp_serde::to_vec_named(&single_task) {
                Ok(data) => data,
                Err(e) => {
                    debug_logger.log_error(Some(worker_id), "SERIALIZE", &format!("date={} 序列化失败: {}", task.date, e));
                    continue;
                }
            };

            let length = packed_data.len() as u32;
            let length_bytes = length.to_le_bytes();

            if stdin.write_all(&length_bytes).is_err()
                || stdin.write_all(&packed_data).is_err()
                || stdin.flush().is_err()
            {
                debug_logger.log_error(Some(worker_id), "COMMUNICATION", &format!("发送任务失败, date={}", task.date));
                // 记录发送失败的任务，后续会重试
                monitor_manager.record_failed_date(task.date);
                needs_restart = true;
                break;
            }

            // 读取结果
            use std::io::Read;
            let mut len_buf = [0u8; 4];
            if stdout.read_exact(&mut len_buf).is_err() {
                debug_logger.log_error(Some(worker_id), "COMMUNICATION", "读取结果长度失败");
                needs_restart = true;
                break;
            }

            let result_len = u32::from_le_bytes(len_buf) as usize;
            let mut result_data = vec![0u8; result_len];

            if stdout.read_exact(&mut result_data).is_err() {
                debug_logger.log_error(Some(worker_id), "COMMUNICATION", "读取结果数据失败");
                needs_restart = true;
                break;
            }

            // 解析worker返回的结果
            #[derive(Debug, Serialize, Deserialize)]
            struct SingleDateOnlyResultWrapper {
                result: DateOnlyWorkerResult,
            }

            match rmp_serde::from_slice::<SingleDateOnlyResultWrapper>(&result_data) {
                Ok(wrapper) => {
                    let wr = wrapper.result;
                    // 将每个 (code, facs) 展开为独立的 TaskResult
                    for record in &wr.records {
                        let task_result = TaskResult {
                            date: wr.date,
                            code: record.code.clone(),
                            timestamp: wr.timestamp,
                            facs: record.facs.clone(),
                        };
                        if let Err(e) = result_sender.send(task_result) {
                            debug_logger.log_error(Some(worker_id), "RESULT_SEND", &format!("发送失败: {}", e));
                        }
                    }
                    monitor_manager.finish_task(worker_id);
                    monitor_manager.update_heartbeat(worker_id);
                }
                Err(e) => {
                    debug_logger.log_error(Some(worker_id), "DESERIALIZE", &format!("date={} 反序列化失败: {}", task.date, e));
                    monitor_manager.finish_task(worker_id);
                    monitor_manager.update_heartbeat(worker_id);
                }
            }
        }

        let _ = stdin.write_all(&[0u8; 4]);
        let _ = stdin.flush();
        let _ = stderr_handle.join();
        let _ = child.wait();
        let _ = std::fs::remove_file(&script_path);

        debug_logger.log_info(Some(worker_id), "PROCESS_END", &format!("共处理{}个任务", task_count));

        if !needs_restart {
            break;
        }
    }

    monitor_manager.remove_worker(worker_id);
}

// ==================== 备份写入(复用run_pools_queue的格式) ====================

fn save_results_to_backup(
    results: &[TaskResult],
    backup_file: &str,
    expected_result_length: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // 直接调用parallel_computing中相同逻辑的备份函数
    use crate::backup_reader::{calculate_record_size, DynamicRecord, FileHeader};
    use memmap2::MmapMut;
    use std::fs::OpenOptions;

    if results.is_empty() {
        return Ok(());
    }

    let factor_count = expected_result_length;
    let record_size = calculate_record_size(factor_count);
    let header_size = 64;

    let file_path = Path::new(backup_file);
    let file_exists = file_path.exists();
    let file_valid = if file_exists {
        file_path
            .metadata()
            .map(|m| m.len() >= header_size as u64)
            .unwrap_or(false)
    } else {
        false
    };

    if !file_valid {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(backup_file)?;

        let header = FileHeader {
            magic: *b"RPBACKUP",
            version: 2,
            record_count: 0,
            record_size: record_size as u32,
            factor_count: factor_count as u32,
            reserved: [0; 36],
        };

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const FileHeader as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        };

        file.write_all(header_bytes)?;
        file.flush()?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(backup_file)?;

    let file_len = file.metadata()?.len() as usize;
    if file_len < header_size {
        return Err(format!("File too small: {} < {}", file_len, header_size).into());
    }

    let mut header_bytes = [0u8; 64];
    use std::io::Read;
    file.read_exact(&mut header_bytes)?;

    let header = unsafe { &mut *(header_bytes.as_mut_ptr() as *mut FileHeader) };

    let file_factor_count = header.factor_count;
    if file_factor_count != factor_count as u32 {
        return Err(format!(
            "Factor count mismatch: file has {}, expected {}",
            file_factor_count, factor_count
        ).into());
    }

    let current_count = header.record_count;
    let new_count = current_count + results.len() as u64;

    let new_file_size = header_size as u64 + new_count * record_size as u64;
    file.set_len(new_file_size)?;

    drop(file);
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(backup_file)?;

    let mut mmap = unsafe { MmapMut::map_mut(&file)? };

    let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut FileHeader) };
    header.record_count = new_count;

    let start_offset = header_size + current_count as usize * record_size;

    for (i, result) in results.iter().enumerate() {
        let record = DynamicRecord::from_task_result(result);
        let record_bytes = record.to_bytes();
        let record_offset = start_offset + i * record_size;

        if record_bytes.len() != record_size {
            return Err(format!(
                "Record size mismatch: got {}, expected {}",
                record_bytes.len(), record_size
            ).into());
        }

        mmap[record_offset..record_offset + record_size].copy_from_slice(&record_bytes);
    }

    mmap.flush()?;

    Ok(())
}

// ==================== 主函数 ====================

#[pyfunction]
#[pyo3(signature = (python_function, dates, n_jobs, backup_file, expected_result_length, restart_interval=None, update_mode=None, task_timeout=None, health_check_interval=None, debug_monitor=None, backup_batch_size=None, debug_log=None))]
pub fn run_pools_queue_date_only(
    python_function: PyObject,
    dates: &PyList,
    n_jobs: usize,
    backup_file: String,
    expected_result_length: usize,
    restart_interval: Option<usize>,
    update_mode: Option<bool>,
    task_timeout: Option<u64>,
    health_check_interval: Option<u64>,
    debug_monitor: Option<bool>,
    backup_batch_size: Option<usize>,
    debug_log: Option<bool>,
) -> PyResult<PyObject> {
    let debug_log_enabled = debug_log.unwrap_or(false);
    let debug_logger = DebugLogger::new("run_pools_queue_date_only.log", debug_log_enabled)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("创建日志文件失败: {}", e)))?;

    let restart_interval_value = restart_interval.unwrap_or(200);
    if restart_interval_value == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "restart_interval must be greater than 0",
        ));
    }

    let update_mode_enabled = update_mode.unwrap_or(false);
    let task_timeout_secs = task_timeout.unwrap_or(120);
    let health_check_interval_secs = health_check_interval.unwrap_or(300);
    let debug_monitor_enabled = debug_monitor.unwrap_or(false);
    let backup_batch_size_value = backup_batch_size.unwrap_or(5000);

    let task_timeout_duration = Duration::from_secs(task_timeout_secs);
    let health_check_duration = Duration::from_secs(health_check_interval_secs);

    let desired_fd_limit = std::cmp::max(65_536_u64, (n_jobs as u64).saturating_mul(16));
    ensure_fd_limit(desired_fd_limit);

    // 解析日期参数
    let mut all_dates: Vec<i64> = Vec::new();
    for item in dates.iter() {
        let date: i64 = item.extract()?;
        all_dates.push(date);
    }

    // 读取已完成的任务（复用backup_reader的函数）
    // 备份文件中存储的是 (date, code) 对，所以只提取已完成的date集合
    let existing_tasks = if update_mode_enabled {
        let task_dates: HashSet<i64> = all_dates.iter().cloned().collect();
        read_existing_backup_with_filter(&backup_file, Some(&task_dates)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取备份失败: {}", e))
        })?
    } else {
        read_existing_backup(&backup_file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取备份失败: {}", e))
        })?
    };

    // 提取已完成的日期集合
    let existing_dates: HashSet<i64> = existing_tasks.iter().map(|(date, _)| *date).collect();

    let pending_dates: Vec<i64> = all_dates
        .iter()
        .filter(|d| !existing_dates.contains(d))
        .cloned()
        .collect();

    if pending_dates.is_empty() {
        println!(
            "[{}] ✅ 所有日期任务都已完成",
            Local::now().format("%Y-%m-%d %H:%M:%S")
        );
        return Python::with_gil(|py| Ok(py.None()));
    }

    let start_time = Instant::now();
    println!(
        "[{}] 📋 总日期数: {}, 待处理: {}, 已完成: {}",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        all_dates.len(),
        pending_dates.len(),
        existing_dates.len()
    );

    let python_code = extract_python_function_code(&python_function)?;
    let python_path = detect_python_interpreter();

    let (task_sender, task_receiver) = unbounded::<DateOnlyTaskParam>();
    let (result_sender, result_receiver) = unbounded::<TaskResult>();

    for date in &pending_dates {
        let _ = task_sender.send(DateOnlyTaskParam { date: *date });
    }
    drop(task_sender);

    let restart_flag = Arc::new(AtomicBool::new(false));

    let monitor_manager = Arc::new(DateOnlyMonitorManager::new(
        task_timeout_duration,
        health_check_duration,
        debug_monitor_enabled,
    ));

    println!(
        "[{}] 🚀 启动 {} 个worker处理 {} 个日期任务",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        n_jobs,
        pending_dates.len()
    );

    // 启动worker线程
    let mut worker_handles = Vec::new();
    for i in 0..n_jobs {
        let w_task_receiver = task_receiver.clone();
        let w_python_code = python_code.clone();
        let w_python_path = python_path.clone();
        let w_result_sender = result_sender.clone();
        let w_restart_flag = restart_flag.clone();
        let w_monitor = monitor_manager.clone();
        let w_logger = debug_logger.clone();
        let w_expected = expected_result_length;

        let handle = thread::spawn(move || {
            run_date_only_worker(
                i,
                w_task_receiver,
                w_python_code,
                w_expected,
                w_python_path,
                w_result_sender,
                w_restart_flag,
                w_monitor,
                w_logger,
            );
        });
        worker_handles.push(handle);
    }
    drop(result_sender);

    // 启动监控线程
    let monitor_clone = monitor_manager.clone();
    let monitor_restart_flag = restart_flag.clone();
    let monitor_debug_logger = debug_logger.clone();
    let monitor_handle = thread::spawn(move || {
        loop {
            if monitor_clone.should_stop_monitoring() {
                break;
            }

            if let Ok(monitors) = monitor_clone.monitors.lock() {
                if monitors.is_empty() {
                    break;
                }
            }

            let stuck_workers = monitor_clone.check_stuck_workers();
            if !stuck_workers.is_empty() {
                for (worker_id, reason) in stuck_workers {
                    monitor_clone.log_stuck_worker(worker_id, reason);
                    monitor_debug_logger.log_warn(Some(worker_id), "STUCK_DETECTED", &format!("原因: {}", reason));

                    if monitor_clone.force_kill_worker(worker_id) {
                        monitor_debug_logger.log_warn(Some(worker_id), "FORCE_KILL", "强制终止Worker");
                    }
                }
                monitor_restart_flag.store(true, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(100));
                monitor_restart_flag.store(false, Ordering::SeqCst);
            }

            for _ in 0..10 {
                if monitor_clone.should_stop_monitoring() {
                    break;
                }
                thread::sleep(monitor_clone.health_check_interval / 10);
            }
        }
    });

    // 启动结果收集器（复用 RPBACKUP 格式写入）
    let backup_file_clone = backup_file.clone();
    let collector_restart_flag = restart_flag.clone();
    let restart_interval_clone = restart_interval_value;
    let backup_batch_clone = backup_batch_size_value;
    let expected_clone = expected_result_length;
    let total_dates = pending_dates.len();
    let completed_dates_set = Arc::new(Mutex::new(HashSet::<i64>::new()));
    let completed_dates_clone = completed_dates_set.clone();
    let collector_handle = thread::spawn(move || {
        let mut batch_results: Vec<TaskResult> = Vec::new();
        let mut total_collected: usize = 0;
        let mut batch_count: usize = 0;
        let mut batch_count_this_chunk: usize = 0;

        println!(
            "[{}] 🔄 结果收集器启动...",
            Local::now().format("%Y-%m-%d %H:%M:%S")
        );

        while let Ok(result) = result_receiver.recv() {
            // 跟踪已完成的日期
            {
                if let Ok(mut dates_set) = completed_dates_clone.lock() {
                    dates_set.insert(result.date);
                }
            }
            total_collected += 1;
            batch_results.push(result);

            if batch_results.len() >= backup_batch_clone {
                batch_count += 1;
                batch_count_this_chunk += 1;

                let elapsed = start_time.elapsed();
                let elapsed_secs = elapsed.as_secs();
                let elapsed_h = elapsed_secs / 3600;
                let elapsed_m = (elapsed_secs % 3600) / 60;
                let elapsed_s = elapsed_secs % 60;

                // 计算进度和预计剩余时间
                let (completed_dates, remaining_h, remaining_m, remaining_s, progress_pct) = {
                    if let Ok(dates_set) = completed_dates_clone.lock() {
                        let completed = dates_set.len();
                        let remaining = total_dates.saturating_sub(completed);
                        let pct = (completed as f64 / total_dates as f64) * 100.0;
                        
                        if completed > 0 && remaining > 0 {
                            let elapsed_per_date = elapsed.as_secs_f64() / completed as f64;
                            let remaining_secs = (elapsed_per_date * remaining as f64) as u64;
                            (completed, remaining_secs / 3600, (remaining_secs % 3600) / 60, remaining_secs % 60, pct)
                        } else {
                            (completed, 0, 0, 0, pct)
                        }
                    } else {
                        (0, 0, 0, 0, 0.0)
                    }
                };

                let current_time = Local::now().format("%Y-%m-%d %H:%M:%S");
                print!(
                    "\r[{}] 💾 第 {} 次备份，已完成 {}/{} 日期 ({:.1}%), 已收集 {} 条记录, 已用{}h{}m{}s, 预计剩余{}h{}m{}s",
                    current_time,
                    batch_count, completed_dates, total_dates, progress_pct, total_collected,
                    elapsed_h, elapsed_m, elapsed_s,
                    remaining_h, remaining_m, remaining_s,
                );
                io::stdout().flush().unwrap();

                match save_results_to_backup(&batch_results, &backup_file_clone, expected_clone) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("❌ 第{}次备份失败: {}", batch_count, e);
                    }
                }
                batch_results.clear();

                if batch_count_this_chunk >= restart_interval_clone {
                    collector_restart_flag.store(true, Ordering::SeqCst);
                    batch_count_this_chunk = 0;
                }
            }
        }

        if !batch_results.is_empty() {
            batch_count += 1;
            println!(
                "\n[{}] 💾 保存最终剩余结果: {} 条",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                batch_results.len()
            );
            match save_results_to_backup(&batch_results, &backup_file_clone, expected_clone) {
                Ok(()) => println!("[{}] ✅ 最终备份成功！", Local::now().format("%Y-%m-%d %H:%M:%S")),
                Err(e) => eprintln!("❌ 最终备份失败: {}", e),
            }
        }

        println!(
            "[{}] 📊 收集器: 总收集 {} 条记录，{} 次备份",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            total_collected, batch_count
        );
    });

    // 等待所有worker完成
    for (i, handle) in worker_handles.into_iter().enumerate() {
        match handle.join() {
            Ok(()) => {}
            Err(e) => eprintln!("❌ Worker {} 异常: {:?}", i, e),
        }
    }

    monitor_manager.stop_monitoring();

    match monitor_handle.join() {
        Ok(()) => {}
        Err(e) => eprintln!("❌ 监控线程异常: {:?}", e),
    }

    match collector_handle.join() {
        Ok(()) => {
            println!("[{}] ✅ 结果收集器已完成", Local::now().format("%Y-%m-%d %H:%M:%S"));
            if let Ok(file) = std::fs::File::open(&backup_file) {
                let _ = file.sync_all();
            }
        }
        Err(e) => eprintln!("❌ 结果收集器异常: {:?}", e),
    }

    monitor_manager.print_stuck_tasks_table();
    
    // ==================== 重试失败的任务 ====================
    let failed_dates = monitor_manager.get_all_failed_dates();
    if !failed_dates.is_empty() {
        println!(
            "\n[{}] 🔄 开始重试 {} 个失败的任务...",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            failed_dates.len()
        );
        
        // 清理旧的监控状态
        monitor_manager.terminate_all_workers(Duration::from_secs(2));
        monitor_manager.cleanup();
        
        // 创建新的通道和重启标志
        let (retry_task_sender, retry_task_receiver) = unbounded::<DateOnlyTaskParam>();
        let (retry_result_sender, retry_result_receiver) = unbounded::<TaskResult>();
        let retry_restart_flag = Arc::new(AtomicBool::new(false));
        
        // 创建新的监控管理器
        let retry_monitor = Arc::new(DateOnlyMonitorManager::new(
            task_timeout_duration,
            health_check_duration,
            debug_monitor_enabled,
        ));
        
        // 发送失败的任务
        for date in &failed_dates {
            let _ = retry_task_sender.send(DateOnlyTaskParam { date: *date });
        }
        drop(retry_task_sender);
        
        println!(
            "[{}] 🚀 启动 {} 个worker重试 {} 个任务",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            n_jobs,
            failed_dates.len()
        );
        
        // 启动重试worker
        let mut retry_worker_handles = Vec::new();
        for i in 0..n_jobs {
            let w_task_receiver = retry_task_receiver.clone();
            let w_python_code = python_code.clone();
            let w_python_path = python_path.clone();
            let w_result_sender = retry_result_sender.clone();
            let w_restart_flag = retry_restart_flag.clone();
            let w_monitor = retry_monitor.clone();
            let w_logger = debug_logger.clone();
            let w_expected = expected_result_length;

            let handle = thread::spawn(move || {
                run_date_only_worker(
                    i,
                    w_task_receiver,
                    w_python_code,
                    w_expected,
                    w_python_path,
                    w_result_sender,
                    w_restart_flag,
                    w_monitor,
                    w_logger,
                );
            });
            retry_worker_handles.push(handle);
        }
        drop(retry_result_sender);
        
        // 启动重试监控线程
        let retry_monitor_clone = retry_monitor.clone();
        let retry_restart_flag_clone = retry_restart_flag.clone();
        let retry_monitor_handle = thread::spawn(move || {
            loop {
                if retry_monitor_clone.should_stop_monitoring() {
                    break;
                }
                if let Ok(monitors) = retry_monitor_clone.monitors.lock() {
                    if monitors.is_empty() {
                        break;
                    }
                }
                let stuck_workers = retry_monitor_clone.check_stuck_workers();
                if !stuck_workers.is_empty() {
                    for (worker_id, reason) in stuck_workers {
                        retry_monitor_clone.log_stuck_worker(worker_id, reason);
                        if retry_monitor_clone.force_kill_worker(worker_id) {
                            eprintln!("⚠️ 重试Worker {} 被强制终止 (原因: {})", worker_id, reason);
                        }
                    }
                    retry_restart_flag_clone.store(true, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(100));
                    retry_restart_flag_clone.store(false, Ordering::SeqCst);
                }
                for _ in 0..10 {
                    if retry_monitor_clone.should_stop_monitoring() {
                        break;
                    }
                    thread::sleep(retry_monitor_clone.health_check_interval / 10);
                }
            }
        });
        
        // 收集重试结果
        let retry_backup = backup_file.clone();
        let retry_expected = expected_result_length;
        let retry_collector_handle = thread::spawn(move || {
            let mut batch_results: Vec<TaskResult> = Vec::new();
            let mut total_collected: usize = 0;
            
            while let Ok(result) = retry_result_receiver.recv() {
                total_collected += 1;
                batch_results.push(result);
            }
            
            if !batch_results.is_empty() {
                println!(
                    "[{}] 💾 保存重试结果: {} 条",
                    Local::now().format("%Y-%m-%d %H:%M:%S"),
                    batch_results.len()
                );
                match save_results_to_backup(&batch_results, &retry_backup, retry_expected) {
                    Ok(()) => println!("[{}] ✅ 重试结果保存成功！", Local::now().format("%Y-%m-%d %H:%M:%S")),
                    Err(e) => eprintln!("❌ 重试结果保存失败: {}", e),
                }
            }
            
            println!("[{}] 📊 重试收集器: 共收集 {} 条记录", Local::now().format("%Y-%m-%d %H:%M:%S"), total_collected);
            total_collected
        });
        
        // 等待重试worker完成
        for handle in retry_worker_handles {
            let _ = handle.join();
        }
        retry_monitor.stop_monitoring();
        let _ = retry_monitor_handle.join();
        let retry_collected = retry_collector_handle.join().unwrap_or(0);
        
        retry_monitor.print_stuck_tasks_table();
        
        // 计算最终统计
        let still_failed_count: usize = failed_dates.len().saturating_sub(retry_collected);
        
        println!(
            "\n📊 重试统计: 原失败 {} 个，成功重试 {} 个，仍有 {} 个失败",
            failed_dates.len(),
            retry_collected,
            still_failed_count
        );
        
        if still_failed_count > 0 {
            eprintln!("⚠️ 仍有 {} 个任务失败，可通过 update_mode=True 再次运行来重试", still_failed_count);
        }
        
        retry_monitor.terminate_all_workers(Duration::from_secs(2));
        
        // 清理重试worker脚本
        for i in 0..n_jobs {
            let script_path = format!("/tmp/date_only_worker_{}.py", i);
            if Path::new(&script_path).exists() {
                let _ = std::fs::remove_file(&script_path);
            }
        }
    } else {
        monitor_manager.terminate_all_workers(Duration::from_secs(2));
        monitor_manager.cleanup();
    }
    
    drop(monitor_manager);

    thread::sleep(Duration::from_millis(100));

    for i in 0..n_jobs {
        let script_path = format!("/tmp/date_only_worker_{}.py", i);
        if Path::new(&script_path).exists() {
            let _ = std::fs::remove_file(&script_path);
        }
    }

    println!("✅ 任务完成");
    Python::with_gil(|py| Ok(py.None()))
}
