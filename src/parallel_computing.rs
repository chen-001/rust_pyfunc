use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use memmap2::MmapMut;
use std::process::{Command, Stdio, Child};
use std::thread;
use crossbeam::channel::{unbounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering, AtomicUsize};
use chrono::Local;
use std::sync::Mutex;
use std::collections::{HashMap, VecDeque};

// 导入备份相关模块
use crate::backup_reader::{
    read_existing_backup, read_existing_backup_with_filter, 
    TaskResult, read_backup_results_with_filter, read_backup_results
};

// 通用结果结构体，用于反序列化单个任务结果
#[derive(Debug, Serialize, Deserialize)]
struct SingleResult {
    result: TaskResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParam {
    pub date: i64,
    pub code: String,
}

// 旧的批处理结构体已删除，只保留单任务结构体
#[derive(Debug, Serialize, Deserialize)]
struct SingleTask {
    python_code: String,
    task: TaskParam,
    expected_result_length: usize,
}

// 新增：Worker监控信息
#[derive(Debug, Clone)]
struct WorkerMonitor {
    worker_id: usize,
    last_heartbeat: Instant,
    last_successful_task: Option<Instant>,
    current_task: Option<TaskParam>,
    task_start_time: Option<Instant>,
    is_alive: bool,
    consecutive_failures: u32,
    process_id: Option<u32>,  // 子进程ID，用于进程存活检测
    consecutive_heartbeats_missed: u32,  // 连续错过的心跳次数
    total_tasks_completed: u32,  // 完成的任务总数
}

impl WorkerMonitor {
    fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            last_heartbeat: Instant::now(),
            last_successful_task: None,
            current_task: None,
            task_start_time: None,
            is_alive: true,
            consecutive_failures: 0,
            process_id: None,
            consecutive_heartbeats_missed: 0,
            total_tasks_completed: 0,
        }
    }
    
    fn start_task(&mut self, task: TaskParam) {
        self.current_task = Some(task);
        self.task_start_time = Some(Instant::now());
    }
    
    fn finish_task(&mut self) {
        self.current_task = None;
        self.task_start_time = None;
        self.last_successful_task = Some(Instant::now());
        self.consecutive_failures = 0;  // 重置失败计数
        self.consecutive_heartbeats_missed = 0;  // 重置心跳计数
        self.total_tasks_completed += 1;  // 增加完成任务计数
    }
    
    fn update_heartbeat(&mut self) {
        let now = Instant::now();
        self.last_heartbeat = now;
        self.is_alive = true;

        // 如果心跳间隔合理，重置连续错过心跳计数
        if let Some(last_heartbeat) = self.last_successful_task {
            if now.duration_since(last_heartbeat) < Duration::from_secs(300) {
                self.consecutive_heartbeats_missed = 0;
            }
        }
    }
    
    fn set_process_id(&mut self, pid: u32) {
        self.process_id = Some(pid);
    }
    
    fn is_process_alive(&self) -> bool {
        if let Some(pid) = self.process_id {
            // 检查/proc/PID目录是否存在（适用于Linux和大多数Unix系统）
            std::path::Path::new(&format!("/proc/{}", pid)).exists()
        } else {
            true  // 如果没有进程ID，假设进程存活
        }
    }
    
    fn is_stuck(&self, task_timeout: Duration, heartbeat_timeout: Duration) -> Option<&'static str> {
        // 首先检查进程是否还活着
        if !self.is_process_alive() {
            return Some("process_death");
        }

        // 检查心跳超时（更严格的检测）
        let heartbeat_elapsed = self.last_heartbeat.elapsed();
        if heartbeat_elapsed > heartbeat_timeout {
            // 如果连续错过多次心跳，才认为是卡死
            if self.consecutive_heartbeats_missed > 3 {
                return Some("heartbeat_timeout");
            }
        }

        // 检查任务执行超时
        if let Some(start_time) = self.task_start_time {
            let task_elapsed = start_time.elapsed();
            if task_elapsed > task_timeout {
                return Some("task_timeout");
            }
        }

        // 检查是否有长时间无任何活动
        let last_activity = self.last_successful_task
            .or(self.task_start_time)
            .unwrap_or(self.last_heartbeat);

        if last_activity.elapsed() > Duration::from_secs(600) { // 10分钟无活动
            return Some("inactivity_timeout");
        }

        // 检查连续失败次数
        if self.consecutive_failures > 5 {
            return Some("too_many_failures");
        }

        None
    }
}

// 新增：诊断统计信息
#[derive(Debug, Clone)]
struct DiagnosticStats {
    total_stuck_detections: u32,
    total_force_kills: u32,
    total_restarts: u32,
    stuck_by_timeout: u32,
    stuck_by_heartbeat: u32,
    stuck_by_process_death: u32,
    stuck_by_inactivity: u32,
    stuck_by_failures: u32,
}

impl DiagnosticStats {
    fn new() -> Self {
        Self {
            total_stuck_detections: 0,
            total_force_kills: 0,
            total_restarts: 0,
            stuck_by_timeout: 0,
            stuck_by_heartbeat: 0,
            stuck_by_process_death: 0,
            stuck_by_inactivity: 0,
            stuck_by_failures: 0,
        }
    }
}

// 卡死任务信息结构体
#[derive(Debug, Clone)]
struct StuckTaskInfo {
    date: i64,
    code: String,
    worker_id: usize,
    runtime: Duration,
    reason: String,
}

// 进程池结构体 - 优化版本
#[derive(Debug)]
struct ProcessPool {
    processes: Vec<ProcessWrapper>,
    available_processes: Arc<Mutex<VecDeque<usize>>>,
    next_process: AtomicUsize,
    pool_size: usize,
}

#[derive(Debug)]
struct ProcessWrapper {
    child: Mutex<Option<Child>>,
    stdin: Arc<Mutex<std::process::ChildStdin>>,
    stdout: Arc<Mutex<std::process::ChildStdout>>,
    pid: u32,
    is_alive: AtomicBool,
}

impl ProcessPool {
    fn new(pool_size: usize, python_path: &str, script_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut processes = Vec::with_capacity(pool_size);
        let mut available = VecDeque::with_capacity(pool_size);
        
        println!("🏭 创建进程池: {} 个Python进程", pool_size);
        
        for i in 0..pool_size {
            let mut child = Command::new(python_path)
                .arg(script_path)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            
            let pid = child.id();
            let stdin = Arc::new(Mutex::new(child.stdin.take().unwrap()));
            let stdout = Arc::new(Mutex::new(child.stdout.take().unwrap()));
            
            let process = ProcessWrapper {
                child: Mutex::new(Some(child)),
                stdin,
                stdout,
                pid,
                is_alive: AtomicBool::new(true),
            };
            
            processes.push(process);
            available.push_back(i);
            
            if (i + 1) % 10 == 0 || i + 1 == pool_size {
                // println!("🔄 已创建 {}/{} 个进程", i + 1, pool_size);
            }
        }
        
        Ok(ProcessPool {
            processes,
            available_processes: Arc::new(Mutex::new(available)),
            next_process: AtomicUsize::new(0),
            pool_size,
        })
    }
    
    fn get_next_process(&self) -> usize {
        // 使用轮询方式选择进程，避免锁竞争
        self.next_process.fetch_add(1, Ordering::Relaxed) % self.pool_size
    }
    
    fn execute_task(&self, process_id: usize, task_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let process = &self.processes[process_id];

        if !process.is_alive.load(Ordering::Relaxed) {
            return Err("Process is not alive".into());
        }

        // 设置超时时间
        let io_timeout = Duration::from_secs(30); // 30秒超时

        // 发送任务
        {
            let mut stdin = process.stdin.lock().unwrap();
            let length = task_data.len() as u32;

            // 添加写入超时
            let write_start = Instant::now();
            let write_result: Result<(), std::io::Error> = (|| {
                stdin.write_all(&length.to_le_bytes())?;
                stdin.write_all(task_data)?;
                stdin.flush()?;
                Ok(())
            })();

            if write_result.is_err() {
                let elapsed = write_start.elapsed();
                if elapsed > io_timeout {
                    return Err("写入任务超时".into());
                } else {
                    return Err(format!("写入任务失败: {}", write_result.unwrap_err()).into());
                }
            }
        }

        // 读取结果
        let mut stdout = process.stdout.lock().unwrap();
        let mut length_bytes = [0u8; 4];
        use std::io::Read;

        // 添加读取长度超时
        let read_length_start = Instant::now();
        let read_length_result = stdout.read_exact(&mut length_bytes);

        if let Err(e) = read_length_result {
            let elapsed = read_length_start.elapsed();
            if elapsed > io_timeout {
                return Err("读取长度超时".into());
            } else {
                return Err(format!("读取长度失败: {}", e).into());
            }
        }

        let length = u32::from_le_bytes(length_bytes) as usize;

        // 验证长度合理性
        if length > 100 * 1024 * 1024 { // 100MB限制
            return Err(format!("返回数据过大: {} bytes", length).into());
        }

        let mut result_data = vec![0u8; length];

        // 添加读取数据超时
        let read_data_start = Instant::now();
        let read_data_result = stdout.read_exact(&mut result_data);

        if let Err(e) = read_data_result {
            let elapsed = read_data_start.elapsed();
            if elapsed > io_timeout {
                return Err("读取数据超时".into());
            } else {
                return Err(format!("读取数据失败: {}", e).into());
            }
        }

        Ok(result_data)
    }
    
    fn shutdown(&self) {
        println!("🔚 关闭进程池...");
        for (i, process) in self.processes.iter().enumerate() {
            // 发送结束信号
            if let Ok(mut stdin) = process.stdin.lock() {
                let _ = stdin.write_all(&[0u8; 4]);
                let _ = stdin.flush();
            }
            
            // 等待进程结束
            if let Ok(mut child_opt) = process.child.lock() {
                if let Some(mut child) = child_opt.take() {
                    let _ = child.wait();
                }
            }
            
            process.is_alive.store(false, Ordering::Relaxed);
            // println!("✅ 进程 {} 已关闭", i);
        }
    }
}

// 新增：Worker监控管理器
#[derive(Debug)]
struct WorkerMonitorManager {
    monitors: Arc<Mutex<HashMap<usize, WorkerMonitor>>>,
    task_timeout: Duration,
    health_check_interval: Duration,
    debug_monitor: bool,
    stats: Arc<Mutex<DiagnosticStats>>,
    should_stop: Arc<AtomicBool>,
    stuck_tasks: Arc<Mutex<Vec<StuckTaskInfo>>>,
}

impl WorkerMonitorManager {
    fn new(task_timeout: Duration, health_check_interval: Duration, debug_monitor: bool) -> Self {
        Self {
            monitors: Arc::new(Mutex::new(HashMap::new())),
            task_timeout,
            health_check_interval,
            debug_monitor,
            stats: Arc::new(Mutex::new(DiagnosticStats::new())),
            should_stop: Arc::new(AtomicBool::new(false)),
            stuck_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    fn add_worker(&self, worker_id: usize) {
        if let Ok(mut monitors) = self.monitors.lock() {
            monitors.insert(worker_id, WorkerMonitor::new(worker_id));
            if self.debug_monitor {
                println!("🔍 监控器: 添加worker {}", worker_id);
            }
        }
    }
    
    fn set_worker_process_id(&self, worker_id: usize, pid: u32) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                monitor.set_process_id(pid);
                if self.debug_monitor {
                    println!("🔍 监控器: Worker {} 设置进程ID: {}", worker_id, pid);
                }
            }
        }
    }
    
    fn start_task(&self, worker_id: usize, task: TaskParam) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                monitor.start_task(task.clone());
                if self.debug_monitor {
                    println!("🔍 监控器: Worker {} 开始任务 date={}, code={}", 
                             worker_id, task.date, task.code);
                }
            }
        }
    }
    
    fn finish_task(&self, worker_id: usize) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                if self.debug_monitor && monitor.current_task.is_some() {
                    let task = monitor.current_task.as_ref().unwrap();
                    println!("🔍 监控器: Worker {} 完成任务 date={}, code={}", 
                             worker_id, task.date, task.code);
                }
                monitor.finish_task();
            }
        }
    }
    
    fn update_heartbeat(&self, worker_id: usize) {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                monitor.update_heartbeat();
            }
        }
    }
    
    fn check_stuck_workers(&self) -> Vec<(usize, &'static str)> {
        let heartbeat_timeout = self.health_check_interval * 3; // 3个检查周期无响应视为卡死
        let mut stuck_workers = Vec::new();

        if let Ok(mut monitors) = self.monitors.lock() {
            for (worker_id, monitor) in monitors.iter_mut() {
                // 跳过已经标记为不存活或没有进程ID的worker
                if !monitor.is_alive || monitor.process_id.is_none() {
                    continue;
                }

                // 检查心跳状态
                let heartbeat_elapsed = monitor.last_heartbeat.elapsed();
                if heartbeat_elapsed > self.health_check_interval {
                    monitor.consecutive_heartbeats_missed += 1;
                } else {
                    monitor.consecutive_heartbeats_missed = 0;
                }

                if let Some(stuck_reason) = monitor.is_stuck(self.task_timeout, heartbeat_timeout) {
                    stuck_workers.push((*worker_id, stuck_reason));

                    // 更新统计信息
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.total_stuck_detections += 1;
                        match stuck_reason {
                            "task_timeout" => stats.stuck_by_timeout += 1,
                            "heartbeat_timeout" => stats.stuck_by_heartbeat += 1,
                            "process_death" => stats.stuck_by_process_death += 1,
                            "inactivity_timeout" => stats.stuck_by_inactivity += 1,
                            "too_many_failures" => stats.stuck_by_failures += 1,
                            _ => {}
                        }
                    }

                    if self.debug_monitor {
                        println!("⚠️ 监控器: 检测到Worker {} 卡死 (原因: {})", worker_id, stuck_reason);
                        if let Some(task) = &monitor.current_task {
                            println!("   正在处理任务: date={}, code={}", task.date, task.code);
                        }
                        println!("   最后心跳: {:?}前", monitor.last_heartbeat.elapsed());
                        println!("   连续错过心跳: {} 次", monitor.consecutive_heartbeats_missed);
                        println!("   完成任务数: {}", monitor.total_tasks_completed);
                        if let Some(start_time) = monitor.task_start_time {
                            println!("   任务运行时间: {:?}", start_time.elapsed());
                        }
                    }
                }
            }
        }

        stuck_workers
    }
    
    fn log_stuck_worker(&self, worker_id: usize, reason: &str) {
        if let Ok(monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get(&worker_id) {
                // 只在debug模式下输出详细信息
                if self.debug_monitor {
                    println!("🚨 Worker {} 被标记为卡死并将重启", worker_id);
                    if let Some(task) = &monitor.current_task {
                        println!("   跳过任务: date={}, code={} (已运行 {:?})", 
                                 task.date, task.code,
                                 monitor.task_start_time.map(|t| t.elapsed()).unwrap_or(Duration::ZERO));
                    }
                    println!("   最后心跳时间: {:?}前", monitor.last_heartbeat.elapsed());
                    if let Some(pid) = monitor.process_id {
                        println!("   进程ID: {}", pid);
                    }
                }
                
                // 记录卡死任务信息
                if let Some(task) = &monitor.current_task {
                    let stuck_task = StuckTaskInfo {
                        date: task.date,
                        code: task.code.clone(),
                        worker_id,
                        runtime: monitor.task_start_time.map(|t| t.elapsed()).unwrap_or(Duration::ZERO),
                        reason: reason.to_string(),
                    };
                    
                    if let Ok(mut stuck_tasks) = self.stuck_tasks.lock() {
                        stuck_tasks.push(stuck_task);
                    }
                }
            }
        }
    }
    
    fn force_kill_worker(&self, worker_id: usize) -> bool {
        if let Ok(mut monitors) = self.monitors.lock() {
            if let Some(monitor) = monitors.get_mut(&worker_id) {
                if let Some(pid) = monitor.process_id {
                    // 首先检查进程是否仍然存在
                    if !monitor.is_process_alive() {
                        if self.debug_monitor {
                            println!("🔍 Worker {} 进程 {} 已不存在，清理监控记录", worker_id, pid);
                        }
                        // 直接移除整个监控记录
                        drop(monitors);  // 释放锁
                        self.remove_worker(worker_id);
                        return true;
                    }
                    
                    if self.debug_monitor {
                        println!("🔥 强制终止Worker {} 进程 (PID: {})", worker_id, pid);
                    }
                    
                    // 尝试强制杀死进程
                    #[cfg(target_os = "linux")]
                    {
                        use std::process::Command;
                        let output = Command::new("kill")
                            .arg("-9")  // SIGKILL
                            .arg(pid.to_string())
                            .output();
                        
                        match output {
                            Ok(result) => {
                                if result.status.success() {
                                    // println!("✅ 成功终止进程 {}", pid);
                                    monitor.process_id = None;  // 清除进程ID
                                    
                                    // 更新统计信息
                                    if let Ok(mut stats) = self.stats.lock() {
                                        stats.total_force_kills += 1;
                                    }
                                    
                                    return true;
                                } else {
                                    let stderr = String::from_utf8_lossy(&result.stderr);
                                    // 如果是"No such process"错误，说明进程已经不存在了
                                    if stderr.contains("No such process") {
                                        if self.debug_monitor {
                                            println!("🔍 进程 {} 已不存在，清理监控记录", pid);
                                        }
                                        // 直接移除整个监控记录
                                        drop(monitors);  // 释放锁
                                        self.remove_worker(worker_id);
                                        return true;
                                    } else {
                                        eprintln!("❌ 终止进程失败: {}", stderr);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("❌ 执行kill命令失败: {}", e);
                            }
                        }
                    }
                    
                    // 非Linux系统的处理（简化）
                    #[cfg(not(target_os = "linux"))]
                    {
                        println!("⚠️ 非Linux系统，无法强制终止进程 {}", pid);
                        monitor.process_id = None;  // 清除进程ID，假设进程已死
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
            if self.debug_monitor {
                println!("🔍 监控器: 移除worker {}", worker_id);
            }
        }
    }
    
    fn stop_monitoring(&self) {
        self.should_stop.store(true, Ordering::SeqCst);
        if self.debug_monitor {
            println!("🔍 监控器: 接收到停止信号");
        }
    }
    
    fn should_stop_monitoring(&self) -> bool {
        self.should_stop.load(Ordering::SeqCst)
    }
    
    fn print_diagnostic_stats(&self) {
        // 使用try_lock避免无限等待
        match self.stats.try_lock() {
            Ok(stats) => {
                if stats.total_stuck_detections > 0 {
                    println!("\n📊 监控器诊断统计:");
                    println!("   总卡死检测次数: {}", stats.total_stuck_detections);
                    println!("   任务超时导致: {}", stats.stuck_by_timeout);
                    println!("   心跳超时导致: {}", stats.stuck_by_heartbeat);
                    println!("   进程死亡导致: {}", stats.stuck_by_process_death);
                    println!("   长时间无活动导致: {}", stats.stuck_by_inactivity);
                    println!("   失败次数过多导致: {}", stats.stuck_by_failures);
                    println!("   强制终止次数: {}", stats.total_force_kills);
                    println!("   重启次数: {}", stats.total_restarts);
                } else {
                    println!("[{}] 📊 监控器统计: 未检测到任何worker卡死", Local::now().format("%Y-%m-%d %H:%M:%S"));
                }
            }
            Err(_) => {
                println!("⚠️ 无法获取诊断统计锁，跳过统计输出");
            }
        }
    }
    
    fn print_stuck_tasks_table(&self) {
        // 使用try_lock避免无限等待，并添加错误处理
        match self.stuck_tasks.try_lock() {
            Ok(stuck_tasks) => {
                if stuck_tasks.is_empty() {
                    println!("\n✅ 没有任务因超时被跳过");
                } else {
                    println!("\n📋 卡死任务统计表");
                    println!("┌──────────┬──────────┬─────────┬──────────────┬──────────────┐");
                    println!("│   Date   │   Code   │ Worker  │   Runtime    │    Reason    │");
                    println!("├──────────┼──────────┼─────────┼──────────────┼──────────────┤");
                    
                    for task in stuck_tasks.iter() {
                        let runtime_str = if task.runtime.as_secs() > 0 {
                            format!("{:.1}s", task.runtime.as_secs_f64())
                        } else {
                            format!("{}ms", task.runtime.as_millis())
                        };
                        
                        println!("│ {:8} │ {:8} │ {:7} │ {:12} │ {:12} │",
                            task.date,
                            task.code,
                            task.worker_id,
                            runtime_str,
                            match task.reason.as_str() {
                                "task_timeout" => "任务超时",
                                "heartbeat_timeout" => "心跳超时",
                                "process_death" => "进程死亡",
                                "inactivity_timeout" => "无活动超时",
                                "too_many_failures" => "失败次数过多",
                                _ => &task.reason
                            }
                        );
                    }
                    
                    println!("└──────────┴──────────┴─────────┴──────────────┴──────────────┘");
                    println!("共 {} 个任务因超时被跳过", stuck_tasks.len());
                }
            }
            Err(_) => {
                println!("⚠️ 无法获取卡死任务统计锁，跳过统计表打印");
            }
        }
    }
    
    /// 清理监控管理器的所有资源，确保没有遗留引用
    fn cleanup(&self) {
        if self.debug_monitor {
            println!("🧹 监控器: 开始清理资源...");
        }
        
        // 清理所有monitor记录
        if let Ok(mut monitors) = self.monitors.try_lock() {
            monitors.clear();
            if self.debug_monitor {
                println!("🧹 监控器: 已清理所有worker监控记录");
            }
        } else if self.debug_monitor {
            println!("⚠️ 监控器: 无法获取monitors锁进行清理");
        }
        
        // 清理卡死任务记录
        if let Ok(mut stuck_tasks) = self.stuck_tasks.try_lock() {
            stuck_tasks.clear();
            if self.debug_monitor {
                println!("🧹 监控器: 已清理所有卡死任务记录");
            }
        } else if self.debug_monitor {
            println!("⚠️ 监控器: 无法获取stuck_tasks锁进行清理");
        }
        
        // 重置统计信息
        if let Ok(mut stats) = self.stats.try_lock() {
            *stats = DiagnosticStats::new();
            if self.debug_monitor {
                println!("🧹 监控器: 已重置诊断统计信息");
            }
        } else if self.debug_monitor {
            println!("⚠️ 监控器: 无法获取stats锁进行清理");
        }
        
        if self.debug_monitor {
            println!("✅ 监控器: 资源清理完成");
        }
    }
}

fn detect_python_interpreter() -> String {
    // 1. 检查环境变量
    if let Ok(python_path) = env::var("PYTHON_INTERPRETER") {
        if Path::new(&python_path).exists() {
            return python_path;
        }
    }
    
    // 2. 检查是否在 conda 环境中
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda_python = format!("{}/bin/python", conda_prefix);
        if Path::new(&conda_python).exists() {
            return conda_python;
        }
    }
    
    // 3. 检查虚拟环境
    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        let venv_python = format!("{}/bin/python", virtual_env);
        if Path::new(&venv_python).exists() {
            return venv_python;
        }
    }
    
    // 4. 尝试常见的 Python 解释器
    let candidates = ["python3", "python"];
    for candidate in &candidates {
        if Command::new("which")
            .arg(candidate)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
        {
            return candidate.to_string();
        }
    }
    
    // 5. 默认值
    "python".to_string()
}

// 保留备份保存功能，但使用来自backup_reader的结构体定义
fn save_results_to_backup(results: &[TaskResult], backup_file: &str, expected_result_length: usize) -> Result<(), Box<dyn std::error::Error>> {
    use crate::backup_reader::{DynamicRecord, FileHeader, calculate_record_size, calculate_hash};
    
    if results.is_empty() {
        return Ok(());
    }
    
    let factor_count = expected_result_length;
    let record_size = calculate_record_size(factor_count);
    let header_size = 64; // HEADER_SIZE from backup_reader
    
    // 检查文件是否存在且有效
    let file_path = Path::new(backup_file);
    let file_exists = file_path.exists();
    let file_valid = if file_exists {
        file_path.metadata().map(|m| m.len() >= header_size as u64).unwrap_or(false)
    } else {
        false
    };
    
    if !file_valid {
        // 创建新文件，写入文件头
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)  
            .open(backup_file)?;
        
        let header = FileHeader {
            magic: *b"RPBACKUP",
            version: 2,  // 版本2表示支持动态因子数量
            record_count: 0,
            record_size: record_size as u32,
            factor_count: factor_count as u32,
            reserved: [0; 36],
        };
        
        // 写入文件头
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const FileHeader as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        };
        
        file.write_all(header_bytes)?;
        file.flush()?;
    }
    
    // 读取当前记录数
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(backup_file)?;
    
    let file_len = file.metadata()?.len() as usize;
    if file_len < header_size {
        return Err(format!("File is too small to contain valid header: {} < {}", file_len, header_size).into());
    }
    
    let mut header_bytes = [0u8; 64];
    use std::io::Read;
    file.read_exact(&mut header_bytes)?;
    
    let header = unsafe {
        &mut *(header_bytes.as_mut_ptr() as *mut FileHeader)
    };
    
    // 验证因子数量匹配
    let file_factor_count = header.factor_count;
    if file_factor_count != factor_count as u32 {
        return Err(format!("Factor count mismatch: file has {}, expected {}", file_factor_count, factor_count).into());
    }
    
    let current_count = header.record_count;
    let new_count = current_count + results.len() as u64;
    
    // 扩展文件大小
    let new_file_size = header_size as u64 + new_count * record_size as u64;
    file.set_len(new_file_size)?;
    
    // 使用内存映射进行高速写入
    drop(file);
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(backup_file)?;
    
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    
    // 更新文件头中的记录数量
    let header = unsafe {
        &mut *(mmap.as_mut_ptr() as *mut FileHeader)
    };
    header.record_count = new_count;
    
    // 写入新记录
    let start_offset = header_size + current_count as usize * record_size;
    
    for (i, result) in results.iter().enumerate() {
        let record = DynamicRecord::from_task_result(result);
        let record_bytes = record.to_bytes();
        let record_offset = start_offset + i * record_size;
        
        // 确保记录大小正确
        if record_bytes.len() != record_size {
            return Err(format!("Record size mismatch: got {}, expected {}", record_bytes.len(), record_size).into());
        }
        
        mmap[record_offset..record_offset + record_size].copy_from_slice(&record_bytes);
    }
    
    mmap.flush()?;
    
    Ok(())
}

fn create_persistent_worker_script() -> String {
    format!(r#"#!/usr/bin/env python3
import sys
import msgpack
import time
import struct
import math
import signal
import os
import traceback

class WorkerHealthManager:
    """Worker健康状态管理器"""
    def __init__(self):
        self.task_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.max_consecutive_errors = 5
        self.max_errors = 100
        self.health_check_interval = 60  # 60秒
        self.max_memory_mb = 1024  # 1GB内存限制

    def record_task_success(self):
        """记录任务成功"""
        self.task_count += 1
        self.consecutive_errors = 0
        self.last_heartbeat = time.time()

    def record_task_error(self):
        """记录任务错误"""
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_heartbeat = time.time()

    def should_restart(self):
        """判断是否应该重启worker"""
        # 连续错误过多
        if self.consecutive_errors >= self.max_consecutive_errors:
            print(f"Worker restart: 连续错误达到 {{self.consecutive_errors}} 次", file=sys.stderr)
            return True

        # 总错误数过多
        if self.error_count >= self.max_errors:
            print(f"Worker restart: 总错误数达到 {{self.error_count}} 次", file=sys.stderr)
            return True

        # 检查内存使用
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > self.max_memory_mb:
                print(f"Worker restart: 内存使用过高 ({{memory_mb:.1f}}MB > {{self.max_memory_mb}}MB)", file=sys.stderr)
                return True
        except ImportError:
            pass  # 如果没有psutil，跳过内存检查

        return False

    def get_stats(self):
        """获取统计信息"""
        uptime = time.time() - self.start_time
        return {{
            'uptime': uptime,
            'task_count': self.task_count,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors
        }}

# 全局健康管理器
health_manager = WorkerHealthManager()

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"Worker received signal {{signum}}, shutting down gracefully...", file=sys.stderr)
    stats = health_manager.get_stats()
    print(f"Worker stats: uptime={{stats['uptime']:.1f}}s, tasks={{stats['task_count']}}, errors={{stats['error_count']}}", file=sys.stderr)
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def normalize_value(x):
    '''将值标准化，将 None、inf、-inf、nan 都转换为 nan'''
    if x is None:
        return float('nan')
    try:
        val = float(x)
        if math.isinf(val) or math.isnan(val):
            return float('nan')
        return val
    except (ValueError, TypeError):
        return float('nan')

def execute_task_with_timeout(func_code, date, code, expected_length, timeout=120):
    '''带超时的任务执行'''
    import threading
    import queue

    result_queue = queue.Queue()

    def worker():
        try:
            namespace = {{'__builtins__': __builtins__}}
            exec(func_code, namespace)

            # 找到用户定义的函数
            user_functions = [name for name, obj in namespace.items()
                             if callable(obj) and not name.startswith('_') and name != 'execute_task']

            if not user_functions:
                result_queue.put([float('nan')] * expected_length)
                return

            func = namespace[user_functions[0]]
            result = func(date, code)

            if isinstance(result, list):
                normalized_result = [normalize_value(x) for x in result]
                result_queue.put(normalized_result)
            else:
                result_queue.put([float('nan')] * expected_length)

        except Exception as e:
            print(f"Task execution error for {{date}}, {{code}}: {{e}}", file=sys.stderr)
            result_queue.put([float('nan')] * expected_length)

    # 启动工作线程
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

    try:
        result = result_queue.get(timeout=timeout)
        thread.join(timeout=1)
        return result
    except queue.Empty:
        print(f"Task timeout for {{date}}, {{code}} after {{timeout}}s", file=sys.stderr)
        return [float('nan')] * expected_length
    except Exception as e:
        print(f"Task error for {{date}}, {{code}}: {{e}}", file=sys.stderr)
        return [float('nan')] * expected_length

def read_message_with_timeout(timeout=30):
    '''带超时的消息读取'''
    import select

    # 检查stdin是否可读
    if not select.select([sys.stdin.buffer], [], [], timeout)[0]:
        return None

    # 读取4字节长度前缀
    length_bytes = sys.stdin.buffer.read(4)
    if len(length_bytes) != 4:
        return None

    length = struct.unpack('<I', length_bytes)[0]
    if length == 0:
        return None

    # 验证长度合理性
    if length > 100 * 1024 * 1024:  # 100MB限制
        print(f"Message too large: {{length}} bytes", file=sys.stderr)
        return None

    # 读取实际数据
    data = sys.stdin.buffer.read(length)
    if len(data) != length:
        return None

    return data

def write_message(data):
    '''向stdout写入一条消息，带长度前缀'''
    try:
        length = len(data)
        length_bytes = struct.pack('<I', length)
        sys.stdout.buffer.write(length_bytes)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except IOError as e:
        print(f"Failed to write message: {{e}}", file=sys.stderr)
        raise

def main():
    print("🚀 Enhanced worker started (PID: {{}})".format(os.getpid()), file=sys.stderr)

    # 持续处理任务，直到收到空消息或需要重启
    while True:
        try:
            # 检查是否需要重启
            if health_manager.should_restart():
                print("🔄 Worker initiating restart due to health check", file=sys.stderr)
                sys.exit(1)

            # 带超时读取任务消息
            message_data = read_message_with_timeout(timeout=30)
            if message_data is None:
                break

            try:
                task_data = msgpack.unpackb(message_data, raw=False)
            except Exception as e:
                print(f"Failed to unpack message: {{e}}", file=sys.stderr)
                continue

            if not isinstance(task_data, dict):
                print(f"Error: Expected dict, got {{type(task_data)}}: {{task_data}}", file=sys.stderr)
                continue

            func_code = task_data['python_code']
            task = task_data['task']
            expected_length = task_data['expected_result_length']

            # 执行单个任务（带超时）
            timestamp = int(time.time() * 1000)
            date = task['date']
            code = task['code']

            try:
                facs = execute_task_with_timeout(func_code, date, code, expected_length, timeout=120)
                health_manager.record_task_success()
            except Exception as e:
                print(f"Task execution failed for {{date}}, {{code}}: {{e}}", file=sys.stderr)
                facs = [float('nan')] * expected_length
                health_manager.record_task_error()

            result = {{
                'date': date,
                'code': code,
                'timestamp': timestamp,
                'facs': facs
            }}

            # 使用MessagePack序列化并发送结果
            output = {{'result': result}}
            packed_output = msgpack.packb(output, use_bin_type=True)
            write_message(packed_output)

        except KeyboardInterrupt:
            print("🏁 Worker interrupted by user", file=sys.stderr)
            break
        except IOError as e:
            print(f"🏁 Worker I/O error: {{e}}", file=sys.stderr)
            break
        except Exception as e:
            print(f"Worker error: {{e}}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            health_manager.record_task_error()

            # 发送错误结果
            error_result = {{
                'result': {{
                    'date': 0,
                    'code': '',
                    'timestamp': int(time.time() * 1000),
                    'facs': [float('nan')] * expected_length
                }}
            }}
            try:
                packed_error = msgpack.packb(error_result, use_bin_type=True)
                write_message(packed_error)
            except Exception as write_error:
                print(f"Failed to send error result: {{write_error}}", file=sys.stderr)
                break

    # 输出最终统计
    stats = health_manager.get_stats()
    print("🏁 Enhanced worker finished", file=sys.stderr)
    print(f"Final stats: uptime={{stats['uptime']:.1f}}s, tasks={{stats['task_count']}}, errors={{stats['error_count']}}", file=sys.stderr)

if __name__ == '__main__':
    main()
"#)
}

fn extract_python_function_code(py_func: &PyObject) -> PyResult<String> {
    Python::with_gil(|py| {
        // 尝试获取函数的源代码
        let inspect = py.import("inspect")?;
        
        match inspect.call_method1("getsource", (py_func,)) {
            Ok(source) => {
                let source_str: String = source.extract()?;
                Ok(source_str)
            }
            Err(_) => {
                // 如果无法获取源代码，尝试使用pickle
                let pickle = py.import("pickle")?;
                match pickle.call_method1("dumps", (py_func,)) {
                    Ok(pickled) => {
                        let bytes: Vec<u8> = pickled.extract()?;
                        let base64 = py.import("base64")?;
                        let encoded = base64.call_method1("b64encode", (bytes,))?;
                        let encoded_str: String = encoded.call_method0("decode")?.extract()?;
                        
                        Ok(format!(r#"
import pickle
import base64
_func_data = base64.b64decode('{}')
user_function = pickle.loads(_func_data)
"#, encoded_str))
                    }
                    Err(_) => {
                        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Cannot serialize the Python function. Please ensure the function can be pickled or provide source code."
                        ))
                    }
                }
            }
        }
    })
}

// 新的进程池worker函数
fn run_process_pool_worker(
    worker_id: usize,
    task_queue: Receiver<TaskParam>,
    python_code: String,
    expected_result_length: usize,
    result_sender: Sender<TaskResult>,
    process_pool: Arc<ProcessPool>,
    monitor_manager: Arc<WorkerMonitorManager>,
) {
    // 向监控管理器注册worker
    monitor_manager.add_worker(worker_id);
    
    let mut task_count = 0;
    
    // 处理任务队列中的任务
    while let Ok(task) = task_queue.recv() {
        task_count += 1;
        
        // 通知监控管理器开始处理任务
        monitor_manager.start_task(worker_id, task.clone());
        monitor_manager.update_heartbeat(worker_id);
        
        // 获取一个进程来执行任务
        let process_id = process_pool.get_next_process();
        
        // 创建单任务数据
        let single_task = SingleTask {
            python_code: python_code.clone(),
            task: task.clone(),
            expected_result_length,
        };
        
        // 序列化任务数据
        let packed_data = match rmp_serde::to_vec_named(&single_task) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("❌ Worker {} 任务序列化失败: {}", worker_id, e);
                continue;
            }
        };
        
        // 使用进程池执行任务
        match process_pool.execute_task(process_id, &packed_data) {
            Ok(result_data) => {
                // 解析结果
                
                match rmp_serde::from_slice::<SingleResult>(&result_data) {
                    Ok(single_result) => {
                        // 发送结果
                        if let Err(e) = result_sender.send(single_result.result) {
                            eprintln!("❌ Worker {} 结果发送失败: {}", worker_id, e);
                        }
                        // 通知监控管理器任务已完成
                        monitor_manager.finish_task(worker_id);
                        monitor_manager.update_heartbeat(worker_id);
                    }
                    Err(e) => {
                        eprintln!("❌ Worker {} 结果解析失败: {}", worker_id, e);
                        
                        // 发送NaN结果
                        let error_result = TaskResult {
                            date: task.date,
                            code: task.code,
                            timestamp: chrono::Utc::now().timestamp_millis(),
                            facs: vec![f64::NAN; expected_result_length],
                        };
                        
                        if let Err(e) = result_sender.send(error_result) {
                            eprintln!("❌ Worker {} 错误结果发送失败: {}", worker_id, e);
                        }
                        monitor_manager.finish_task(worker_id);
                        monitor_manager.update_heartbeat(worker_id);
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Worker {} 进程池执行失败: {}", worker_id, e);
                
                // 发送NaN结果
                let error_result = TaskResult {
                    date: task.date,
                    code: task.code,
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    facs: vec![f64::NAN; expected_result_length],
                };
                
                if let Err(e) = result_sender.send(error_result) {
                    eprintln!("❌ Worker {} 错误结果发送失败: {}", worker_id, e);
                }
                monitor_manager.finish_task(worker_id);
                monitor_manager.update_heartbeat(worker_id);
            }
        }
    }
    
    println!("🏁 Worker {} 结束，共处理 {} 个任务", worker_id, task_count);
    
    // Worker完全结束时，从监控器中移除记录
    monitor_manager.remove_worker(worker_id);
}


#[pyfunction]
#[pyo3(signature = (python_function, args, n_jobs, backup_file, expected_result_length, restart_interval=None, update_mode=None, return_results=None, task_timeout=None, health_check_interval=None, debug_monitor=None, backup_batch_size=None))]
pub fn run_pools_queue(
    python_function: PyObject,
    args: &PyList,
    n_jobs: usize,
    backup_file: String,
    expected_result_length: usize,
    restart_interval: Option<usize>,
    update_mode: Option<bool>,
    return_results: Option<bool>,
    task_timeout: Option<u64>,
    health_check_interval: Option<u64>,
    debug_monitor: Option<bool>,
    backup_batch_size: Option<usize>,
) -> PyResult<PyObject> {
    // 处理 restart_interval 参数
    let restart_interval_value = restart_interval.unwrap_or(200);
    if restart_interval_value == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "restart_interval must be greater than 0"
        ));
    }
    
    // 处理 update_mode 参数
    let update_mode_enabled = update_mode.unwrap_or(false);
    
    // 处理 return_results 参数
    let return_results_enabled = return_results.unwrap_or(true);
    
    // 处理新的监控参数
    let task_timeout_secs = task_timeout.unwrap_or(120);
    let health_check_interval_secs = health_check_interval.unwrap_or(300); // 优化: 从120秒增加到300秒
    let debug_monitor_enabled = debug_monitor.unwrap_or(false);
    
    // 处理批处理大小参数
    let backup_batch_size_value = backup_batch_size.unwrap_or(5000); // 优化: 从1000增加到5000
    
    let task_timeout_duration = Duration::from_secs(task_timeout_secs);
    let health_check_duration = Duration::from_secs(health_check_interval_secs);
    
    if debug_monitor_enabled {
        println!("🔍 监控配置: 任务超时={}s, 健康检查间隔={}s", 
                 task_timeout_secs, health_check_interval_secs);
    }
    
    // 解析参数
    let mut all_tasks = Vec::new();
    for item in args.iter() {
        let task_args: &PyList = item.extract()?;
        if task_args.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each task should have exactly 2 parameters: date and code"
            ));
        }
        
        let date: i64 = task_args.get_item(0)?.extract()?;
        let code: String = task_args.get_item(1)?.extract()?;
        
        all_tasks.push(TaskParam { date, code });
    }
    
    // 保存所有任务的副本以便后续使用
    let all_tasks_clone = all_tasks.clone();
    
    // 读取现有备份，过滤已完成的任务
    let existing_tasks = if update_mode_enabled {
        // update_mode开启时，只读取传入参数中涉及的日期
        let task_dates: HashSet<i64> = all_tasks.iter().map(|t| t.date).collect();
        read_existing_backup_with_filter(&backup_file, Some(&task_dates))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read backup: {}", e)))?
    } else {
        // 正常模式，读取所有备份数据
        read_existing_backup(&backup_file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read backup: {}", e)))?
    };
    
    let pending_tasks: Vec<TaskParam> = all_tasks
        .into_iter()
        .filter(|task| !existing_tasks.contains(&(task.date, task.code.clone())))
        .collect();
    
    if pending_tasks.is_empty() {
        // 所有任务都已完成，直接返回结果
        println!("[{}] ✅ 所有任务都已完成，从备份文件读取结果", Local::now().format("%Y-%m-%d %H:%M:%S"));
        
        return if return_results_enabled {
            // 直接读取备份文件，避免使用线程池可能导致的死锁问题
            let read_start_time = Instant::now();
            println!("[{}] 🔍 开始读取备份文件: {}", Local::now().format("%Y-%m-%d %H:%M:%S"), backup_file);
            
            let result = if update_mode_enabled {
                // update_mode下，只返回传入参数中涉及的日期和代码
                let task_dates: HashSet<i64> = all_tasks_clone.iter().map(|t| t.date).collect();
                let task_codes: HashSet<String> = all_tasks_clone.iter().map(|t| t.code.clone()).collect();
                println!("[{}] 🔍 使用过滤模式读取 {} 个日期和 {} 个代码", 
                        Local::now().format("%Y-%m-%d %H:%M:%S"), 
                        task_dates.len(), 
                        task_codes.len());
                read_backup_results_with_filter(&backup_file, Some(&task_dates), Some(&task_codes))
            } else {
                println!("[{}] 🔍 读取完整备份文件", Local::now().format("%Y-%m-%d %H:%M:%S"));
                read_backup_results(&backup_file)
            };
            
            println!("[{}] ✅ 备份文件读取完成，耗时: {:?}", 
                     Local::now().format("%Y-%m-%d %H:%M:%S"), 
                     read_start_time.elapsed());
            
            result.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取备份文件失败: {}", e)))
        } else {
            println!("✅ 所有任务都已完成，不返回结果");
            Python::with_gil(|py| Ok(py.None()))
        };
    }
    
    let start_time = Instant::now();
    if update_mode_enabled {
        // update_mode下，只显示传入任务的统计信息
        println!("[{}] 📋 传入任务数: {}, 待处理: {}, 已完成: {}", 
                 Local::now().format("%Y-%m-%d %H:%M:%S"),
                 all_tasks_clone.len(), pending_tasks.len(), existing_tasks.len());
    } else {
        // 正常模式，显示总的统计信息
        println!("[{}] 📋 总任务数: {}, 待处理: {}, 已完成: {}", 
                 Local::now().format("%Y-%m-%d %H:%M:%S"),
                 pending_tasks.len() + existing_tasks.len(), pending_tasks.len(), existing_tasks.len());
    }
    
    // 提取Python函数代码
    let python_code = extract_python_function_code(&python_function)?;
    
    // 获取Python解释器路径
    let python_path = detect_python_interpreter();
    
    // 创建任务队列和结果收集通道
    let (task_sender, task_receiver) = unbounded::<TaskParam>();
    let (result_sender, result_receiver) = unbounded::<TaskResult>();
    
    // 将所有待处理任务放入队列
    for task in pending_tasks.clone() {
        if let Err(e) = task_sender.send(task) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to send task to queue: {}", e)
            ));
        }
    }
    drop(task_sender); // 关闭任务队列，worker会在队列空时退出
    
    let restart_flag = Arc::new(AtomicBool::new(false));
    
    // 创建监控管理器
    let monitor_manager = Arc::new(WorkerMonitorManager::new(
        task_timeout_duration,
        health_check_duration,
        debug_monitor_enabled,
    ));
    
    println!("[{}] 🚀 启动 {} 个worker处理 {} 个任务", Local::now().format("%Y-%m-%d %H:%M:%S"), n_jobs, pending_tasks.len());
    
    // 优化: 创建进程池，减少进程数量（从n_jobs个进程减少到合理数量）
    let optimal_process_count = std::cmp::min(n_jobs, 500); // 最多64个进程，避免系统过载
    let actual_process_count = std::cmp::max(optimal_process_count, 4); // 最少4个进程
    
    // 创建共享脚本文件
    let shared_script_path = "/tmp/persistent_worker_shared.py";
    let script_content = create_persistent_worker_script();
    if let Err(e) = std::fs::write(shared_script_path, script_content) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("创建共享脚本失败: {}", e)
        ));
    }
    
    // 创建进程池
    let process_pool = match ProcessPool::new(actual_process_count, &python_path, shared_script_path) {
        Ok(pool) => Arc::new(pool),
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("创建进程池失败: {}", e)
            ));
        }
    };
    
    println!("[{}] ✅ 进程池创建完成: {} 个Python进程服务于 {} 个worker线程", 
             Local::now().format("%Y-%m-%d %H:%M:%S"), 
             actual_process_count, n_jobs);
    
    // 启动worker线程 (使用进程池)
    let mut worker_handles = Vec::new();
    for i in 0..n_jobs {
        let worker_task_receiver = task_receiver.clone();
        let worker_python_code = python_code.clone();
        let worker_result_sender = result_sender.clone();
        let worker_process_pool = process_pool.clone();
        let worker_monitor_manager = monitor_manager.clone();
        
        let handle = thread::spawn(move || {
            run_process_pool_worker(
                i,
                worker_task_receiver,
                worker_python_code,
                expected_result_length,
                worker_result_sender,
                worker_process_pool,
                worker_monitor_manager,
            );
        });
        
        worker_handles.push(handle);
    }
    
    // 关闭主线程的result_sender
    drop(result_sender);
    
    // 启动监控线程
    let monitor_manager_clone = monitor_manager.clone();
    let monitor_restart_flag = restart_flag.clone();
    let worker_count = n_jobs;
    let monitor_handle = thread::spawn(move || {
        let mut workers_completed = 0;
        loop {
            // 检查是否应该退出监控循环
            if monitor_manager_clone.should_stop_monitoring() {
                println!("[{}] 🔍 监控器: 接收到停止信号，正在退出监控循环",
                         Local::now().format("%Y-%m-%d %H:%M:%S"));
                break;
            }

            // 检查是否所有worker都已完成（监控器中没有活跃worker）
            if let Ok(monitors) = monitor_manager_clone.monitors.lock() {
                // 如果监控器中没有活跃的worker，说明所有worker都已经完成并被移除
                if monitors.is_empty() {
                    println!("[{}] 🔍 监控器: 所有worker已完成，正在退出",
                             Local::now().format("%Y-%m-%d %H:%M:%S"));
                    break;
                } else {
                    // 调试信息：查看还有哪些worker在监控器中
                    if monitor_manager_clone.debug_monitor {
                        let active_workers: Vec<usize> = monitors.keys().cloned().collect();
                        println!("[{}] 🔍 监控器: 仍有活跃worker {:?}",
                                 Local::now().format("%Y-%m-%d %H:%M:%S"), active_workers);
                    }
                }
            }
            
            // 检查卡死的worker
            let stuck_workers = monitor_manager_clone.check_stuck_workers();
            if !stuck_workers.is_empty() {
                for (worker_id, reason) in stuck_workers {
                    monitor_manager_clone.log_stuck_worker(worker_id, reason);
                    
                    // 尝试强制终止卡死的worker进程
                    if monitor_manager_clone.force_kill_worker(worker_id) {
                        // 简化输出，避免频繁打断运行流程
                        if monitor_manager_clone.debug_monitor {
                            println!("🔄 已强制终止Worker {} (原因: {}), worker将自动重启", worker_id, reason);
                        }
                    }
                }
                
                // 触发重启（通过设置重启标志，worker会检测到并重启）
                monitor_restart_flag.store(true, Ordering::SeqCst);
                
                // 等待一小段时间让worker检测到重启信号
                thread::sleep(Duration::from_millis(100));
                
                // 重置重启标志，为下次监控做准备
                monitor_restart_flag.store(false, Ordering::SeqCst);
            }
            
            // 等待下一次检查，但在收到停止信号时立即退出
            for _ in 0..10 { // 检查10次，每次间隔1/10的health_check_interval
                if monitor_manager_clone.should_stop_monitoring() {
                    break;
                }
                thread::sleep(monitor_manager_clone.health_check_interval / 10);
            }
        }
    });
    
    // 启动结果收集器
    let backup_file_clone = backup_file.clone();
    let expected_result_length_clone = expected_result_length;
    let pending_tasks_len = pending_tasks.len();
    let collector_restart_flag = restart_flag.clone();
    let restart_interval_clone = restart_interval_value;
    let backup_batch_size_clone = backup_batch_size_value;
    let collector_handle = thread::spawn(move || {
        let mut batch_results = Vec::new();
        let mut total_collected = 0;
        let mut batch_count = 0;
        let mut batch_count_this_chunk = 0;
        let total_batches = (pending_tasks_len + backup_batch_size_clone - 1) / backup_batch_size_clone;
        
        println!("[{}] 🔄 结果收集器启动，等待worker结果...", Local::now().format("%Y-%m-%d %H:%M:%S"));
        
        while let Ok(result) = result_receiver.recv() {
            total_collected += 1;
            batch_results.push(result);
            
            // 根据backup_batch_size动态备份
            if batch_results.len() >= backup_batch_size_clone {
                batch_count += 1;
                batch_count_this_chunk += 1;
                
                let elapsed = start_time.elapsed();
                let elapsed_secs = elapsed.as_secs();
                let elapsed_h = elapsed_secs / 3600;
                let elapsed_m = (elapsed_secs % 3600) / 60;
                let elapsed_s = elapsed_secs % 60;

                let progress = if total_batches > 0 { batch_count as f64 / total_batches as f64 } else { 1.0 };
                let estimated_total_secs = if progress > 0.0 && progress <= 1.0 { elapsed.as_secs_f64() / progress } else { elapsed.as_secs_f64() };
                let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() { (estimated_total_secs - elapsed.as_secs_f64()) as u64 } else { 0 };

                let remaining_h = remaining_secs / 3600;
                let remaining_m = (remaining_secs % 3600) / 60;
                let remaining_s = remaining_secs % 60;
                
                let current_time = Local::now().format("%Y-%m-%d %H:%M:%S");
                print!("\r[{}] 💾 第 {}/{} 次备份。已用{}小时{}分钟{}秒，预余{}小时{}分钟{}秒", 
                       current_time, batch_count, total_batches, 
                       elapsed_h, elapsed_m, elapsed_s,
                       remaining_h, remaining_m, remaining_s);
                io::stdout().flush().unwrap(); // 强制刷新输出缓冲区
                
                match save_results_to_backup(&batch_results, &backup_file_clone, expected_result_length_clone) {
                    Ok(()) => {
                        // println!("✅ 第{}次备份成功！", batch_count);
                    }
                    Err(e) => {
                        eprintln!("❌ 第{}次备份失败: {}", batch_count, e);
                    }
                }
                batch_results.clear();

                if batch_count_this_chunk >= restart_interval_clone {
                    // println!("\n🔄 达到{}次备份，触发 workers 重启...", restart_interval_clone);
                    collector_restart_flag.store(true, Ordering::SeqCst);
                    batch_count_this_chunk = 0;
                }
            }
        }
        
        // 保存剩余结果
        if !batch_results.is_empty() {
            batch_count += 1;
            println!("[{}] 💾 保存最终剩余结果: {} 个", Local::now().format("%Y-%m-%d %H:%M:%S"), batch_results.len());
            
            match save_results_to_backup(&batch_results, &backup_file_clone, expected_result_length_clone) {
                Ok(()) => {
                    println!("[{}] ✅ 最终备份成功！", Local::now().format("%Y-%m-%d %H:%M:%S"));
                }
                Err(e) => {
                    eprintln!("❌ 最终备份失败: {}", e);
                }
            }
        }
        
        println!("[{}] 📊 收集器统计: 总收集{}个结果，进行了{}次备份", Local::now().format("%Y-%m-%d %H:%M:%S"), total_collected, batch_count);

    });
    
    // 等待所有worker完成
    println!("[{}] ⏳ 等待所有worker完成...", Local::now().format("%Y-%m-%d %H:%M:%S"));
    for (i, handle) in worker_handles.into_iter().enumerate() {
        match handle.join() {
            Ok(()) => {},
            Err(e) => eprintln!("❌ Worker {} 异常: {:?}", i, e),
        }
    }
    
    // 关闭进程池
    println!("[{}] 🔚 关闭进程池...", Local::now().format("%Y-%m-%d %H:%M:%S"));
    process_pool.shutdown();
    drop(process_pool); // 显式释放进程池
    
    // 立即停止监控线程，避免检查已死进程
    if debug_monitor_enabled {
        println!("🔍 监控器: 所有worker已完成，立即停止监控");
    }
    monitor_manager.stop_monitoring();
    
    // 等待监控线程结束
    println!("[{}] ⏳ 等待监控线程结束...", Local::now().format("%Y-%m-%d %H:%M:%S"));
    match monitor_handle.join() {
        Ok(()) => {
            if debug_monitor_enabled {
                println!("✅ 监控线程已安全退出");
            }
        },
        Err(e) => eprintln!("❌ 监控线程异常: {:?}", e),
    }
    
    // 等待收集器完成
    println!("[{}] ⏳ 等待结果收集器完成...", Local::now().format("%Y-%m-%d %H:%M:%S"));
    match collector_handle.join() {
        Ok(()) => {
            println!("[{}] ✅ 结果收集器已完成", Local::now().format("%Y-%m-%d %H:%M:%S"));
            // 确保备份文件的所有写入操作已同步到磁盘
            println!("[{}] 🔄 同步备份文件到磁盘...", Local::now().format("%Y-%m-%d %H:%M:%S"));
            if let Ok(file) = std::fs::File::open(&backup_file) {
                let _ = file.sync_all();
            }
        },
        Err(e) => eprintln!("❌ 结果收集器异常: {:?}", e),
    }
    
    // 打印监控诊断统计
    monitor_manager.print_diagnostic_stats();
    
    // 打印卡死任务统计表
    monitor_manager.print_stuck_tasks_table();
    
    // 显式清理监控管理器资源，避免与后续操作冲突
    println!("[{}] 🧹 清理监控器资源...", Local::now().format("%Y-%m-%d %H:%M:%S"));
    monitor_manager.cleanup();
    
    // 显式释放monitor_manager，确保所有Arc引用被清理
    drop(monitor_manager);
    
    // 等待短暂时间，确保所有资源完全释放，避免文件访问冲突
    println!("[{}] ⏳ 等待资源完全释放...", Local::now().format("%Y-%m-%d %H:%M:%S"));
    thread::sleep(Duration::from_millis(100));
    
    // 清理共享脚本文件
    let shared_script_path = "/tmp/persistent_worker_shared.py";
    if Path::new(shared_script_path).exists() {
        let _ = std::fs::remove_file(shared_script_path);
        println!("[{}] 🧹 已清理共享脚本文件", Local::now().format("%Y-%m-%d %H:%M:%S"));
    }
    
    // 读取并返回最终结果
    if return_results_enabled {
        println!("[{}] 📖 读取最终备份结果...", Local::now().format("%Y-%m-%d %H:%M:%S"));
        
        // 直接读取备份文件，避免线程池冲突
        println!("[{}] 🔍 开始读取备份文件: {}", Local::now().format("%Y-%m-%d %H:%M:%S"), backup_file);
        let start_read_time = Instant::now();
        
        let result = if update_mode_enabled {
            // update_mode下，只返回传入参数中涉及的日期和代码
            let task_dates: HashSet<i64> = all_tasks_clone.iter().map(|t| t.date).collect();
            let task_codes: HashSet<String> = all_tasks_clone.iter().map(|t| t.code.clone()).collect();
            println!("[{}] 🔍 使用过滤模式读取 {} 个日期和 {} 个代码", 
                    Local::now().format("%Y-%m-%d %H:%M:%S"), 
                    task_dates.len(), 
                    task_codes.len());
            read_backup_results_with_filter(&backup_file, Some(&task_dates), Some(&task_codes))
        } else {
            println!("[{}] 🔍 读取完整备份文件", Local::now().format("%Y-%m-%d %H:%M:%S"));
            read_backup_results(&backup_file)
        };
        
        println!("[{}] ✅ 备份文件读取完成，耗时: {:?}", 
                 Local::now().format("%Y-%m-%d %H:%M:%S"), 
                 start_read_time.elapsed());
        result
    } else {
        println!("✅ 任务完成，不返回结果");
        Python::with_gil(|py| Ok(py.None()))
    }
}