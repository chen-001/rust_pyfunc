use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self,Read, Write};
use std::path::Path;
use memmap2::{Mmap, MmapMut};
use std::process::{Command, Stdio};
use std::thread;
use crossbeam::channel::{unbounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::env;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use chrono::Local;
use std::sync::Mutex;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParam {
    pub date: i64,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub date: i64,
    pub code: String,
    pub timestamp: i64,
    pub facs: Vec<f64>,
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
#[allow(dead_code)]
struct WorkerMonitor {
    worker_id: usize,
    last_heartbeat: Instant,
    current_task: Option<TaskParam>,
    task_start_time: Option<Instant>,
    is_alive: bool,
    consecutive_failures: u32,
    process_id: Option<u32>,  // 子进程ID，用于进程存活检测
}

impl WorkerMonitor {
    fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            last_heartbeat: Instant::now(),
            current_task: None,
            task_start_time: None,
            is_alive: true,
            consecutive_failures: 0,
            process_id: None,
        }
    }
    
    fn start_task(&mut self, task: TaskParam) {
        self.current_task = Some(task);
        self.task_start_time = Some(Instant::now());
    }
    
    fn finish_task(&mut self) {
        self.current_task = None;
        self.task_start_time = None;
        self.consecutive_failures = 0;  // 重置失败计数
    }
    
    fn update_heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
        self.is_alive = true;
    }
    
    fn set_process_id(&mut self, pid: u32) {
        self.process_id = Some(pid);
    }
    
    fn is_process_alive(&self) -> bool {
        if let Some(pid) = self.process_id {
            // 在Linux上，检查/proc/PID目录是否存在
            #[cfg(target_os = "linux")]
            {
                std::path::Path::new(&format!("/proc/{}", pid)).exists()
            }
            
            // 在其他系统上，简化为Linux的方法，因为大多数系统都有/proc
            #[cfg(not(target_os = "linux"))]
            {
                // 简化处理：在非Linux系统也尝试/proc方法，如果失败就假设进程存活
                std::path::Path::new(&format!("/proc/{}", pid)).exists()
            }
        } else {
            true  // 如果没有进程ID，假设进程存活
        }
    }
    
    fn is_stuck(&self, task_timeout: Duration, heartbeat_timeout: Duration) -> Option<&'static str> {
        // 首先检查进程是否还活着
        if !self.is_process_alive() {
            return Some("process_death");
        }
        
        // 检查心跳超时
        if self.last_heartbeat.elapsed() > heartbeat_timeout {
            return Some("heartbeat_timeout");
        }
        
        // 检查任务执行超时
        if let Some(start_time) = self.task_start_time {
            if start_time.elapsed() > task_timeout {
                return Some("task_timeout");
            }
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
        
        if let Ok(monitors) = self.monitors.lock() {
            for (worker_id, monitor) in monitors.iter() {
                // 跳过已经标记为不存活或没有进程ID的worker
                if !monitor.is_alive || monitor.process_id.is_none() {
                    continue;
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
                            _ => {}
                        }
                    }
                    
                    if self.debug_monitor {
                        println!("⚠️ 监控器: 检测到Worker {} 卡死 (原因: {})", worker_id, stuck_reason);
                        if let Some(task) = &monitor.current_task {
                            println!("   正在处理任务: date={}, code={}", task.date, task.code);
                        }
                        println!("   最后心跳: {:?}前", monitor.last_heartbeat.elapsed());
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
    
    #[allow(dead_code)]
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

fn read_existing_backup(file_path: &str) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
    read_existing_backup_with_filter(file_path, None)
}

fn read_existing_backup_with_filter(file_path: &str, date_filter: Option<&HashSet<i64>>) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
    let mut existing_tasks = HashSet::new();
    
    if !Path::new(file_path).exists() {
        return Ok(existing_tasks);
    }
    
    let file = File::open(file_path)?;
    let file_len = file.metadata()?.len() as usize;
    
    if file_len < HEADER_SIZE {
        // 回退到旧格式
        return read_existing_backup_legacy(file_path);
    }
    
    // 尝试新格式
    let mmap = unsafe { Mmap::map(&file)? };
    
    // 检查魔数
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    if &header.magic != b"RPBACKUP" {
        // 不是新格式，回退到旧格式
        return read_existing_backup_legacy(file_path);
    }
    
    let record_count = header.record_count as usize;
    let factor_count = header.factor_count as usize;
    let record_size = calculate_record_size(factor_count);
    let records_start = HEADER_SIZE;
    
    // 检查版本号
    if header.version == 2 {
        // 新的动态格式
        for i in 0..record_count {
            let record_offset = records_start + i * record_size;
            let record_bytes = &mmap[record_offset..record_offset + record_size];
            
            match DynamicRecord::from_bytes(record_bytes, factor_count) {
                Ok(record) => {
                    // 如果有日期过滤器，只有匹配的日期才会被包含
                    if let Some(filter) = date_filter {
                        if !filter.contains(&record.date) {
                            continue;
                        }
                    }
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                    existing_tasks.insert((record.date, code));
                }
                Err(_) => {
                    // 记录损坏，跳过
                    continue;
                }
            }
        }
    } else {
        // 旧格式，回退到legacy处理
        return read_existing_backup_legacy_with_filter(file_path, date_filter);
    }
    
    Ok(existing_tasks)
}

fn read_existing_backup_legacy(file_path: &str) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
    read_existing_backup_legacy_with_filter(file_path, None)
}

fn read_existing_backup_legacy_with_filter(file_path: &str, date_filter: Option<&HashSet<i64>>) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
    let mut existing_tasks = HashSet::new();
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    if buffer.is_empty() {
        return Ok(existing_tasks);
    }
    
    let mut cursor = 0;
    
    // 尝试新的批次格式
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3],
            size_bytes[4], size_bytes[5], size_bytes[6], size_bytes[7],
        ]) as usize;
        
        cursor += 8;
        
        if cursor + batch_size > buffer.len() {
            cursor -= 8;
            break;
        }
        
        match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..cursor + batch_size]) {
            Ok(batch) => {
                for result in &batch {
                    // 如果有日期过滤器，只有匹配的日期才会被包含
                    if let Some(filter) = date_filter {
                        if !filter.contains(&result.date) {
                            continue;
                        }
                    }
                    existing_tasks.insert((result.date, result.code.clone()));
                }
                cursor += batch_size;
            }
            Err(_) => {
                cursor -= 8;
                break;
            }
        }
    }
    
    // 如果失败，尝试原始格式
    if cursor < buffer.len() {
        while cursor < buffer.len() {
            match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..]) {
                Ok(batch) => {
                    for result in &batch {
                        // 如果有日期过滤器，只有匹配的日期才会被包含
                        if let Some(filter) = date_filter {
                            if !filter.contains(&result.date) {
                                continue;
                            }
                        }
                        existing_tasks.insert((result.date, result.code.clone()));
                    }
                    let batch_size = bincode::serialized_size(&batch)? as usize;
                    cursor += batch_size;
                }
                Err(_) => {
                    cursor += std::cmp::min(64, buffer.len() - cursor);
                }
            }
        }
    }
    
    Ok(existing_tasks)
}

// 动态因子数量支持 - 基于expected_result_length计算记录大小
const HEADER_SIZE: usize = 64;  // 文件头64字节
const MAX_FACTORS: usize = 256; // 临时向后兼容常量
const RECORD_SIZE: usize = 2116; // 临时向后兼容常量（对应256个因子）

// 动态计算记录大小
fn calculate_record_size(factor_count: usize) -> usize {
    8 +        // date: i64
    8 +        // code_hash: u64 
    8 +        // timestamp: i64
    4 +        // factor_count: u32
    4 +        // code_len: u32
    32 +       // code_bytes: [u8; 32]
    factor_count * 8 +  // factors: [f64; factor_count]
    4          // checksum: u32
}

#[repr(C, packed)]
struct FileHeader {
    magic: [u8; 8],          // 魔数 "RPBACKUP"
    version: u32,            // 版本号
    record_count: u64,       // 记录总数
    record_size: u32,        // 单条记录大小
    factor_count: u32,       // 因子数量
    reserved: [u8; 36],      // 保留字段
}

// 临时向后兼容的固定记录结构
#[repr(C, packed)]
struct FixedRecord {
    date: i64,
    code_hash: u64,
    timestamp: i64,
    factor_count: u32,
    code_len: u32,
    code_bytes: [u8; 32],
    factors: [f64; MAX_FACTORS],
    checksum: u32,
}

// 动态大小记录结构
#[derive(Debug, Clone)]
struct DynamicRecord {
    date: i64,
    code_hash: u64,
    timestamp: i64,
    factor_count: u32,
    code_len: u32,
    code_bytes: [u8; 32],
    factors: Vec<f64>,  // 动态大小的因子数组
    checksum: u32,
}

impl DynamicRecord {
    fn from_task_result(result: &TaskResult) -> Self {
        let mut record = DynamicRecord {
            date: result.date,
            code_hash: calculate_hash(&result.code),
            timestamp: result.timestamp,
            factor_count: result.facs.len() as u32,
            code_len: 0,
            code_bytes: [0; 32],
            factors: result.facs.clone(),
            checksum: 0,
        };
        
        // 处理code字符串，确保安全访问
        let code_bytes = result.code.as_bytes();
        let safe_len = std::cmp::min(code_bytes.len(), 32);
        record.code_len = safe_len as u32;
        record.code_bytes[..safe_len].copy_from_slice(&code_bytes[..safe_len]);
        
        // 计算校验和
        record.checksum = record.calculate_checksum();
        
        record
    }
    
    // fn to_task_result(&self) -> TaskResult {
    //     let code = String::from_utf8_lossy(&self.code_bytes[..self.code_len as usize]).to_string();
    //     TaskResult {
    //         date: self.date,
    //         code,
    //         timestamp: self.timestamp,
    //         facs: self.factors.clone(),
    //     }
    // }
    
    fn calculate_checksum(&self) -> u32 {
        let mut sum = 0u32;
        sum = sum.wrapping_add(self.date as u32);
        sum = sum.wrapping_add((self.date >> 32) as u32);
        sum = sum.wrapping_add(self.code_hash as u32);
        sum = sum.wrapping_add((self.code_hash >> 32) as u32);
        sum = sum.wrapping_add(self.timestamp as u32);
        sum = sum.wrapping_add(self.factor_count);
        sum = sum.wrapping_add(self.code_len);
        
        for &factor in &self.factors {
            sum = sum.wrapping_add(factor.to_bits() as u32);
            sum = sum.wrapping_add((factor.to_bits() >> 32) as u32);
        }
        
        sum
    }
    
    // 将记录序列化为字节数组
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        bytes.extend_from_slice(&self.date.to_le_bytes());
        bytes.extend_from_slice(&self.code_hash.to_le_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(&self.factor_count.to_le_bytes());
        bytes.extend_from_slice(&self.code_len.to_le_bytes());
        bytes.extend_from_slice(&self.code_bytes);
        
        for &factor in &self.factors {
            bytes.extend_from_slice(&factor.to_le_bytes());
        }
        
        bytes.extend_from_slice(&self.checksum.to_le_bytes());
        
        bytes
    }
    
    // 从字节数组反序列化记录
    fn from_bytes(bytes: &[u8], expected_factor_count: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < calculate_record_size(expected_factor_count) {
            return Err("Insufficient bytes for record".into());
        }
        
        let mut offset = 0;
        
        let date = i64::from_le_bytes(bytes[offset..offset+8].try_into()?);
        offset += 8;
        
        let code_hash = u64::from_le_bytes(bytes[offset..offset+8].try_into()?);
        offset += 8;
        
        let timestamp = i64::from_le_bytes(bytes[offset..offset+8].try_into()?);
        offset += 8;
        
        let factor_count = u32::from_le_bytes(bytes[offset..offset+4].try_into()?);
        offset += 4;
        
        let code_len = u32::from_le_bytes(bytes[offset..offset+4].try_into()?);
        offset += 4;
        
        let mut code_bytes = [0u8; 32];
        code_bytes.copy_from_slice(&bytes[offset..offset+32]);
        offset += 32;
        
        let mut factors = Vec::with_capacity(expected_factor_count);
        for _ in 0..expected_factor_count {
            let factor = f64::from_le_bytes(bytes[offset..offset+8].try_into()?);
            factors.push(factor);
            offset += 8;
        }
        
        let checksum = u32::from_le_bytes(bytes[offset..offset+4].try_into()?);
        
        Ok(DynamicRecord {
            date,
            code_hash,
            timestamp,
            factor_count,
            code_len,
            code_bytes,
            factors,
            checksum,
        })
    }
}

fn calculate_hash(s: &str) -> u64 {
    // 简单的哈希函数
    let mut hash = 0u64;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    hash
}

fn save_results_to_backup(results: &[TaskResult], backup_file: &str, expected_result_length: usize) -> Result<(), Box<dyn std::error::Error>> {
    if results.is_empty() {
        return Ok(());
    }
    
    let factor_count = expected_result_length;
    let record_size = calculate_record_size(factor_count);
    
    // 检查文件是否存在且有效
    let file_path = Path::new(backup_file);
    let file_exists = file_path.exists();
    let file_valid = if file_exists {
        file_path.metadata().map(|m| m.len() >= HEADER_SIZE as u64).unwrap_or(false)
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
    if file_len < HEADER_SIZE {
        return Err(format!("File is too small to contain valid header: {} < {}", file_len, HEADER_SIZE).into());
    }
    
    let mut header_bytes = [0u8; HEADER_SIZE];
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
    let new_file_size = HEADER_SIZE as u64 + new_count * record_size as u64;
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
    let start_offset = HEADER_SIZE + current_count as usize * record_size;
    
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

def execute_task(func_code, date, code, expected_length):
    '''执行单个任务，返回结果或NaN'''
    try:
        namespace = {{'__builtins__': __builtins__}}
        exec(func_code, namespace)
        
        # 找到用户定义的函数（非内置函数）
        user_functions = [name for name, obj in namespace.items() 
                         if callable(obj) and not name.startswith('_') and name != 'execute_task']
        
        if not user_functions:
            return [float('nan')] * expected_length
            
        # 使用第一个用户定义的函数
        func = namespace[user_functions[0]]
        result = func(date, code)
        
        if isinstance(result, list):
            # 使用 normalize_value 函数处理所有特殊值
            return [normalize_value(x) for x in result]
        else:
            return [float('nan')] * expected_length
            
    except Exception as e:
        print(f"Task error for {{date}}, {{code}}: {{e}}", file=sys.stderr)
        return [float('nan')] * expected_length

def read_message():
    '''从stdin读取一条消息，返回长度+数据'''
    # 读取4字节长度前缀
    length_bytes = sys.stdin.buffer.read(4)
    if len(length_bytes) != 4:
        return None
    
    length = struct.unpack('<I', length_bytes)[0]
    if length == 0:
        return None
    
    # 读取实际数据
    data = sys.stdin.buffer.read(length)
    if len(data) != length:
        return None
    
    return data

def write_message(data):
    '''向stdout写入一条消息，带长度前缀'''
    length = len(data)
    length_bytes = struct.pack('<I', length)
    sys.stdout.buffer.write(length_bytes)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

def main():
    print("🚀 Persistent worker started", file=sys.stderr)
    
    # 持续处理任务，直到收到空消息
    while True:
        try:
            # 读取任务消息
            message_data = read_message()
            if message_data is None:
                break
            
            task_data = msgpack.unpackb(message_data, raw=False)
            
            if not isinstance(task_data, dict):
                print(f"Error: Expected dict, got {{type(task_data)}}: {{task_data}}", file=sys.stderr)
                continue
            
            func_code = task_data['python_code']
            task = task_data['task']
            expected_length = task_data['expected_result_length']
            
            # 执行单个任务
            timestamp = int(time.time() * 1000)
            date = task['date']
            code = task['code']
            
            facs = execute_task(func_code, date, code, expected_length)
            
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

        except Exception as e:
            print(f"Failed to process task: {{e}}", file=sys.stderr)
            # 发送错误结果
            error_result = {{
                'result': {{
                    'date': 0,
                    'code': '',
                    'timestamp': int(time.time() * 1000),
                    'facs': [float('nan')] * 1
                }}
            }}
            packed_error = msgpack.packb(error_result, use_bin_type=True)
            write_message(packed_error)

    print("🏁 Persistent worker finished", file=sys.stderr)

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







fn read_backup_results(file_path: &str) -> PyResult<PyObject> {
    read_backup_results_with_filter(file_path, None, None)
}

fn read_backup_results_with_filter(file_path: &str, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Python::with_gil(|py| Ok(py.None()));
    }
    
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open backup file: {}", e)))?;
    
    let file_len = file.metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to get file metadata: {}", e)))?
        .len() as usize;
    
    if file_len < HEADER_SIZE {
        // 尝试旧格式的回退处理
        return read_legacy_backup_results_with_filter(file_path, date_filter, code_filter);
    }
    
    // 使用内存映射进行超高速读取
    let mmap = unsafe { 
        Mmap::map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to memory map file: {}", e)))?
    };
    let mmap = Arc::new(mmap);
    
    // 读取文件头
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    // 验证魔数
    if &header.magic != b"RPBACKUP" {
        // 不是新格式，尝试旧格式
        return read_legacy_backup_results_with_filter(file_path, date_filter, code_filter);
    }
    
    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No records found in backup file"
        ));
    }
    
    // 检查版本并计算预期文件大小
    let factor_count = header.factor_count as usize;
    let record_size = if header.version == 2 {
        calculate_record_size(factor_count)
    } else {
        RECORD_SIZE  // 旧格式使用固定大小
    };
    
    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Backup file appears to be truncated"
        ));
    }
    
    // 预计算矩阵维度
    let factor_count = header.factor_count as usize;
    let num_cols = 3 + factor_count;
    
    // 根据版本选择不同的读取方式
    let parallel_results: Result<Vec<_>, _> = if header.version == 2 {
        // 新的动态格式读取
        (0..record_count)
            .collect::<Vec<_>>()
            .chunks(std::cmp::max(64, record_count / rayon::current_num_threads()))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
            .par_iter()
            .map(|chunk| {
                let mut chunk_data = Vec::with_capacity(chunk.len() * num_cols);
                let records_start = HEADER_SIZE;
                
                for &i in chunk {
                    let record_offset = records_start + i * record_size;
                    let record_bytes = &mmap[record_offset..record_offset + record_size];
                    
                    match DynamicRecord::from_bytes(record_bytes, factor_count) {
                        Ok(record) => {
                            // 检查日期过滤器
                            if let Some(date_filter) = date_filter {
                                if !date_filter.contains(&record.date) {
                                    continue;
                                }
                            }
                            
                            // 检查代码过滤器
                            let code_len = std::cmp::min(record.code_len as usize, 32);
                            let code_str = String::from_utf8_lossy(&record.code_bytes[..code_len]);
                            if let Some(code_filter) = code_filter {
                                if !code_filter.contains(code_str.as_ref()) {
                                    continue;
                                }
                            }
                            
                            chunk_data.push(record.date as f64);
                            
                            // 安全的code转换
                            let code_num = if let Ok(num) = code_str.parse::<f64>() {
                                num
                            } else {
                                f64::NAN
                            };
                            chunk_data.push(code_num);
                            
                            chunk_data.push(record.timestamp as f64);
                            
                            // 复制因子数据
                            for j in 0..factor_count {
                                if j < record.factors.len() {
                                    chunk_data.push(record.factors[j]);
                                } else {
                                    chunk_data.push(f64::NAN);
                                }
                            }
                        }
                        Err(_) => {
                            // 记录损坏，填充NaN
                            for _ in 0..num_cols {
                                chunk_data.push(f64::NAN);
                            }
                        }
                    }
                }
                
                Ok(chunk_data)
            })
            .collect()
    } else {
        // 旧格式，使用FixedRecord
        (0..record_count)
            .collect::<Vec<_>>()
            .chunks(std::cmp::max(64, record_count / rayon::current_num_threads()))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
            .par_iter()
            .map(|chunk| {
                let mut chunk_data = Vec::with_capacity(chunk.len() * num_cols);
                let records_start = HEADER_SIZE;
                
                for &i in chunk {
                    let record_offset = records_start + i * RECORD_SIZE;
                    let record = unsafe {
                        &*(mmap.as_ptr().add(record_offset) as *const FixedRecord)
                    };
                    
                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        let date = record.date; // 复制到本地变量避免unaligned reference
                        if !date_filter.contains(&date) {
                            continue;
                        }
                    }
                    
                    // 检查代码过滤器
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code_str = unsafe {
                        std::str::from_utf8_unchecked(&record.code_bytes[..code_len])
                    };
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(code_str) {
                            continue;
                        }
                    }
                    
                    // 直接复制数据到输出数组
                    chunk_data.push(record.date as f64);
                    
                    // 尝试快速解析数字，失败则使用NaN
                    let code_num = if let Ok(num) = code_str.parse::<f64>() {
                        num
                    } else {
                        // 对于非数字股票代码，可以使用哈希值或直接使用NaN
                        record.code_hash as f64
                    };
                    chunk_data.push(code_num);
                    
                    chunk_data.push(record.timestamp as f64);
                    
                    // 批量复制因子数据
                    let actual_factor_count = std::cmp::min(
                        std::cmp::min(record.factor_count as usize, MAX_FACTORS),
                        factor_count
                    );
                    
                    // 直接内存复制因子数据（更快）
                    for j in 0..actual_factor_count {
                        chunk_data.push(record.factors[j]);
                    }
                    
                    // 如果因子数量不足，填充NaN
                    for _ in actual_factor_count..factor_count {
                        chunk_data.push(f64::NAN);
                    }
                }
                
                Ok(chunk_data)
            })
            .collect()
    };
    
    let all_chunk_data = parallel_results
        .map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    // 合并所有chunk的数据
    let mut flat_data = Vec::with_capacity(record_count * num_cols);
    for chunk_data in all_chunk_data {
        flat_data.extend(chunk_data);
    }
    
    // 计算实际的行数（考虑过滤）
    let actual_row_count = flat_data.len() / num_cols;
    
    // 超高速转换：直接从内存映射创建numpy数组
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        
        // 创建numpy数组并reshape（使用实际行数）
        let array = numpy.call_method1("array", (flat_data,))?;
        let reshaped = array.call_method1("reshape", ((actual_row_count, num_cols),))?;
        
        Ok(reshaped.into())
    })
}

// 向后兼容的旧格式读取函数
#[allow(dead_code)]
fn read_legacy_backup_results(file_path: &str) -> PyResult<PyObject> {
    read_legacy_backup_results_with_filter(file_path, None, None)
}

fn read_legacy_backup_results_with_filter(file_path: &str, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    let mut file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open backup file: {}", e)))?;
    
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read backup file: {}", e)))?;
    
    if buffer.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Backup file is empty"
        ));
    }
    
    let mut all_results = Vec::new();
    let mut cursor = 0;
    
    // 尝试新的批次格式（带大小头）
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3],
            size_bytes[4], size_bytes[5], size_bytes[6], size_bytes[7],
        ]) as usize;
        
        cursor += 8;
        
        if cursor + batch_size > buffer.len() {
            cursor -= 8;
            break;
        }
        
        match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..cursor + batch_size]) {
            Ok(batch) => {
                for result in batch {
                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&result.date) {
                            continue;
                        }
                    }
                    
                    // 检查代码过滤器
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&result.code) {
                            continue;
                        }
                    }
                    
                    all_results.push(result);
                }
                cursor += batch_size;
            }
            Err(_) => {
                cursor -= 8;
                break;
            }
        }
    }
    
    // 如果失败，尝试原始格式
    if cursor < buffer.len() {
        while cursor < buffer.len() {
            match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..]) {
                Ok(batch) => {
                    let batch_size = bincode::serialized_size(&batch)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e)))? as usize;
                    for result in batch {
                        // 检查日期过滤器
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&result.date) {
                                continue;
                            }
                        }
                        
                        // 检查代码过滤器
                        if let Some(code_filter) = code_filter {
                            if !code_filter.contains(&result.code) {
                                continue;
                            }
                        }
                        
                        all_results.push(result);
                    }
                    cursor += batch_size;
                }
                Err(_) => {
                    cursor += std::cmp::min(64, buffer.len() - cursor);
                }
            }
        }
    }
    
    if all_results.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No valid results found in backup file"
        ));
    }
    
    convert_results_to_py_dict(&all_results)
}

// 将TaskResult切片转换为包含Numpy数组的Python字典
fn convert_results_to_py_dict(results: &[TaskResult]) -> PyResult<PyObject> {
    if results.is_empty() {
        return Python::with_gil(|py| Ok(pyo3::types::PyDict::new(py).into()));
    }
    
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);

        let num_rows = results.len();
        let factor_count = results.get(0).map_or(0, |r| r.facs.len());

        let mut dates = Vec::with_capacity(num_rows);
        let mut codes = Vec::with_capacity(num_rows);
        let mut timestamps = Vec::with_capacity(num_rows);
        let mut factors_flat = Vec::with_capacity(num_rows * factor_count);

        for result in results {
            dates.push(result.date);
            codes.push(result.code.clone());
            timestamps.push(result.timestamp);
            factors_flat.extend_from_slice(&result.facs);
        }

        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("timestamp", numpy.call_method1("array", (timestamps,))?)?;
        
        let factors_array = numpy.call_method1("array", (factors_flat,))?;
        
        if num_rows > 0 && factor_count > 0 {
            let factors_reshaped = factors_array.call_method1("reshape", ((num_rows, factor_count),))?;
            dict.set_item("factors", factors_reshaped)?;
        } else {
            dict.set_item("factors", factors_array)?;
        }

        Ok(dict.into())
    })
}

#[pyfunction]
#[pyo3(signature = (backup_file,))]
pub fn query_backup(backup_file: String) -> PyResult<PyObject> {
    read_backup_results(&backup_file)
}

/// 高速并行备份查询函数，专门优化大文件读取
#[pyfunction]
#[pyo3(signature = (backup_file, num_threads=None, dates=None, codes=None))]
pub fn query_backup_fast(backup_file: String, num_threads: Option<usize>, dates: Option<Vec<i64>>, codes: Option<Vec<String>>) -> PyResult<PyObject> {
    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());
    
    // 使用自定义线程池而不是全局线程池
    if let Some(threads) = num_threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create thread pool: {}", e)))?;
        
        pool.install(|| read_backup_results_ultra_fast_v4_with_filter(
            &backup_file, 
            date_filter.as_ref(), 
            code_filter.as_ref()
        ))
    } else {
        read_backup_results_ultra_fast_v4_with_filter(
            &backup_file, 
            date_filter.as_ref(), 
            code_filter.as_ref()
        )
    }
}




/// 读取备份文件中的指定列
/// column_index: 要读取的因子列索引（0表示第一列因子值）
/// 返回三列：date, code, 指定列的因子值
fn read_backup_results_single_column(file_path: &str, column_index: usize) -> PyResult<PyObject> {
    read_backup_results_single_column_with_filter(file_path, column_index, None, None)
}

fn read_backup_results_single_column_with_filter(file_path: &str, column_index: usize, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在"
        ));
    }
    
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let file_len = file.metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e)))?
        .len() as usize;
    
    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_single_column_with_filter(file_path, column_index, date_filter, code_filter);
    }
    
    // 内存映射
    let mmap = unsafe { 
        Mmap::map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e)))?
    };
    let mmap = Arc::new(mmap);
    
    // 读取文件头
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    if &header.magic != b"RPBACKUP" {
        return read_legacy_backup_results_single_column_with_filter(file_path, column_index, date_filter, code_filter);
    }
    
    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item("code", numpy.call_method1("array", (Vec::<String>::new(),))?)?;
            dict.set_item("factor", numpy.call_method1("array", (Vec::<f64>::new(),))?)?;
            Ok(dict.into())
        });
    }
    
    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;
    
    // 检查列索引是否有效
    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("列索引 {} 超出范围，因子列数为 {}", column_index, factor_count)
        ));
    }
    
    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
           format!("记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.", record_size, calculated_record_size)
       ));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了"
        ));
    }
    
    // 使用自定义线程池并直接从mmap读取，避免大量内存复制
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 8))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e)))?;
    
    let records_start = HEADER_SIZE;
    let results: Vec<_> = pool.install(|| {
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];
                
                match DynamicRecord::from_bytes(record_bytes, factor_count) {
                    Ok(record) => {
                        // 检查日期过滤器
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&record.date) {
                                return None;
                            }
                        }
                        
                        let code_len = std::cmp::min(record.code_len as usize, 32);
                        let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                        
                        // 检查代码过滤器
                        if let Some(code_filter) = code_filter {
                            if !code_filter.contains(&code) {
                                return None;
                            }
                        }
                        
                        // 获取指定列的因子值
                        let factor_value = if column_index < record.factors.len() {
                            record.factors[column_index]
                        } else {
                            f64::NAN
                        };
                        
                        Some((record.date, code, factor_value))
                    }
                    Err(_) => {
                        // 记录损坏，返回None
                        None
                    }
                }
            })
            .collect::<Vec<_>>()
    });
    
    // 显式释放mmap
    drop(mmap);
        
    let num_rows = results.len();
    let mut dates = Vec::with_capacity(num_rows);
    let mut codes = Vec::with_capacity(num_rows);
    let mut factors = Vec::with_capacity(num_rows);

    for (date, code, factor_value) in results {
        dates.push(date);
        codes.push(code);
        factors.push(factor_value);
    }

    // 创建Python字典
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("factor", numpy.call_method1("array", (factors,))?)?;

        Ok(dict.into())
    })
}

fn read_backup_results_columns_range_with_filter(file_path: &str, column_start: usize, column_end: usize, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在"
        ));
    }
    
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let file_len = file.metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e)))?
        .len() as usize;
    
    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_columns_range_with_filter(file_path, column_start, column_end, date_filter, code_filter);
    }
    
    // 内存映射
    let mmap = unsafe { 
        Mmap::map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e)))?
    };
    let mmap = Arc::new(mmap);
    
    // 读取文件头
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    if &header.magic != b"RPBACKUP" {
        return read_legacy_backup_results_columns_range_with_filter(file_path, column_start, column_end, date_filter, code_filter);
    }
    
    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item("code", numpy.call_method1("array", (Vec::<String>::new(),))?)?;
            dict.set_item("factors", numpy.call_method1("array", (Vec::<Vec<f64>>::new(),))?)?;
            Ok(dict.into())
        });
    }
    
    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;
    
    // 检查列索引是否有效
    if column_start >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("起始列索引 {} 超出范围，因子列数为 {}", column_start, factor_count)
        ));
    }
    
    if column_end >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("结束列索引 {} 超出范围，因子列数为 {}", column_end, factor_count)
        ));
    }
    
    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
           format!("记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.", record_size, calculated_record_size)
       ));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了"
        ));
    }
    
    // 并行读取指定列范围
    let records_start = HEADER_SIZE;
    let num_columns = column_end - column_start + 1;
    let results: Vec<_> = (0..record_count)
        .into_par_iter()
        .filter_map(|i| {
            let record_offset = records_start + i * record_size;
            let record_bytes = &mmap[record_offset..record_offset + record_size];
            
            match DynamicRecord::from_bytes(record_bytes, factor_count) {
                Ok(record) => {
                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&record.date) {
                            return None;
                        }
                    }
                    
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                    
                    // 检查代码过滤器
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&code) {
                            return None;
                        }
                    }
                    
                    // 获取指定列范围的因子值
                    let mut factor_values = Vec::with_capacity(num_columns);
                    for col_idx in column_start..=column_end {
                        let factor_value = if col_idx < record.factors.len() {
                            record.factors[col_idx]
                        } else {
                            f64::NAN
                        };
                        factor_values.push(factor_value);
                    }
                    
                    Some((record.date, code, factor_values))
                }
                Err(_) => {
                    // 记录损坏，返回None
                    None
                }
            }
        })
        .collect();
        
    let num_rows = results.len();
    let mut dates = Vec::with_capacity(num_rows);
    let mut codes = Vec::with_capacity(num_rows);
    let mut factors = Vec::with_capacity(num_rows);

    for (date, code, factor_values) in results {
        dates.push(date);
        codes.push(code);
        factors.push(factor_values);
    }

    // 创建Python字典
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("factors", numpy.call_method1("array", (factors,))?)?;

        Ok(dict.into())
    })
}

fn read_legacy_backup_results_columns_range_with_filter(file_path: &str, column_start: usize, column_end: usize, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    let mut file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取备份文件: {}", e)))?;
    
    if buffer.is_empty() {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item("code", numpy.call_method1("array", (Vec::<String>::new(),))?)?;
            dict.set_item("factors", numpy.call_method1("array", (Vec::<Vec<f64>>::new(),))?)?;
            Ok(dict.into())
        });
    }
    
    let mut cursor = 0;
    let mut all_results = Vec::new();
    
    // 尝试新的批次格式
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3],
            size_bytes[4], size_bytes[5], size_bytes[6], size_bytes[7],
        ]) as usize;
        
        cursor += 8;
        
        if cursor + batch_size > buffer.len() {
            cursor -= 8;
            break;
        }
        
        match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..cursor + batch_size]) {
            Ok(batch) => {
                for result in batch {
                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&result.date) {
                            continue;
                        }
                    }
                    
                    // 检查代码过滤器
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&result.code) {
                            continue;
                        }
                    }
                    
                    // 检查列索引是否有效
                    if column_start >= result.facs.len() {
                        continue;
                    }
                    
                    let actual_end = std::cmp::min(column_end, result.facs.len() - 1);
                    if actual_end < column_start {
                        continue;
                    }
                    
                    let num_columns = actual_end - column_start + 1;
                    let mut factor_values = Vec::with_capacity(num_columns);
                    for col_idx in column_start..=actual_end {
                        factor_values.push(result.facs[col_idx]);
                    }
                    
                    all_results.push((result.date, result.code, factor_values));
                }
                cursor += batch_size;
            }
            Err(_) => {
                cursor -= 8;
                break;
            }
        }
    }
    
    // 如果失败，尝试原始格式
    if cursor < buffer.len() {
        while cursor < buffer.len() {
            match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..]) {
                Ok(batch) => {
                    let batch_size = bincode::serialized_size(&batch).unwrap_or(0) as usize;
                    
                    for result in batch {
                        // 检查日期过滤器
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&result.date) {
                                continue;
                            }
                        }
                        
                        // 检查代码过滤器
                        if let Some(code_filter) = code_filter {
                            if !code_filter.contains(&result.code) {
                                continue;
                            }
                        }
                        
                        // 检查列索引是否有效
                        if column_start >= result.facs.len() {
                            continue;
                        }
                        
                        let actual_end = std::cmp::min(column_end, result.facs.len() - 1);
                        if actual_end < column_start {
                            continue;
                        }
                        
                        let num_columns = actual_end - column_start + 1;
                        let mut factor_values = Vec::with_capacity(num_columns);
                        for col_idx in column_start..=actual_end {
                            factor_values.push(result.facs[col_idx]);
                        }
                        
                        all_results.push((result.date, result.code, factor_values));
                    }
                    cursor += batch_size;
                }
                Err(_) => {
                    cursor += std::cmp::min(64, buffer.len() - cursor);
                }
            }
        }
    }
    
    // 整理结果
    let num_rows = all_results.len();
    let mut dates = Vec::with_capacity(num_rows);
    let mut codes = Vec::with_capacity(num_rows);
    let mut factors = Vec::with_capacity(num_rows);
    
    for (date, code, factor_values) in all_results {
        dates.push(date);
        codes.push(code);
        factors.push(factor_values);
    }
    
    // 创建Python字典
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("factors", numpy.call_method1("array", (factors,))?)?;

        Ok(dict.into())
    })
}

// 支持旧格式的单列读取
fn read_legacy_backup_results_single_column_with_filter(file_path: &str, column_index: usize, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    let mut file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取备份文件: {}", e)))?;
    
    if buffer.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件为空"
        ));
    }
    
    let mut all_results = Vec::new();
    let mut cursor = 0;
    
    // 尝试新的批次格式（带大小头）
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3],
            size_bytes[4], size_bytes[5], size_bytes[6], size_bytes[7],
        ]) as usize;
        
        cursor += 8;
        
        if cursor + batch_size > buffer.len() {
            cursor -= 8;
            break;
        }
        
        match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..cursor + batch_size]) {
            Ok(batch) => {
                for result in batch {
                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&result.date) {
                            continue;
                        }
                    }
                    
                    // 检查代码过滤器
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&result.code) {
                            continue;
                        }
                    }
                    
                    all_results.push(result);
                }
                cursor += batch_size;
            }
            Err(_) => {
                cursor -= 8;
                break;
            }
        }
    }
    
    // 如果失败，尝试原始格式
    if cursor < buffer.len() {
        while cursor < buffer.len() {
            match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..]) {
                Ok(batch) => {
                    let batch_size = bincode::serialized_size(&batch)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("序列化错误: {}", e)))? as usize;
                    for result in batch {
                        // 检查日期过滤器
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&result.date) {
                                continue;
                            }
                        }
                        
                        // 检查代码过滤器
                        if let Some(code_filter) = code_filter {
                            if !code_filter.contains(&result.code) {
                                continue;
                            }
                        }
                        
                        all_results.push(result);
                    }
                    cursor += batch_size;
                }
                Err(_) => {
                    cursor += std::cmp::min(64, buffer.len() - cursor);
                }
            }
        }
    }
    
    if all_results.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件中没有找到有效结果"
        ));
    }
    
    // 检查列索引是否有效
    if let Some(first_result) = all_results.first() {
        if column_index >= first_result.facs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("列索引 {} 超出范围，因子列数为 {}", column_index, first_result.facs.len())
            ));
        }
    }
    
    // 提取指定列的数据
    let mut dates = Vec::with_capacity(all_results.len());
    let mut codes = Vec::with_capacity(all_results.len());
    let mut factors = Vec::with_capacity(all_results.len());
    
    for result in all_results {
        dates.push(result.date);
        codes.push(result.code);
        let factor_value = if column_index < result.facs.len() {
            result.facs[column_index]
        } else {
            f64::NAN
        };
        factors.push(factor_value);
    }
    
    // 创建Python字典
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("factor", numpy.call_method1("array", (factors,))?)?;

        Ok(dict.into())
    })
}

/// 终极版本：线程安全的并行+零分配+缓存优化
#[allow(dead_code)]
fn read_backup_results_ultra_fast_v4(file_path: &str) -> PyResult<PyObject> {
    read_backup_results_ultra_fast_v4_with_filter(file_path, None, None)
}

fn read_backup_results_ultra_fast_v4_with_filter(file_path: &str, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "Backup file not found"
        ));
    }
    
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open backup file: {}", e)))?;
    
    let file_len = file.metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to get file metadata: {}", e)))?
        .len() as usize;
    
    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_with_filter(file_path, date_filter, code_filter);
    }
    
    // 内存映射
    let mmap = unsafe { 
        Mmap::map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to memory map file: {}", e)))?
    };
    let mmap = Arc::new(mmap);
    
    // 读取文件头
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    if &header.magic != b"RPBACKUP" {
        return read_legacy_backup_results_with_filter(file_path, date_filter, code_filter);
    }
    
    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| Ok(pyo3::types::PyDict::new(py).into()));
    }
    
    // --- BUG修复：使用文件头中的 record_size ---
    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;

    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
           format!("Record size mismatch: header says {}, calculated {}. File may be corrupt.", record_size, calculated_record_size)
       ));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Backup file appears to be truncated"
        ));
    }
    
    // --- 返回类型修改：并行收集为元组，再转换为Python字典 ---
    let records_start = HEADER_SIZE;
    let results: Vec<_> = (0..record_count)
        .into_par_iter()
        .filter_map(|i| {
            let record_offset = records_start + i * record_size;
            let record_bytes = &mmap[record_offset..record_offset + record_size];
            
            match DynamicRecord::from_bytes(record_bytes, factor_count) {
                Ok(record) => {
                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&record.date) {
                            return None;
                        }
                    }
                    
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                    
                    // 检查代码过滤器
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&code) {
                            return None;
                        }
                    }
                    
                    Some((record.date, code, record.timestamp, record.factors))
                }
                Err(_) => {
                    // 记录损坏，返回None而不是默认值
                    None
                }
            }
        })
        .collect();
        
    let num_rows = results.len();
    let mut dates = Vec::with_capacity(num_rows);
    let mut codes = Vec::with_capacity(num_rows);
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut factors_flat = Vec::with_capacity(num_rows * factor_count);

    for (date, code, timestamp, facs) in results {
        dates.push(date);
        codes.push(code);
        timestamps.push(timestamp);
        if facs.len() == factor_count {
            factors_flat.extend_from_slice(&facs);
        } else {
            factors_flat.resize(factors_flat.len() + factor_count, f64::NAN);
        }
    }

    // 创建Numpy数组字典
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("timestamp", numpy.call_method1("array", (timestamps,))?)?;
        
        let factors_array = numpy.call_method1("array", (factors_flat,))?;
        
        if num_rows > 0 && factor_count > 0 {
            let factors_reshaped = factors_array.call_method1("reshape", ((num_rows, factor_count),))?;
            dict.set_item("factors", factors_reshaped)?;
        } else {
            dict.set_item("factors", factors_array)?;
        }

        Ok(dict.into())
    })
}

fn run_persistent_task_worker(
    worker_id: usize,
    task_queue: Receiver<TaskParam>,
    python_code: String,
    expected_result_length: usize,
    python_path: String,
    result_sender: Sender<TaskResult>,
    restart_flag: Arc<AtomicBool>,
    monitor_manager: Arc<WorkerMonitorManager>,
) {
    // 向监控管理器注册worker
    monitor_manager.add_worker(worker_id);
    
    loop { // 循环以支持worker重启
        if restart_flag.compare_exchange(true, false, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
            // println!("🔄 Worker {} 检测到重启信号，正在重启...", worker_id);
        }

        // println!("🚀 Persistent Worker {} 启动，创建持久Python进程", worker_id);
        
        let script_content = create_persistent_worker_script();
        let script_path = format!("/tmp/persistent_worker_{}.py", worker_id);
        
        // 创建worker脚本
        if let Err(e) = std::fs::write(&script_path, script_content) {
            eprintln!("❌ Worker {} 创建脚本失败: {}", worker_id, e);
            continue; // 继续外层循环，尝试重新创建脚本
        }
        
        // 启动持久的Python子进程
        let mut child = match Command::new(&python_path)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                eprintln!("❌ Worker {} 启动Python进程失败: {}", worker_id, e);
                continue; // 继续外层循环，尝试重新启动进程
            }
        };
        
        // 设置子进程ID到监控管理器
        let pid = child.id();
        monitor_manager.set_worker_process_id(worker_id, pid);
        monitor_manager.update_heartbeat(worker_id);
        
        let mut stdin = child.stdin.take().expect("Failed to get stdin");
        let mut stdout = child.stdout.take().expect("Failed to get stdout");
        
        let mut task_count = 0;
        let mut needs_restart = false;

        // 持续从队列中取任务并发送给Python进程
        while let Ok(task) = task_queue.recv() {
            // 在处理任务前检查重启标志
            if restart_flag.load(Ordering::Relaxed) {
                needs_restart = true;
                break;
            }

            task_count += 1;
            
            // 通知监控管理器开始处理任务
            monitor_manager.start_task(worker_id, task.clone());
            monitor_manager.update_heartbeat(worker_id);
            
            // 创建单任务数据
            let single_task = SingleTask {
                python_code: python_code.clone(),
                task: task.clone(),
                expected_result_length,
            };
            
            // 序列化任务数据
            let packed_data = match rmp_serde::to_vec_named(&single_task) {
                Ok(data) => data,
                Err(_e) => {
                    // eprintln!("❌ Worker {} 任务 #{} 序列化失败: {}", worker_id, task_count, e);
                    continue;
                }
            };
            
            // 发送任务到Python进程（带长度前缀）
            let length = packed_data.len() as u32;
            let length_bytes = length.to_le_bytes();
            
            if let Err(_e) = stdin.write_all(&length_bytes) {
                // eprintln!("❌ Worker {} 发送长度前缀失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            if let Err(_e) = stdin.write_all(&packed_data) {
                // eprintln!("❌ Worker {} 发送任务数据失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            if let Err(_e) = stdin.flush() {
                // eprintln!("❌ Worker {} flush失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            // 读取结果（带长度前缀）
            let mut length_bytes = [0u8; 4];
            if let Err(_e) = stdout.read_exact(&mut length_bytes) {
                // eprintln!("❌ Worker {} 读取结果长度失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            let length = u32::from_le_bytes(length_bytes) as usize;
            let mut result_data = vec![0u8; length];
            
            if let Err(_e) = stdout.read_exact(&mut result_data) {
                // eprintln!("❌ Worker {} 读取结果数据失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            // 解析结果
            #[derive(Debug, Serialize, Deserialize)]
            struct SingleResult {
                result: TaskResult,
            }
            
            match rmp_serde::from_slice::<SingleResult>(&result_data) {
                Ok(single_result) => {
                    // 发送结果
                    if let Err(e) = result_sender.send(single_result.result) {
                        eprintln!("❌ Worker {} 任务 #{} 结果发送失败: {}", worker_id, task_count, e);
                        // 结果发送失败可能是收集器已退出，但不影响其他worker，继续处理下一个任务
                        // 不设置needs_restart，避免不必要的子进程重启
                    }
                    // 通知监控管理器任务已完成
                    monitor_manager.finish_task(worker_id);
                    monitor_manager.update_heartbeat(worker_id);
                }
                Err(e) => {
                    eprintln!("❌ Worker {} 任务 #{} 结果解析失败: {}", worker_id, task_count, e);
                    
                    // 发送NaN结果
                    let error_result = TaskResult {
                        date: task.date,
                        code: task.code,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        facs: vec![f64::NAN; expected_result_length],
                    };
                    
                    if let Err(e) = result_sender.send(error_result) {
                        eprintln!("❌ Worker {} 错误结果发送失败: {}", worker_id, e);
                        // 错误结果发送失败也不影响其他worker，继续处理下一个任务
                        // 不设置needs_restart，避免不必要的子进程重启
                    }
                    // 通知监控管理器任务已完成（即使失败）
                    monitor_manager.finish_task(worker_id);
                    monitor_manager.update_heartbeat(worker_id);
                }
            }
        }
        
        // 发送结束信号（长度为0）
        let _ = stdin.write_all(&[0u8; 4]);
        let _ = stdin.flush();
        
        // 等待子进程结束
        let _ = child.wait();
        
        // 清理临时文件
        let _ = std::fs::remove_file(&script_path);
        // println!("🏁 Persistent Worker {} 结束，共处理 {} 个任务", worker_id, task_count);
        
        if !needs_restart {
            // 如果不是因为重启信号而退出，说明所有任务都完成了
            break;
        }
    }
    
    // Worker完全结束时，从监控器中移除记录
    monitor_manager.remove_worker(worker_id);
}


#[pyfunction]
#[pyo3(signature = (python_function, args, n_jobs, backup_file, expected_result_length, restart_interval=None, update_mode=None, return_results=None, task_timeout=None, health_check_interval=None, debug_monitor=None))]
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
    let health_check_interval_secs = health_check_interval.unwrap_or(120);
    let debug_monitor_enabled = debug_monitor.unwrap_or(false);
    
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
        return if return_results_enabled {
            // 使用自定义线程池避免资源竞争
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(std::cmp::min(rayon::current_num_threads(), 4))
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建读取线程池失败: {}", e)))?;
            
            pool.install(|| {
                if update_mode_enabled {
                    // update_mode下，只返回传入参数中涉及的日期和代码
                    let task_dates: HashSet<i64> = all_tasks_clone.iter().map(|t| t.date).collect();
                    let task_codes: HashSet<String> = all_tasks_clone.iter().map(|t| t.code.clone()).collect();
                    read_backup_results_with_filter(&backup_file, Some(&task_dates), Some(&task_codes))
                } else {
                    read_backup_results(&backup_file)
                }
            })
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
    
    // 启动worker线程
    let mut worker_handles = Vec::new();
    for i in 0..n_jobs {
        let worker_task_receiver = task_receiver.clone();
        let worker_python_code = python_code.clone();
        let worker_python_path = python_path.clone();
        let worker_result_sender = result_sender.clone();
        let worker_restart_flag = restart_flag.clone();
        let worker_monitor_manager = monitor_manager.clone();
        
        let handle = thread::spawn(move || {
            run_persistent_task_worker(
                i,
                worker_task_receiver,
                worker_python_code,
                expected_result_length,
                worker_python_path,
                worker_result_sender,
                worker_restart_flag,
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
    let monitor_handle = thread::spawn(move || {
        loop {
            // 检查是否应该退出监控循环
            if monitor_manager_clone.should_stop_monitoring() {
                println!("[{}] 🔍 监控器: 接收到停止信号，正在退出监控循环", 
                         Local::now().format("%Y-%m-%d %H:%M:%S"));
                break;
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
            
            // 等待下一次检查
            thread::sleep(monitor_manager_clone.health_check_interval);
        }
    });
    
    // 启动结果收集器
    let backup_file_clone = backup_file.clone();
    let expected_result_length_clone = expected_result_length;
    let pending_tasks_len = pending_tasks.len();
    let collector_restart_flag = restart_flag.clone();
    let restart_interval_clone = restart_interval_value;
    let collector_handle = thread::spawn(move || {
        let mut batch_results = Vec::new();
        let mut total_collected = 0;
        let mut batch_count = 0;
        let mut batch_count_this_chunk = 0;
        let total_batches = (pending_tasks_len + 999) / 1000;
        
        println!("[{}] 🔄 结果收集器启动，等待worker结果...", Local::now().format("%Y-%m-%d %H:%M:%S"));
        
        while let Ok(result) = result_receiver.recv() {
            total_collected += 1;
            batch_results.push(result);
            
            // println!("📥 收到任务结果: date={}, code={}, 当前批次: {}, 总收集: {}", 
            //          batch_results.last().unwrap().date,
            //          batch_results.last().unwrap().code,
            //          batch_results.len(), 
            //          total_collected);
            
            // 每1000个结果备份一次
            if batch_results.len() >= 1000 {
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
                print!("\r[{}] 💾 第 {}/{} 次备份，存{}个结果。已用{}小时{}分钟{}秒，预余{}小时{}分钟{}秒", 
                       current_time, batch_count, total_batches, batch_results.len(),
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

/// 查询备份文件中的指定列
/// 
/// 参数:
/// - backup_file: 备份文件路径
/// - column_index: 要读取的因子列索引（0表示第一列因子值）
/// 
/// 返回:
/// 包含三个numpy数组的字典: {"date": 日期数组, "code": 代码数组, "factor": 指定列的因子值数组}
#[pyfunction]
#[pyo3(signature = (backup_file, column_index))]
pub fn query_backup_single_column(backup_file: String, column_index: usize) -> PyResult<PyObject> {
    // 优先使用超高速版本
    read_backup_results_single_column_ultra_fast_v2(&backup_file, column_index)
}

/// 查询备份文件中的指定列，支持过滤
/// 
/// 参数:
/// - backup_file: 备份文件路径
/// - column_index: 要读取的因子列索引（0表示第一列因子值）
/// - dates: 可选的日期过滤列表
/// - codes: 可选的代码过滤列表
/// 
/// 返回:
/// 包含三个numpy数组的字典: {"date": 日期数组, "code": 代码数组, "factor": 指定列的因子值数组}
#[pyfunction]
#[pyo3(signature = (backup_file, column_index, dates=None, codes=None))]
pub fn query_backup_single_column_with_filter(backup_file: String, column_index: usize, dates: Option<Vec<i64>>, codes: Option<Vec<String>>) -> PyResult<PyObject> {
    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());
    
    read_backup_results_single_column_with_filter(&backup_file, column_index, date_filter.as_ref(), code_filter.as_ref())
}

/// 查询备份文件中的指定列范围，支持过滤
/// 
/// 参数:
/// - backup_file: 备份文件路径
/// - column_start: 开始列索引（包含）
/// - column_end: 结束列索引（包含）
/// - dates: 可选的日期过滤列表
/// - codes: 可选的代码过滤列表
/// 
/// 返回:
/// 包含numpy数组的字典: {"date": 日期数组, "code": 代码数组, "factors": 指定列范围的因子值数组}
#[pyfunction]
#[pyo3(signature = (backup_file, column_start, column_end, dates=None, codes=None))]
pub fn query_backup_columns_range_with_filter(backup_file: String, column_start: usize, column_end: usize, dates: Option<Vec<i64>>, codes: Option<Vec<String>>) -> PyResult<PyObject> {
    // 检查参数有效性
    if column_start > column_end {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "column_start must be <= column_end"
        ));
    }
    
    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());
    
    read_backup_results_columns_range_with_filter(&backup_file, column_start, column_end, date_filter.as_ref(), code_filter.as_ref())
}

/// 读取备份文件中的指定列因子值（纯因子值数组）
/// column_index: 要读取的因子列索引（0表示第一列因子值）
/// 返回: 只包含因子值的numpy数组
fn read_backup_results_factor_only(file_path: &str, column_index: usize) -> PyResult<PyObject> {
    read_backup_results_factor_only_with_filter(file_path, column_index, None, None)
}

fn read_backup_results_factor_only_with_filter(file_path: &str, column_index: usize, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在"
        ));
    }
    
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let file_len = file.metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e)))?
        .len() as usize;
    
    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_factor_only_with_filter(file_path, column_index, date_filter, code_filter);
    }
    
    // 内存映射
    let mmap = unsafe { 
        Mmap::map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e)))?
    };
    let mmap = Arc::new(mmap);
    
    // 读取文件头
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    if &header.magic != b"RPBACKUP" {
        return read_legacy_backup_results_factor_only_with_filter(file_path, column_index, date_filter, code_filter);
    }
    
    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            Ok(numpy.call_method1("array", (Vec::<f64>::new(),))?.into())
        });
    }
    
    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;
    
    // 检查列索引是否有效
    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("列索引 {} 超出范围，因子列数为 {}", column_index, factor_count)
        ));
    }
    
    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
           format!("记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.", record_size, calculated_record_size)
       ));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了"
        ));
    }
    
    // 使用自定义线程池避免资源竞争和泄漏
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 8))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e)))?;
    
    // 并行读取只获取因子值
    let records_start = HEADER_SIZE;
    let factors: Vec<f64> = pool.install(|| {
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];
                
                match DynamicRecord::from_bytes(record_bytes, factor_count) {
                    Ok(record) => {
                        // 检查日期过滤器
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&record.date) {
                                return None;
                            }
                        }
                        
                        // 检查代码过滤器
                        if let Some(code_filter) = code_filter {
                            let code_len = std::cmp::min(record.code_len as usize, 32);
                            let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                            
                            if !code_filter.contains(&code) {
                                return None;
                            }
                        }
                        
                        // 只返回指定列的因子值
                        if column_index < record.factors.len() {
                            Some(record.factors[column_index])
                        } else {
                            Some(f64::NAN)
                        }
                    }
                    Err(_) => {
                        // 记录损坏，返回NaN
                        Some(f64::NAN)
                    }
                }
            })
            .collect()
    });
    
    // 显式释放mmap
    drop(mmap);

    // 创建numpy数组
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        Ok(numpy.call_method1("array", (factors,))?.into())
    })
}

// 支持旧格式的纯因子值读取
fn read_legacy_backup_results_factor_only_with_filter(file_path: &str, column_index: usize, date_filter: Option<&HashSet<i64>>, code_filter: Option<&HashSet<String>>) -> PyResult<PyObject> {
    let mut file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取备份文件: {}", e)))?;
    
    if buffer.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件为空"
        ));
    }
    
    let mut all_results = Vec::new();
    let mut cursor = 0;
    
    // 尝试新的批次格式（带大小头）
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3],
            size_bytes[4], size_bytes[5], size_bytes[6], size_bytes[7],
        ]) as usize;
        
        cursor += 8;
        
        if cursor + batch_size > buffer.len() {
            cursor -= 8;
            break;
        }
        
        match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..cursor + batch_size]) {
            Ok(batch) => {
                for result in batch {
                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&result.date) {
                            continue;
                        }
                    }
                    
                    // 检查代码过滤器
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&result.code) {
                            continue;
                        }
                    }
                    
                    all_results.push(result);
                }
                cursor += batch_size;
            }
            Err(_) => {
                cursor -= 8;
                break;
            }
        }
    }
    
    // 如果失败，尝试原始格式
    if cursor < buffer.len() {
        while cursor < buffer.len() {
            match bincode::deserialize::<Vec<TaskResult>>(&buffer[cursor..]) {
                Ok(batch) => {
                    let batch_size = bincode::serialized_size(&batch)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("序列化错误: {}", e)))? as usize;
                    for result in batch {
                        // 检查日期过滤器
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&result.date) {
                                continue;
                            }
                        }
                        
                        // 检查代码过滤器
                        if let Some(code_filter) = code_filter {
                            if !code_filter.contains(&result.code) {
                                continue;
                            }
                        }
                        
                        all_results.push(result);
                    }
                    cursor += batch_size;
                }
                Err(_) => {
                    cursor += std::cmp::min(64, buffer.len() - cursor);
                }
            }
        }
    }
    
    if all_results.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件中没有找到有效结果"
        ));
    }
    
    // 检查列索引是否有效
    if let Some(first_result) = all_results.first() {
        if column_index >= first_result.facs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("列索引 {} 超出范围，因子列数为 {}", column_index, first_result.facs.len())
            ));
        }
    }
    
    // 只提取指定列的因子值
    let factors: Vec<f64> = all_results
        .into_iter()
        .map(|result| {
            if column_index < result.facs.len() {
                result.facs[column_index]
            } else {
                f64::NAN
            }
        })
        .collect();
    
    // 创建numpy数组
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        Ok(numpy.call_method1("array", (factors,))?.into())
    })
}

/// 查询备份文件中的指定列因子值（纯因子值数组）
/// 
/// 参数:
/// - backup_file: 备份文件路径
/// - column_index: 要读取的因子列索引（0表示第一列因子值）
/// 
/// 返回:
/// 只包含因子值的numpy数组
#[pyfunction]
#[pyo3(signature = (backup_file, column_index))]
pub fn query_backup_factor_only(backup_file: String, column_index: usize) -> PyResult<PyObject> {
    read_backup_results_factor_only(&backup_file, column_index)
}

/// 查询备份文件中的指定列因子值（纯因子值数组），支持过滤
/// 
/// 参数:
/// - backup_file: 备份文件路径
/// - column_index: 要读取的因子列索引（0表示第一列因子值）
/// - dates: 可选的日期过滤列表
/// - codes: 可选的代码过滤列表
/// 
/// 返回:
/// 只包含因子值的numpy数组
#[pyfunction]
#[pyo3(signature = (backup_file, column_index, dates=None, codes=None))]
pub fn query_backup_factor_only_with_filter(backup_file: String, column_index: usize, dates: Option<Vec<i64>>, codes: Option<Vec<String>>) -> PyResult<PyObject> {
    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());
    
    read_backup_results_factor_only_with_filter(&backup_file, column_index, date_filter.as_ref(), code_filter.as_ref())
}

/// 超高速因子值读取（直接字节偏移版本）
fn read_backup_results_factor_only_ultra_fast(file_path: &str, column_index: usize) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在"
        ));
    }
    
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let file_len = file.metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e)))?
        .len() as usize;
    
    if file_len < HEADER_SIZE {
        return read_backup_results_factor_only(&file_path, column_index);
    }
    
    // 内存映射
    let mmap = unsafe { 
        Mmap::map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e)))?
    };
    
    // 读取文件头
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    if &header.magic != b"RPBACKUP" {
        return read_backup_results_factor_only(&file_path, column_index);
    }
    
    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            Ok(numpy.call_method1("array", (Vec::<f64>::new(),))?.into())
        });
    }
    
    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;
    
    // 检查列索引是否有效
    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("列索引 {} 超出范围，因子列数为 {}", column_index, factor_count)
        ));
    }
    
    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
           format!("记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.", record_size, calculated_record_size)
       ));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了"
        ));
    }
    
    // 直接偏移读取因子值
    let records_start = HEADER_SIZE;
    
    // 计算因子值在记录中的偏移量
    // 记录格式: date(8) + code_hash(8) + timestamp(8) + factor_count(4) + code_len(4) + code_bytes(32) + factors(factor_count * 8)
    let factor_base_offset = 8 + 8 + 8 + 4 + 4 + 32; // date + code_hash + timestamp + factor_count + code_len + code_bytes
    let factor_offset = factor_base_offset + column_index * 8;
    
    // 使用自定义线程池避免资源竞争和泄漏
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 16))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e)))?;
    
    // 并行读取所有因子值
    let factors: Vec<f64> = pool.install(|| {
        (0..record_count)
            .into_par_iter()
            .map(|i| {
                let record_offset = records_start + i * record_size;
                
                // 直接读取因子值，完全跳过其他字段的解析
                unsafe {
                    let factor_ptr = mmap.as_ptr().add(record_offset + factor_offset) as *const f64;
                    *factor_ptr
                }
            })
            .collect()
    });
    
    // 显式释放mmap
    drop(mmap);
    
    // 创建numpy数组
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        Ok(numpy.call_method1("array", (factors,))?.into())
    })
}

/// 超高速查询备份文件中的指定列因子值
/// 
/// 参数:
/// - backup_file: 备份文件路径
/// - column_index: 要读取的因子列索引（0表示第一列因子值）
/// 
/// 返回:
/// 只包含因子值的numpy数组
#[pyfunction]
#[pyo3(signature = (backup_file, column_index))]
pub fn query_backup_factor_only_ultra_fast(backup_file: String, column_index: usize) -> PyResult<PyObject> {
    read_backup_results_factor_only_ultra_fast(&backup_file, column_index)
}

/// 超高速查询备份文件中的指定列（完整版本v2）
/// 直接字节偏移读取，避免完整记录解析
fn read_backup_results_single_column_ultra_fast_v2(file_path: &str, column_index: usize) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在"
        ));
    }
    
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e)))?;
    
    let file_len = file.metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e)))?
        .len() as usize;
    
    if file_len < HEADER_SIZE {
        return read_backup_results_single_column(&file_path, column_index);
    }
    
    // 内存映射
    let mmap = unsafe { 
        Mmap::map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e)))?
    };
    
    // 读取文件头
    let header = unsafe {
        &*(mmap.as_ptr() as *const FileHeader)
    };
    
    if &header.magic != b"RPBACKUP" {
        return read_backup_results_single_column(&file_path, column_index);
    }
    
    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item("code", numpy.call_method1("array", (Vec::<String>::new(),))?)?;
            dict.set_item("factor", numpy.call_method1("array", (Vec::<f64>::new(),))?)?;
            Ok(dict.into())
        });
    }
    
    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;
    
    // 检查列索引是否有效
    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("列索引 {} 超出范围，因子列数为 {}", column_index, factor_count)
        ));
    }
    
    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
           format!("记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.", record_size, calculated_record_size)
       ));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了"
        ));
    }
    
    // 计算各字段在记录中的偏移量
    // 记录格式: date(8) + code_hash(8) + timestamp(8) + factor_count(4) + code_len(4) + code_bytes(32) + factors(factor_count * 8)
    let date_offset = 0;
    let code_len_offset = 8 + 8 + 8 + 4; // date + code_hash + timestamp + factor_count
    let code_bytes_offset = code_len_offset + 4; // + code_len
    let factor_base_offset = 8 + 8 + 8 + 4 + 4 + 32; // date + code_hash + timestamp + factor_count + code_len + code_bytes
    let factor_offset = factor_base_offset + column_index * 8;
    
    let records_start = HEADER_SIZE;
    
    // 使用更大的线程池以提高并行度
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 16))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e)))?;
    
    // 预先分配Vec避免多次重新分配
    let mut dates = Vec::with_capacity(record_count);
    let mut codes = Vec::with_capacity(record_count);
    let mut factors = Vec::with_capacity(record_count);
    
    // 使用更大的批次来减少同步开销
    const BATCH_SIZE: usize = 10000;
    let num_batches = (record_count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    pool.install(|| {
        // 并行处理每个批次
        let batch_results: Vec<Vec<(i64, String, f64)>> = (0..num_batches)
            .into_par_iter()
            .map(|batch_idx| {
                let start_idx = batch_idx * BATCH_SIZE;
                let end_idx = std::cmp::min(start_idx + BATCH_SIZE, record_count);
                let mut batch_data = Vec::with_capacity(end_idx - start_idx);
                
                for i in start_idx..end_idx {
                    let record_offset = records_start + i * record_size;
                    
                    // 直接读取日期
                    let date = unsafe {
                        let date_ptr = mmap.as_ptr().add(record_offset + date_offset) as *const i64;
                        *date_ptr
                    };
                    
                    // 直接读取代码长度
                    let code_len = unsafe {
                        let code_len_ptr = mmap.as_ptr().add(record_offset + code_len_offset) as *const u32;
                        std::cmp::min(*code_len_ptr as usize, 32)
                    };
                    
                    // 直接读取代码字节
                    let code = unsafe {
                        let code_bytes_ptr = mmap.as_ptr().add(record_offset + code_bytes_offset);
                        let code_slice = std::slice::from_raw_parts(code_bytes_ptr, code_len);
                        String::from_utf8_lossy(code_slice).into_owned()
                    };
                    
                    // 直接读取因子值
                    let factor = unsafe {
                        let factor_ptr = mmap.as_ptr().add(record_offset + factor_offset) as *const f64;
                        *factor_ptr
                    };
                    
                    batch_data.push((date, code, factor));
                }
                
                batch_data
            })
            .collect();
        
        // 合并所有批次结果
        for batch in batch_results {
            for (date, code, factor) in batch {
                dates.push(date);
                codes.push(code);
                factors.push(factor);
            }
        }
    });
    
    // 显式释放mmap
    drop(mmap);
    
    // 创建Python字典
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("factor", numpy.call_method1("array", (factors,))?)?;

        Ok(dict.into())
    })
}