use pyo3::prelude::*;
use std::process::{Command, Stdio, Child};
use std::io::{Write, BufRead, BufReader};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::env;
use crossbeam::channel::{unbounded, Receiver as CrossbeamReceiver, Sender as CrossbeamSender};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use crate::backup::BackupManager;
use std::sync::OnceLock;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};


/// 全局日志收集器
static LOG_COLLECTOR: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();




/// 添加日志消息
fn log_message(message: String) {
    let collector = LOG_COLLECTOR.get_or_init(|| Arc::new(Mutex::new(Vec::new())));
    if let Ok(mut logs) = collector.lock() {
        logs.push(message);
    }
}

/// 计算字符串的哈希值
fn calculate_hash(input: &str) -> String {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// 通过Python输出并清空所有日志
pub fn flush_logs_to_python(py: Python) {
    let collector = LOG_COLLECTOR.get_or_init(|| Arc::new(Mutex::new(Vec::new())));
    if let Ok(mut logs) = collector.lock() {
        if let Ok(builtins) = py.import("builtins") {
            for log in logs.iter() {
                let _ = builtins.call_method1("print", (log,));
            }
        }
        logs.clear();
    }
}

/// 诊断工作进程状态的函数
fn diagnose_process_status(worker_id: usize, child: &mut std::process::Child) {
    match child.try_wait() {
        Ok(Some(status)) => {
            let status_msg = if status.success() {
                format!("✅ 工作进程 {} 正常退出，退出码: {:?}", worker_id, status.code())
            } else {
                format!("❌ 工作进程 {} 异常退出，退出状态: {:?}", worker_id, status)
            };
            log_message(status_msg.clone());
            eprintln!("{}", status_msg);
            
            // 尝试读取stderr获取错误信息
            if let Some(ref mut stderr) = child.stderr.as_mut() {
                use std::io::Read;
                let mut stderr_output = String::new();
                if let Ok(_) = stderr.read_to_string(&mut stderr_output) {
                    if !stderr_output.trim().is_empty() {
                        let stderr_msg = format!("🚨 工作进程 {} 错误输出:\n{}", worker_id, stderr_output.trim());
                        log_message(stderr_msg.clone());
                        eprintln!("{}", stderr_msg);
                    } else {
                        log_message(format!("📝 工作进程 {} 没有stderr输出", worker_id));
                    }
                } else {
                    log_message(format!("⚠️ 无法读取工作进程 {} 的stderr", worker_id));
                }
            } else {
                log_message(format!("⚠️ 工作进程 {} 没有stderr管道", worker_id));
            }
        }
        Ok(None) => {
            let running_msg = format!("🔄 工作进程 {} 仍在运行，可能卡住了", worker_id);
            log_message(running_msg.clone());
            eprintln!("{}", running_msg);
            
            // 尝试获取进程的PID和其他信息
            let id = child.id();
            let pid_msg = format!("🆔 工作进程 {} 的PID: {}", worker_id, id);
            log_message(pid_msg.clone());
            eprintln!("{}", pid_msg);
        }
        Err(e) => {
            let err_msg = format!("🚫 无法检查工作进程 {} 状态: {}", worker_id, e);
            log_message(err_msg.clone());
            eprintln!("{}", err_msg);
        }
    }
}



/// 任务数据结构
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Task {
    pub date: i32,
    pub code: String,
}

/// 发送给工作进程的指令
#[derive(Serialize, Deserialize, Debug)]
pub enum WorkerCommand {
    Task(Task),
    FunctionCode(String),
    Execute {},
    Ping {},
    Exit {},
}

/// 工作进程请求
#[derive(Serialize, Deserialize, Debug)]
pub struct WorkerRequest {
    pub tasks: Vec<Task>,
    pub function_code: String,
    pub go_class_serialized: Option<String>,
}


/// 工作进程响应
#[derive(Serialize, Deserialize, Debug)]
pub struct WorkerResponse {
    pub results: Vec<Vec<Value>>,  // 使用JSON Value来支持混合类型
    pub errors: Vec<String>,
    pub task_count: usize,
}

/// Ping 响应
#[derive(Serialize, Deserialize, Debug)]
pub struct PingResponse {
    pub status: String,
}

/// 计算结果
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProcessResult {
    pub date: i32,
    pub code: String,
    pub timestamp: i64,
    pub facs: Vec<f64>,
}

/// 返回结果（不含timestamp）
#[derive(Clone, Debug)]
pub struct ReturnResult {
    pub date: i32,
    pub code: String,
    pub facs: Vec<f64>,
}

impl ProcessResult {
    pub fn to_return_result(&self) -> ReturnResult {
        ReturnResult {
            date: self.date,
            code: self.code.clone(),
            facs: self.facs.clone(),
        }
    }
}

/// 工作进程管理器
pub struct WorkerProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout_reader: BufReader<std::process::ChildStdout>,
    id: usize,
    function_code_hash: Option<String>,  // 跟踪已发送的函数代码版本
}

/// 任务分发器状态
#[derive(Debug)]
struct TaskDispatcherState {
    completed_tasks: usize,
    dispatched_tasks: usize,
}

/// 进度更新信息
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub completed: usize,
    pub total: usize,
    pub elapsed_secs: f64,
    pub estimated_remaining_secs: f64,
}

impl WorkerProcess {
    /// 创建新的工作进程
    pub fn new(id: usize, python_path: &str) -> PyResult<Self> {
        // 获取工作脚本路径 - 使用绝对路径避免当前目录问题
        let mut script_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        script_path.push("python");
        script_path.push("worker_process.py");
        
        log_message(format!("工作进程 {} - 脚本路径: {}", id, script_path.display()));
        
        // 检查脚本文件是否存在
        if !script_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("工作进程脚本不存在: {:?}", script_path)
            ));
        }
            
        // 创建Python工作进程
        let mut child = Command::new(python_path)
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("启动工作进程失败: {}", e)
            ))?;

        let stdin = child.stdin.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("无法获取进程stdin")
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("无法获取进程stdout")
        })?;

        let stdout_reader = BufReader::new(stdout);

        let worker = WorkerProcess {
            child,
            stdin,
            stdout_reader,
            id,
            function_code_hash: None,  // 初始化为None，将在发送函数代码时更新
        };

        // 给工作进程一些时间启动
        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(worker)
    }

    /// 发送函数代码并更新哈希值
    pub fn send_function_code(&mut self, function_code: &str) -> PyResult<()> {
        let new_hash = calculate_hash(function_code);
        
        // 检查是否需要发送（函数代码未改变）
        if let Some(ref current_hash) = self.function_code_hash {
            if current_hash == &new_hash {
                // 函数代码未改变，跳过发送
                return Ok(());
            }
        }
        
        // 发送函数代码
        self.send_command(&WorkerCommand::FunctionCode(function_code.to_string()))?;
        
        // 更新哈希值
        self.function_code_hash = Some(new_hash);
        
        Ok(())
    }

    /// 向工作进程发送指令
    pub fn send_command(&mut self, command: &WorkerCommand) -> PyResult<()> {
        match self.child.try_wait() {
            Ok(Some(status)) => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("工作进程已退出，状态码: {:?}", status)
                ));
            }
            Ok(None) => {
                // 进程仍在运行，继续
            }
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("检查进程状态失败: {}", e)
                ));
            }
        }

        let json_command = serde_json::to_string(command)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("序列化指令失败: {}", e)
            ))?;

        if let Err(e) = writeln!(self.stdin, "{}", json_command) {
            // 尝试获取进程的stderr信息
            let mut stderr_output = String::new();
            if let Some(ref mut stderr) = self.child.stderr.as_mut() {
                use std::io::Read;
                let _ = stderr.read_to_string(&mut stderr_output);
            }
            
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("向工作进程 {} 写入失败: {}. Stderr: {}", self.id, e, stderr_output)
            ));
        }
        Ok(())
    }

    /// 从工作进程接收通用响应
    fn receive_response<T: for<'de> serde::Deserialize<'de>>(&mut self) -> PyResult<T> {
        let mut line = String::new();

        self.stdout_reader.read_line(&mut line)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("从工作进程 {} 读取失败: {}", self.id, e)
            ))?;
        
        if line.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("从工作进程 {} 读取到空行，可能已退出", self.id)
            ));
        }
        
        serde_json::from_str(&line.trim())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("反序列化来自工作进程 {} 的响应失败: {}. 响应内容: '{}'", self.id, e, line.trim())
            ))
    }

    /// 从工作进程接收结果
    pub fn receive_result(&mut self) -> PyResult<WorkerResponse> {
        self.receive_response::<WorkerResponse>()
    }

    /// ping工作进程
    pub fn ping(&mut self) -> PyResult<()> {
        self.send_command(&WorkerCommand::Ping {})?;
        let response: PingResponse = self.receive_response::<PingResponse>()?;
        if response.status == "pong" {
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Ping 失败，收到未知响应"
            ))
        }
    }

    /// 终止工作进程
    pub fn terminate(&mut self) -> PyResult<()> {
        // 首先尝试优雅关闭，忽略发送错误（因为进程可能已经关闭）
        let _ = self.send_command(&WorkerCommand::Exit {});
        
        // 等待一小段时间让进程自己退出
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        // 如果进程还在运行，强制杀死
        match self.child.try_wait() {
            Ok(Some(_)) => {
                // 进程已经退出
            }
            Ok(None) => {
                // 进程仍在运行，强制杀死
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
            Err(_) => {
                // 检查状态失败，直接杀死
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
        
        Ok(())
    }
}

/// 多进程池管理器
pub struct ProcessPool {
    workers: Vec<WorkerProcess>,
    num_processes: usize,
    python_path: String,
}

impl ProcessPool {
    /// 创建新的进程池
    pub fn new(num_processes: usize, python_path: &str) -> PyResult<Self> {
        let mut workers = Vec::new();
        
        log_message(format!("创建 {} 个工作进程...", num_processes));
        log_message(format!("使用Python解释器: {}", python_path));
        
        for i in 0..num_processes {
            log_message(format!("正在创建工作进程 {}...", i));
            let worker = WorkerProcess::new(i, python_path)?;
            log_message(format!("工作进程 {} 创建成功", i));
            workers.push(worker);
        }
        
        log_message("进程池创建完成".to_string());
        
        Ok(ProcessPool {
            workers,
            num_processes,
            python_path: python_path.to_string(),
        })
    }

    /// 异步流水线执行所有任务
    pub fn execute_tasks_async(
        &mut self,
        _py: Python,
        function_code: &str,
        tasks: Vec<Task>,
        backup_sender: Option<Sender<ProcessResult>>,
    ) -> PyResult<Vec<ProcessResult>> {
        let total_tasks = tasks.len();
        if total_tasks == 0 {
            return Ok(Vec::new());
        }

        log_message(format!("开始异步流水线执行，总任务数: {}", total_tasks));
        let start_time = Instant::now();

        // 创建任务队列和结果收集通道
        let (task_sender, task_receiver): (CrossbeamSender<Task>, CrossbeamReceiver<Task>) = unbounded();
        let (result_sender, result_receiver) = channel::<(usize, ProcessResult)>();
        
        // 将所有任务放入队列
        for task in tasks {
            task_sender.send(task).unwrap();
        }
        drop(task_sender); // 关闭发送端，表示没有更多任务

        // 共享状态
        let state = Arc::new(Mutex::new(TaskDispatcherState {
            completed_tasks: 0,
            dispatched_tasks: 0,
        }));

        // 1. 初始化所有工作进程
        for (i, worker) in &mut self.workers.iter_mut().enumerate() {
            if let Err(e) = worker.ping() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("工作进程 {} 无响应: {}", i, e)
                ));
            }
            worker.send_function_code(function_code)?;
        }

        // 2. 启动工作进程线程
        let workers_drained: Vec<_> = self.workers.drain(..).collect();
        let mut worker_handles = Vec::new();

        for mut worker in workers_drained {
            let task_receiver = task_receiver.clone();
            let result_sender = result_sender.clone();
            let state = Arc::clone(&state);
            
            let handle = thread::spawn(move || {
                Self::worker_loop(worker.id, &mut worker, task_receiver, result_sender, state)
            });
            worker_handles.push(handle);
        }

        // 3. 启动结果收集线程（流式处理，不累积结果）
        let state_for_collection = Arc::clone(&state);
        let collection_handle = thread::spawn(move || {
            Self::result_collection_loop(result_receiver, backup_sender, state_for_collection, total_tasks)
        });

        // 4. 等待所有工作完成，不再占用主进程处理进度回调
        loop {
            // 检查工作线程是否都完成了
            let workers_finished = worker_handles.iter().all(|h| h.is_finished());
            
            if workers_finished {
                break;
            }
            
            thread::sleep(std::time::Duration::from_millis(100));
        }
        
        // 等待结果收集完成
        let _ = collection_handle.join();

        // 重建进程池
        for i in 0..self.num_processes {
            let new_worker = WorkerProcess::new(i, &self.python_path)?;
            self.workers.push(new_worker);
        }

        log_message(format!("异步流水线执行完成，总耗时: {:.2}秒", start_time.elapsed().as_secs_f64()));
        
        // 流式处理：结果已经写入备份文件，返回空结果
        // 调用方应该从备份文件读取结果
        Ok(Vec::new())
    }

    /// 工作进程循环：持续从队列获取任务并处理
    fn worker_loop(
        worker_id: usize,
        worker: &mut WorkerProcess,
        task_receiver: CrossbeamReceiver<Task>,
        result_sender: Sender<(usize, ProcessResult)>,
        state: Arc<Mutex<TaskDispatcherState>>,
    ) {
        loop {
            // 从队列获取任务
            match task_receiver.recv() {
                Ok(task) => {
                    // 更新分发计数
                    {
                        let mut state = state.lock().unwrap();
                        state.dispatched_tasks += 1;
                    }

                    // 保存任务信息用于构建结果，避免clone
                    let task_date = task.date;
                    let task_code = task.code.clone();
                    
                    // 发送任务给工作进程
                    // log_message(format!("工作进程 {} 发送任务: date={}, code={}", worker_id, task_date, task_code));
                    if let Err(e) = worker.send_command(&WorkerCommand::Task(task)) {
                        eprintln!("工作进程 {} 发送任务失败: {}", worker_id, e);
                        break;
                    }
                    
                    // log_message(format!("工作进程 {} 发送执行指令", worker_id));
                    if let Err(e) = worker.send_command(&WorkerCommand::Execute {}) {
                        eprintln!("工作进程 {} 发送执行指令失败: {}", worker_id, e);
                        break;
                    }

                    // 接收结果（添加超时和详细日志）
                    // log_message(format!("工作进程 {} 开始接收任务结果...", worker_id));
                    match worker.receive_result() {
                        Ok(response) => {
                            log_message(format!("工作进程 {} 成功接收结果", worker_id));
                            if !response.errors.is_empty() {
                                for error_msg in &response.errors {
                                    log_message(format!("⚠️ 工作进程 {} 返回错误: {}", worker_id, error_msg));
                                    eprintln!("⚠️ 工作进程 {} 返回错误: {}", worker_id, error_msg);
                                }
                            }

                            // 处理结果（应该只有一个结果，因为我们一次只发送一个任务）
                            let raw_values = response.results.into_iter().next().unwrap_or_else(|| Vec::new());
                            // 将JSON Value转换为f64，识别特殊的NaN标记
                            let facs: Vec<f64> = raw_values.into_iter()
                                .map(|value| match value {
                                    Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                                    Value::String(s) if s == "__NaN__" => f64::NAN,
                                    _ => f64::NAN,  // 其他类型都转为NaN
                                })
                                .collect();
                            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
                            let result = ProcessResult {
                                date: task_date,
                                code: task_code,
                                timestamp: now,
                                facs,
                            };
                            
                            if result_sender.send((worker_id, result)).is_err() {
                                eprintln!("工作进程 {} 发送结果失败，结果收集器可能已关闭", worker_id);
                                break;
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("❌ 工作进程 {} 接收结果失败: {}", worker_id, e);
                            eprintln!("{}", error_msg);
                            log_message(error_msg.clone());
                            
                            // 使用诊断函数检查进程状态
                            log_message(format!("🔍 检查工作进程 {} 的状态...", worker_id));
                            diagnose_process_status(worker_id, &mut worker.child);
                            
                            // 尝试读取进程的stderr以获取更多错误信息
                            if let Some(ref mut stderr) = worker.child.stderr.as_mut() {
                                use std::io::Read;
                                let mut stderr_output = String::new();
                                if let Ok(_) = stderr.read_to_string(&mut stderr_output) {
                                    if !stderr_output.trim().is_empty() {
                                        let stderr_msg = format!("🚨 工作进程 {} stderr输出:\n{}", worker_id, stderr_output.trim());
                                        eprintln!("{}", stderr_msg);
                                        log_message(stderr_msg);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
                Err(_) => {
                    // 任务队列已关闭，退出循环
                    break;
                }
            }
        }
        
        // 清理工作进程
        let _ = worker.terminate();
        log_message(format!("工作进程 {} 已退出", worker_id));
    }

    /// 结果收集循环（流式处理，不在内存中累积）
    fn result_collection_loop(
        result_receiver: Receiver<(usize, ProcessResult)>,
        backup_sender: Option<Sender<ProcessResult>>,
        state: Arc<Mutex<TaskDispatcherState>>,
        total_tasks: usize,
    ) {
        while let Ok((_worker_id, result)) = result_receiver.recv() {
            // 直接发送到备份线程（流式处理核心：不在内存中存储）
            if let Some(ref sender) = backup_sender {
                let _ = sender.send(result);
            }

            // 更新完成计数
            let completed = {
                let mut state = state.lock().unwrap();
                state.completed_tasks += 1;
                state.completed_tasks
            };

            // 如果所有任务都完成了，退出
            if completed >= total_tasks {
                break;
            }
        }
    }


}

impl Drop for ProcessPool {
    fn drop(&mut self) {
        // 终止所有工作进程
        for worker in &mut self.workers {
            let _ = worker.terminate();
        }
    }
}

/// 多进程执行配置
#[derive(Clone)]
pub struct MultiProcessConfig {
    pub num_processes: Option<usize>,
    pub backup_batch_size: usize,
    pub backup_file: Option<String>,
    pub storage_format: String,
    pub resume_from_backup: bool,
    pub python_path: String,
    /// 流式处理配置：每多少批次强制fsync（0表示每批都fsync）
    pub fsync_frequency: usize,
    /// 是否强制使用备份文件（流式处理模式下建议开启）
    pub require_backup: bool,
}

impl Default for MultiProcessConfig {
    fn default() -> Self {
        Self {
            num_processes: None,
            backup_batch_size: 50, // 流式处理：降低批处理大小
            backup_file: None,
            storage_format: "binary".to_string(),
            resume_from_backup: false,
            python_path: "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python".to_string(),
            fsync_frequency: 10, // 每10批强制fsync一次
            require_backup: false, // 暂时设为false，避免在没有backup_file时报错
        }
    }
}



/// 多进程执行器
pub struct MultiProcessExecutor {
    config: MultiProcessConfig,
    backup_manager: Option<BackupManager>,
}

impl MultiProcessExecutor {
    pub fn new(config: MultiProcessConfig) -> PyResult<Self> {
        // 流式处理模式下强制检查备份文件配置
        if config.require_backup && config.backup_file.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "流式处理模式下必须指定备份文件"
            ));
        }

        let backup_manager = if let Some(backup_file) = &config.backup_file {
            Some(BackupManager::new(backup_file, &config.storage_format)?)
        } else {
            None
        };

        Ok(Self {
            config,
            backup_manager,
        })
    }

    /// 提取函数代码
    fn extract_function_code(&self, py: Python, func: &PyAny) -> PyResult<String> {
        log_message("正在提取函数代码...".to_string());
        
        let inspect = py.import("inspect")?;
        
        match inspect.call_method1("getsource", (func,)) {
            Ok(source) => {
                let source_str: String = source.extract()?;
                log_message(format!("✅ 成功获取函数源代码，长度: {} 字符", source_str.len()));
                
                if source_str.trim().is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "函数源代码为空"
                    ));
                }
                
                // 使用textwrap.dedent去除缩进
                let textwrap = py.import("textwrap")?;
                let dedented_source = textwrap.call_method1("dedent", (source_str,))?;
                let final_source: String = dedented_source.extract()?;
                log_message(format!("✅ 去除缩进后源代码长度: {} 字符", final_source.len()));
                
                Ok(final_source)
            }
            Err(e) => {
                log_message(format!("⚠️ 无法获取函数源代码: {}", e));
                
                let func_name = func.getattr("__name__")
                    .and_then(|name| name.extract::<String>())
                    .unwrap_or_else(|_| "user_function".to_string());
                
                log_message(format!("📝 创建函数包装，函数名: {}", func_name));
                
                match py.import("dill") {
                    Ok(dill) => {
                        let serialized = dill.call_method1("dumps", (func,))?;
                        let bytes: Vec<u8> = serialized.extract()?;
                        use base64::Engine;
                        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                        log_message(format!("✅ 成功使用dill序列化，长度: {} 字符", encoded.len()));
                        Ok(encoded)
                    }
                    Err(_) => {
                        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            format!("无法序列化函数 '{}'，请确保函数是全局定义的或安装dill库", func_name)
                        ))
                    }
                }
            }
        }
    }

    /// 多进程执行
    pub fn run_multiprocess(
        &mut self,
        py: Python,
        func: &PyAny,
        args: Vec<(i32, String)>,
        _go_class: Option<&PyAny>,
        progress_callback: Option<&PyAny>,
    ) -> PyResult<Vec<ReturnResult>> {
        let total_tasks = args.len();
        log_message(format!("开始多进程执行，总任务数: {}", total_tasks));

        // 保存原始参数用于最后读取
        let original_args = args.clone();

        // 预估备份文件大小（基于820万行*266列=16.6GB的经验数据）
        if let Some(ref backup_file) = self.config.backup_file {
            let estimated_size_gb = Self::estimate_backup_size(total_tasks);
            // 立即输出备份信息，而不是缓存到日志中
            Python::with_gil(|py| {
                let _ = py.import("builtins").and_then(|builtins| {
                    builtins.call_method1("print", (format!("备份文件: {}", backup_file),))
                });
                let _ = py.import("builtins").and_then(|builtins| {
                    builtins.call_method1("print", (format!("预估备份文件大小: {:.2}GB（基于 {} 个任务）", estimated_size_gb, total_tasks),))
                });
            });
        }

        // 将参数转换为Task结构
        let tasks: Vec<Task> = args.into_iter().map(|(date, code)| Task { date, code }).collect();

        // 检查是否需要从备份恢复
        let (remaining_tasks, existing_results) = if self.config.resume_from_backup {
            self.load_existing_results(&tasks)?
        } else {
            (tasks, Vec::new())
        };

        let remaining_count = remaining_tasks.len();
        log_message(format!("需要计算的任务数: {}", remaining_count));

        if remaining_count == 0 {
            return Ok(existing_results.into_iter().map(|r| r.to_return_result()).collect());
        }

        // 备份线程设置
        let (backup_sender, backup_receiver) = if self.backup_manager.is_some() {
            let (tx, rx) = channel::<ProcessResult>();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let backup_handle = if let Some(receiver) = backup_receiver {
            if let Some(ref backup_file) = self.config.backup_file {
                // 重新创建备份管理器而不是移走原有的
                let backup_manager = BackupManager::new(backup_file, &self.config.storage_format)?;
                let batch_size = self.config.backup_batch_size;
                let fsync_frequency = self.config.fsync_frequency;
                Some(thread::spawn(move || {
                    Self::backup_worker(backup_manager, receiver, batch_size, fsync_frequency);
                }))
            } else { None }
        } else { None };

        // 创建进程池
        let num_processes = self.config.num_processes.unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        });
        let mut process_pool = ProcessPool::new(num_processes, &self.config.python_path)?;

        // 提取函数代码
        let function_code = self.extract_function_code(py, func)?;
        
        // 异步流水线执行（流式处理）
        match process_pool.execute_tasks_async(
            py,
            &function_code,
            remaining_tasks,
            backup_sender.clone(),
        ) {
            Ok(_) => {
                // 流式处理完成，结果已写入备份文件
            },
            Err(e) => {
                if let Some(cb) = progress_callback {
                    let _ = cb.call_method1("set_error", (e.to_string(),));
                }
                return Err(e);
            }
        };

        // 等待备份完成
        if let Some(sender) = backup_sender {
            drop(sender);
        }
        if let Some(handle) = backup_handle {
            let _ = handle.join();
        }

        // 流式处理完成

        log_message(format!("多进程执行完成，总任务数: {}", total_tasks));

        // 流式处理：从备份文件读取所有结果
        if let Some(backup_manager) = &self.backup_manager {
            let final_results = backup_manager.load_existing_results(&original_args)?;
            Ok(final_results.into_iter().map(|r| ReturnResult {
                date: r.date,
                code: r.code,
                facs: r.facs,
            }).collect())
        } else {
            // 如果没有备份文件，流式处理无法返回结果
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "流式处理模式下必须指定备份文件才能返回结果"
            ));
        }
    }

    /// 估算备份文件大小（基于820万行*266列=16.6GB的经验数据）
    fn estimate_backup_size(total_tasks: usize) -> f64 {
        // 经验数据：820万行 * 266列 = 16.6GB
        let known_rows = 8_200_000f64;
        let known_size_gb = 16.6f64;
        
        // 计算每行的平均大小（GB）
        let gb_per_row = known_size_gb / known_rows;
        
        // 估算总大小
        total_tasks as f64 * gb_per_row
    }

    /// 备份工作线程（支持可配置的fsync频率）
    fn backup_worker(
        mut backup_manager: BackupManager,
        receiver: Receiver<ProcessResult>,
        batch_size: usize,
        fsync_frequency: usize,
    ) {
        let mut batch = Vec::with_capacity(batch_size);
        let mut batch_count = 0;
        
        loop {
            match receiver.recv() {
                Ok(result) => {
                    // 转换为ComputeResult
                    let compute_result = crate::parallel::ComputeResult {
                        date: result.date,
                        code: result.code,
                        timestamp: result.timestamp,
                        facs: result.facs,
                    };
                    batch.push(compute_result);
                    
                    if batch.len() >= batch_size {
                        if let Err(e) = backup_manager.save_batch(&batch) {
                            eprintln!("备份失败: {}", e);
                        }
                        // 使用clear而不是重新分配，更高效且避免内存累积
                        batch.clear();
                        // 立即释放多余容量，防止内存累积
                        batch.shrink_to_fit();
                        batch_count += 1;
                        
                        // 根据配置决定是否强制fsync
                        if fsync_frequency > 0 && batch_count % fsync_frequency == 0 {
                            // 这里可以添加fsync逻辑，当前backup_manager已包含flush
                            // 未来可扩展为真正的fsync调用
                        }
                    }
                }
                Err(_) => {
                    // 通道关闭，保存剩余数据
                    if !batch.is_empty() {
                        if let Err(e) = backup_manager.save_batch(&batch) {
                            eprintln!("最终备份失败: {}", e);
                        }
                    }
                    break;
                }
            }
        }
    }

    /// 加载已有结果
    fn load_existing_results(
        &self,
        tasks: &[Task],
    ) -> PyResult<(Vec<Task>, Vec<ProcessResult>)> {
        if let Some(backup_manager) = &self.backup_manager {
            // 转换为(i32, String)格式以兼容现有的备份管理器
            let args: Vec<(i32, String)> = tasks.iter()
                .map(|task| (task.date, task.code.clone()))
                .collect();
                
            let existing = backup_manager.load_existing_results(&args)?;
            let existing_keys: std::collections::HashSet<(i32, String)> = 
                existing.iter().map(|r| (r.date, r.code.clone())).collect();
            
            let remaining: Vec<Task> = tasks
                .iter()
                .filter(|task| !existing_keys.contains(&(task.date, task.code.clone())))
                .cloned()
                .collect();

            // 输出备份信息
            if !existing.is_empty() && !remaining.is_empty() {
                let latest_backup_date = existing.iter().map(|r| r.date).max().unwrap_or(0);
                let earliest_remaining_date = remaining.iter().map(|t| t.date).min().unwrap_or(0);
                log_message(format!("备份中最晚日期为{}，即将从{}日期开始计算", latest_backup_date, earliest_remaining_date));
            }

            // 转换为ProcessResult
            let process_results: Vec<ProcessResult> = existing.into_iter()
                .map(|r| ProcessResult {
                    date: r.date,
                    code: r.code,
                    timestamp: r.timestamp,
                    facs: r.facs,
                })
                .collect();

            Ok((remaining, process_results))
        } else {
            Ok((tasks.to_vec(), Vec::new()))
        }
    }
}
