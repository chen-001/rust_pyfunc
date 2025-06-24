use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::process::{Command, Stdio, Child};
use std::io::{Write, BufRead, BufReader};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use crossbeam::channel::{unbounded, Receiver as CrossbeamReceiver, Sender as CrossbeamSender};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use serde::{Serialize, Deserialize};
use crate::backup::BackupManager;

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
    GoClass(String),
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
    pub results: Vec<Vec<f64>>,
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
}

/// 任务分发器状态
#[derive(Debug)]
struct TaskDispatcherState {
    total_tasks: usize,
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
        // let script_path = std::path::PathBuf::from("/home/chenzongwei/rust_pyfunc/python/worker_process.py");
        let mut script_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        script_path.push("python");
        script_path.push("worker_process.py");
        println!("script_path: {}", script_path.display());
        
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
        };

        // 给工作进程一些时间启动
        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(worker)
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
        // 设置读取超时
        // self.stdout_reader.get_mut().set_read_timeout(Some(std::time::Duration::from_secs(60)))
        //     .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("设置读取超时失败: {}", e)))?;

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
        
        println!("创建 {} 个工作进程...", num_processes);
        
        for i in 0..num_processes {
            let worker = WorkerProcess::new(i, python_path)?;
            workers.push(worker);
        }
        
        println!("进程池创建完成");
        
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
        go_class_serialized: Option<String>,
        progress_callback: Option<&PyAny>,
        backup_sender: Option<Sender<ProcessResult>>,
    ) -> PyResult<Vec<ProcessResult>> {
        let total_tasks = tasks.len();
        if total_tasks == 0 {
            return Ok(Vec::new());
        }

        println!("开始异步流水线执行，总任务数: {}", total_tasks);
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
            total_tasks,
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
            worker.send_command(&WorkerCommand::FunctionCode(function_code.to_string()))?;
            if let Some(go_ser) = &go_class_serialized {
                worker.send_command(&WorkerCommand::GoClass(go_ser.clone()))?;
            }
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

        // 3. 启动结果收集和进度更新线程
        let results = Arc::new(Mutex::new(Vec::new()));
        let results_clone = Arc::clone(&results);
        let state_clone = Arc::clone(&state);
        
        let progress_handle = if progress_callback.is_some() {
            let cb_ptr = progress_callback.unwrap() as *const PyAny;
            Some(thread::spawn(move || {
                Self::progress_update_loop(cb_ptr, state_clone, start_time);
            }))
        } else {
            None
        };

        let collection_handle = thread::spawn(move || {
            Self::result_collection_loop(result_receiver, results_clone, backup_sender, state, total_tasks)
        });

        // 4. 等待所有工作完成
        for handle in worker_handles {
            let _ = handle.join();
        }
        
        // 等待结果收集完成
        let _ = collection_handle.join();
        
        // 停止进度更新
        if let Some(handle) = progress_handle {
            let _ = handle.join();
        }

        // 重建进程池
        for i in 0..self.num_processes {
            let new_worker = WorkerProcess::new(i, &self.python_path)?;
            self.workers.push(new_worker);
        }

        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        println!("异步流水线执行完成，总耗时: {:.2}秒", start_time.elapsed().as_secs_f64());
        
        Ok(final_results)
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

                    // 发送任务给工作进程
                    if let Err(e) = worker.send_command(&WorkerCommand::Task(task.clone())) {
                        eprintln!("工作进程 {} 发送任务失败: {}", worker_id, e);
                        break;
                    }
                    
                    if let Err(e) = worker.send_command(&WorkerCommand::Execute {}) {
                        eprintln!("工作进程 {} 发送执行指令失败: {}", worker_id, e);
                        break;
                    }

                    // 接收结果
                    match worker.receive_result() {
                        Ok(response) => {
                            if !response.errors.is_empty() {
                                for error_msg in response.errors {
                                    eprintln!("工作进程 {} 返回错误: {}", worker_id, error_msg);
                                }
                            }

                            // 处理结果（应该只有一个结果，因为我们一次只发送一个任务）
                            if let Some(facs) = response.results.into_iter().next() {
                                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
                                let result = ProcessResult {
                                    date: task.date,
                                    code: task.code.clone(),
                                    timestamp: now,
                                    facs,
                                };
                                
                                if result_sender.send((worker_id, result)).is_err() {
                                    eprintln!("工作进程 {} 发送结果失败，结果收集器可能已关闭", worker_id);
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("工作进程 {} 接收结果失败: {}", worker_id, e);
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
        println!("工作进程 {} 已退出", worker_id);
    }

    /// 结果收集循环
    fn result_collection_loop(
        result_receiver: Receiver<(usize, ProcessResult)>,
        results: Arc<Mutex<Vec<ProcessResult>>>,
        backup_sender: Option<Sender<ProcessResult>>,
        state: Arc<Mutex<TaskDispatcherState>>,
        total_tasks: usize,
    ) {
        while let Ok((_worker_id, result)) = result_receiver.recv() {
            // 发送到备份线程
            if let Some(ref sender) = backup_sender {
                let _ = sender.send(result.clone());
            }

            // 存储结果
            {
                let mut results = results.lock().unwrap();
                results.push(result);
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

    /// 进度更新循环
    fn progress_update_loop(
        callback_ptr: *const PyAny,
        state: Arc<Mutex<TaskDispatcherState>>,
        start_time: Instant,
    ) {
        let mut last_update_time = Instant::now();
        
        loop {
            thread::sleep(std::time::Duration::from_millis(500)); // 每500ms更新一次
            
            let (completed, total, finished) = {
                let state = state.lock().unwrap();
                (state.completed_tasks, state.total_tasks, state.completed_tasks >= state.total_tasks)
            };

            if finished {
                break;
            }

            // 限制更新频率
            if last_update_time.elapsed().as_millis() < 500 {
                continue;
            }
            last_update_time = Instant::now();

            let elapsed = start_time.elapsed().as_secs_f64();
            let estimated_remaining = if completed > 0 {
                (elapsed / completed as f64) * (total - completed) as f64
            } else {
                0.0
            };

            // 安全地调用Python回调
            Python::with_gil(|py| {
                unsafe {
                    let callback = &*callback_ptr;
                    let progress_info: Vec<PyObject> = vec![
                        completed.to_object(py),
                        total.to_object(py),
                        estimated_remaining.to_object(py),
                        elapsed.to_object(py),
                    ];
                    
                    if let Err(e) = callback.call_method1("update", (progress_info,)) {
                        eprintln!("调用进度回调失败: {}", e);
                    }
                }
            });
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
}

impl Default for MultiProcessConfig {
    fn default() -> Self {
        Self {
            num_processes: None,
            backup_batch_size: 1000,
            backup_file: None,
            storage_format: "binary".to_string(),
            resume_from_backup: false,
            python_path: "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python".to_string(),
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
        println!("正在提取函数代码...");
        
        let inspect = py.import("inspect")?;
        
        match inspect.call_method1("getsource", (func,)) {
            Ok(source) => {
                let source_str: String = source.extract()?;
                println!("✅ 成功获取函数源代码，长度: {} 字符", source_str.len());
                
                if source_str.trim().is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "函数源代码为空"
                    ));
                }
                
                Ok(source_str)
            }
            Err(e) => {
                println!("⚠️ 无法获取函数源代码: {}", e);
                
                let func_name = func.getattr("__name__")
                    .and_then(|name| name.extract::<String>())
                    .unwrap_or_else(|_| "user_function".to_string());
                
                println!("📝 创建函数包装，函数名: {}", func_name);
                
                match py.import("dill") {
                    Ok(dill) => {
                        let serialized = dill.call_method1("dumps", (func,))?;
                        let bytes: Vec<u8> = serialized.extract()?;
                        use base64::Engine;
                        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                        println!("✅ 成功使用dill序列化，长度: {} 字符", encoded.len());
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

    /// 主要的多进程执行函数
    pub fn run_multiprocess(
        &mut self,
        py: Python,
        func: &PyAny,
        args: Vec<(i32, String)>,
        go_class: Option<&PyAny>,
        progress_callback: Option<&PyAny>,
        _chunk_size: Option<usize>, // 异步模式下不再使用chunk_size
    ) -> PyResult<Vec<ReturnResult>> {
        let total_tasks = args.len();
        println!("开始多进程执行，总任务数: {}", total_tasks);

        // 将参数转换为Task结构
        let tasks: Vec<Task> = args.into_iter().map(|(date, code)| Task { date, code }).collect();

        // 检查是否需要从备份恢复
        let (remaining_tasks, existing_results) = if self.config.resume_from_backup {
            self.load_existing_results(&tasks)?
        } else {
            (tasks, Vec::new())
        };

        let remaining_count = remaining_tasks.len();
        println!("需要计算的任务数: {}", remaining_count);

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
            if let Some(backup_manager) = self.backup_manager.take() {
                let batch_size = self.config.backup_batch_size;
                Some(thread::spawn(move || {
                    Self::backup_worker(backup_manager, receiver, batch_size);
                }))
            } else { None }
        } else { None };

        // 创建进程池
        let num_processes = self.config.num_processes.unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        });
        let mut process_pool = ProcessPool::new(num_processes, &self.config.python_path)?;

        // 提取和序列化代码/对象
        let function_code = self.extract_function_code(py, func)?;
        let go_class_serialized = if let Some(go_class) = go_class {
            py.import("dill").ok().and_then(|dill| {
                dill.call_method1("dumps", (go_class,)).ok().and_then(|s| {
                    s.extract::<Vec<u8>>().ok().map(|bytes| {
                        use base64::Engine;
                        base64::engine::general_purpose::STANDARD.encode(bytes)
                    })
                })
            })
        } else { None };
        
        // 异步流水线执行
        let async_results = match process_pool.execute_tasks_async(
            py,
            &function_code,
            remaining_tasks,
            go_class_serialized,
            progress_callback,
            backup_sender.clone(),
        ) {
            Ok(res) => res,
            Err(e) => {
                if let Some(cb) = progress_callback {
                    Python::with_gil(|_py| {
                        let _ = cb.call_method1("set_error", (e.to_string(),));
                    });
                }
                return Err(e);
            }
        };

        // 合并结果
        let mut all_results = existing_results;
        all_results.extend(async_results);

        // 标记任务完成
        if let Some(cb) = progress_callback {
            Python::with_gil(|_py| {
                let _ = cb.call_method0("finish");
            });
        }

        // 等待备份完成
        if let Some(sender) = backup_sender {
            drop(sender);
        }
        if let Some(handle) = backup_handle {
            let _ = handle.join();
        }

        println!("多进程执行完成，总任务数: {}", total_tasks);

        Ok(all_results.into_iter().map(|r| r.to_return_result()).collect())
    }

    /// 备份工作线程
    fn backup_worker(
        mut backup_manager: BackupManager,
        receiver: Receiver<ProcessResult>,
        batch_size: usize,
    ) {
        let mut batch = Vec::with_capacity(batch_size);
        
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
                        batch.clear();
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
                println!("备份中最晚日期为{}，即将从{}日期开始计算", latest_backup_date, earliest_remaining_date);
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
}