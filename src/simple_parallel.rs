///! 极简版并行计算模块
///!
///! 只负责并行执行Python函数，不收集结果，不备份数据
use chrono::Local;
use crossbeam::channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_SIMPLE_WORKER_TASK_TIMEOUT_SECS: u64 = 1800;

fn simple_worker_task_timeout() -> Duration {
    env::var("SIMPLE_WORKER_TASK_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(DEFAULT_SIMPLE_WORKER_TASK_TIMEOUT_SECS))
}

// ============================================================================
// 日志记录器
// ============================================================================

/// 轻量级日志记录器，用于捕获子进程的 stderr 输出
#[derive(Clone)]
struct SimpleLogger {
    file: Arc<Mutex<File>>,
}

impl SimpleLogger {
    fn new(log_path: &str) -> PyResult<Self> {
        let file = File::create(log_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "创建日志文件失败 {}: {}",
                log_path, e
            ))
        })?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }

    fn log_error(&self, worker_id: usize, category: &str, message: &str) {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
        let log_line = format!(
            "[{}] [Worker {}] [{}] {}\n",
            timestamp, worker_id, category, message
        );
        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(log_line.as_bytes());
        }
    }
}

// ============================================================================
// 数据结构定义
// ============================================================================

/// 任务参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParam {
    pub date: String,
    pub code: String,
}

/// 单个任务数据（用于发送给Python worker）
#[derive(Debug, Serialize, Deserialize)]
struct SingleTask {
    python_code: String,
    task: TaskParam,
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 检测Python解释器路径
fn detect_python_interpreter() -> String {
    if let Ok(python_path) = env::var("PYTHON_INTERPRETER") {
        if Path::new(&python_path).exists() {
            return python_path;
        }
    }

    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda_python = format!("{}/bin/python", conda_prefix);
        if Path::new(&conda_python).exists() {
            return conda_python;
        }
    }

    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        let venv_python = format!("{}/bin/python", virtual_env);
        if Path::new(&venv_python).exists() {
            return venv_python;
        }
    }

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

    "python".to_string()
}

/// 提取Python函数代码 - 使用pickle序列化代替inspect.getsource
/// 避免因模块热更新或源码解析问题导致提取到错误的函数
fn extract_python_function_code(py_func: &PyObject) -> PyResult<String> {
    Python::with_gil(|py| {
        let pickle = py.import("pickle")?;
        match pickle.call_method1("dumps", (py_func,)) {
            Ok(pickled) => {
                let base64 = py.import("base64")?;
                let encoded = base64.call_method1("b64encode", (pickled,))?;
                let encoded_str: String = encoded.call_method0("decode")?.extract()?;

                Ok(format!(
                    r#"
import pickle
import base64
_func_data = base64.b64decode('{}')
user_function = pickle.loads(_func_data)
"#,
                    encoded_str
                ))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cannot pickle the Python function: {}", e),
            )),
        }
    })
}

/// 创建极简worker脚本
fn create_simple_worker_script() -> String {
    r#"#!/usr/bin/env python3
import sys
import msgpack
import struct
import os
import textwrap
import traceback

def main():
    while True:
        try:
            # 读取任务长度
            length_bytes = sys.stdin.buffer.read(4)
            if len(length_bytes) != 4:
                break

            length = struct.unpack('<I', length_bytes)[0]
            if length == 0:
                break

            # 读取任务数据
            data = sys.stdin.buffer.read(length)
            if len(data) != length:
                break

            # 解析任务
            task_data = msgpack.unpackb(data, raw=False)
            func_code = task_data['python_code']
            task = task_data['task']
            date = task['date']
            code = task['code']
            func_code = textwrap.dedent(func_code)
            # 执行任务
            try:
                namespace = {'__builtins__': __builtins__}
                exec(func_code, namespace)

                # 查找用户函数：优先用 user_function（pickle路径），
                # 其次取第一个非内置可调用对象（getsource路径）
                if 'user_function' in namespace:
                    func = namespace['user_function']
                else:
                    user_functions = [name for name, obj in namespace.items()
                                     if callable(obj) and not name.startswith('_')]
                    func = namespace[user_functions[0]] if user_functions else None

                if func is not None:
                    func(date, code)  # 执行函数，不收集结果

                # 任务完成后，发送确认信号到 stdout
                sys.stdout.buffer.write(b'DONE\n')
                sys.stdout.buffer.flush()

            except Exception as e:
                error_msg = traceback.format_exc()
                print(f"❌ Worker任务失败: {date}, {code} -> {e}", file=sys.stderr, flush=True)
                print(error_msg, file=sys.stderr, flush=True)

                # 即使出错也发送确认信号，避免阻塞
                sys.stdout.buffer.write(b'DONE\n')
                sys.stdout.buffer.flush()

        except Exception:
            break

if __name__ == '__main__':
    main()
"#
    .to_string()
}

/// Worker函数：从队列中取任务并执行
fn run_simple_worker(
    worker_id: usize,
    task_queue: Receiver<TaskParam>,
    python_code: String,
    python_path: String,
    completion_sender: Sender<()>,
    logger: SimpleLogger,
) {
    let script_content = create_simple_worker_script();
    let script_path = format!("/tmp/simple_worker_{}.py", worker_id);

    if let Err(e) = std::fs::write(&script_path, script_content) {
        logger.log_error(worker_id, "SCRIPT_CREATE", &format!("创建脚本失败: {}", e));
        return;
    }

    let mut command = Command::new(&python_path);
    command
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped()) // 需要读取stdout来获取确认信号
        .stderr(Stdio::piped()) // 捕获stderr用于日志记录
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("NUMEXPR_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .env("BLIS_NUM_THREADS", "1")
        .env("POLARS_MAX_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1");

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            logger.log_error(worker_id, "PROCESS_START", &format!("启动Python进程失败: {}", e));
            let _ = std::fs::remove_file(&script_path);
            return;
        }
    };

    let mut stdin = match child.stdin.take() {
        Some(s) => s,
        None => {
            logger.log_error(worker_id, "STDIN", "获取 stdin 失败");
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&script_path);
            return;
        }
    };
    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            logger.log_error(worker_id, "STDOUT", "获取 stdout 失败");
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&script_path);
            return;
        }
    };
    let stderr = match child.stderr.take() {
        Some(s) => s,
        None => {
            logger.log_error(worker_id, "STDERR", "获取 stderr 失败");
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&script_path);
            return;
        }
    };
    let task_timeout = simple_worker_task_timeout();

    let (stdout_sender, stdout_receiver) = unbounded::<Result<String, ()>>();
    let stdout_handle = thread::spawn(move || {
        let mut stdout_reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match stdout_reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if stdout_sender.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Err(_) => {
                    let _ = stdout_sender.send(Err(()));
                    break;
                }
            }
        }
    });

    // 启动stderr读取线程，逐行写入日志文件
    let stderr_logger = logger.clone();
    let stderr_worker_id = worker_id;
    let stderr_handle = thread::spawn(move || {
        let stderr_reader = BufReader::new(stderr);
        for line in stderr_reader.lines() {
            match line {
                Ok(text) => {
                    stderr_logger.log_error(stderr_worker_id, "PYTHON_STDERR", &text);
                }
                Err(_) => break,
            }
        }
    });

    // 处理所有任务
    while let Ok(task) = task_queue.recv() {
        let single_task = SingleTask {
            python_code: python_code.clone(),
            task: task.clone(),
        };

        // 序列化任务
        let packed_data = match rmp_serde::to_vec_named(&single_task) {
            Ok(data) => data,
            Err(e) => {
                logger.log_error(
                    worker_id,
                    "SERIALIZE",
                    &format!("任务(date={}, code={})序列化失败: {}", task.date, task.code, e),
                );
                continue;
            }
        };

        let length = packed_data.len() as u32;
        let length_bytes = length.to_le_bytes();

        // 发送任务
        if stdin.write_all(&length_bytes).is_err() {
            logger.log_error(worker_id, "COMMUNICATION", "写入任务长度失败");
            break;
        }
        if stdin.write_all(&packed_data).is_err() {
            logger.log_error(worker_id, "COMMUNICATION", "写入任务数据失败");
            break;
        }
        if stdin.flush().is_err() {
            logger.log_error(worker_id, "COMMUNICATION", "flush stdin失败");
            break;
        }

        // 等待Python子进程完成任务并读取确认信号，避免无限阻塞在 read_line 上。
        let line = match stdout_receiver.recv_timeout(task_timeout) {
            Ok(Ok(line)) => line,
            Ok(Err(())) => {
                logger.log_error(worker_id, "COMMUNICATION", "读取确认信号失败");
                break;
            }
            Err(RecvTimeoutError::Timeout) => {
                logger.log_error(
                    worker_id,
                    "TIMEOUT",
                    &format!("等待确认信号超时: {} 秒", task_timeout.as_secs()),
                );
                let _ = child.kill();
                break;
            }
            Err(RecvTimeoutError::Disconnected) => {
                logger.log_error(worker_id, "COMMUNICATION", "stdout 通道提前关闭");
                break;
            }
        };

        // 只有收到确认信号后才通知主线程完成任务
        if line.trim() == "DONE" {
            let _ = completion_sender.send(());
        } else {
            logger.log_error(
                worker_id,
                "PROTOCOL",
                &format!("收到异常确认信号: {:?}", line),
            );
            continue;
        }
    }

    // 发送终止信号
    let _ = stdin.write_all(&[0u8; 4]);
    let _ = stdin.flush();

    // 等待进程退出，超时后强制kill防止僵尸进程
    match child.try_wait() {
        Ok(Some(_)) => {}
        _ => {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    break;
                }
                if child.try_wait().ok().flatten().is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    let _ = child.wait();

    // stdout/stderr pipe 关闭后，读取线程会自然退出。
    let _ = stdout_handle.join();

    // join stderr 线程确保资源释放
    let _ = stderr_handle.join();

    // 清理脚本
    let _ = std::fs::remove_file(&script_path);
}

// ============================================================================
// 主函数
// ============================================================================

/// 极简版并行计算函数 - 只执行不返回
#[pyfunction]
#[pyo3(signature = (python_function, args, n_jobs, log_path="run_pools_simple.log"))]
pub fn run_pools_simple(
    python_function: PyObject,
    args: &PyList,
    n_jobs: usize,
    log_path: &str,
) -> PyResult<()> {
    // 解析任务列表
    let mut all_tasks = Vec::new();
    for item in args.iter() {
        let task_args: &PyList = item.extract()?;
        if task_args.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each task should have exactly 2 parameters",
            ));
        }

        let date: String = task_args.get_item(0)?.str()?.extract()?;
        let code: String = task_args.get_item(1)?.str()?.extract()?;

        all_tasks.push(TaskParam { date, code });
    }

    let start_time = Instant::now();
    let total_tasks = all_tasks.len();

    println!(
        "[{}] 📋 总任务数: {}",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        total_tasks
    );

    // 提取Python函数代码
    let python_code = extract_python_function_code(&python_function)?;
    let python_path = detect_python_interpreter();

    // 创建日志记录器
    let logger = SimpleLogger::new(log_path)?;

    // 创建任务队列和完成通知channel
    let (task_sender, task_receiver) = unbounded::<TaskParam>();
    let (completion_sender, completion_receiver) = unbounded::<()>();

    // 将所有任务发送到队列
    for task in all_tasks {
        if let Err(e) = task_sender.send(task) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to send task: {}",
                e
            )));
        }
    }
    drop(task_sender);

    println!(
        "[{}] 🚀 启动 {} 个worker处理任务",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        n_jobs
    );

    // 启动workers
    let mut worker_handles = Vec::new();
    for i in 0..n_jobs {
        let worker_task_receiver = task_receiver.clone();
        let worker_python_code = python_code.clone();
        let worker_python_path = python_path.clone();
        let worker_completion_sender = completion_sender.clone();
        let worker_logger = logger.clone();

        let handle = thread::spawn(move || {
            run_simple_worker(
                i,
                worker_task_receiver,
                worker_python_code,
                worker_python_path,
                worker_completion_sender,
                worker_logger,
            );
        });

        worker_handles.push(handle);
    }

    drop(completion_sender);

    // 监控进度
    let mut completed = 0;
    while completion_receiver.recv().is_ok() {
        completed += 1;
        print!(
            "\r[{}] 📊 已完成 {}/{} 个任务",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            completed,
            total_tasks
        );
        io::stdout().flush().unwrap();
    }

    // 等待所有workers完成
    println!(
        "[{}] ⏳ 等待所有worker完成...",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    );

    for (i, handle) in worker_handles.into_iter().enumerate() {
        if let Err(e) = handle.join() {
            eprintln!("❌ Worker {} 异常: {:?}", i, e);
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "[{}] ✅ 任务完成！共处理 {} 个任务，耗时: {:?}",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        completed,
        elapsed
    );

    Ok(())
}
