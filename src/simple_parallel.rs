///! 极简版并行计算模块
///!
///! 只负责并行执行Python函数，不收集结果，不备份数据
use chrono::Local;
use crossbeam::channel::{unbounded, Receiver, Sender};
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Instant;

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

/// 提取Python函数代码
fn extract_python_function_code(py_func: &PyObject) -> PyResult<String> {
    Python::with_gil(|py| {
        let inspect = py.import("inspect")?;

        match inspect.call_method1("getsource", (py_func,)) {
            Ok(source) => {
                let source_str: String = source.extract()?;
                Ok(source_str)
            }
            Err(_) => {
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
                    Err(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Cannot serialize the Python function",
                    )),
                }
            }
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

                # 找到用户定义的函数
                user_functions = [name for name, obj in namespace.items()
                                 if callable(obj) and not name.startswith('_')]

                if user_functions:
                    func = namespace[user_functions[0]]
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
) {
    let script_content = create_simple_worker_script();
    let script_path = format!("/tmp/simple_worker_{}.py", worker_id);

    if let Err(e) = std::fs::write(&script_path, script_content) {
        eprintln!("❌ Worker {} 创建脚本失败: {}", worker_id, e);
        return;
    }

    let mut child = match Command::new(&python_path)
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped()) // 需要读取stdout来获取确认信号
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            eprintln!("❌ Worker {} 启动Python进程失败: {}", worker_id, e);
            let _ = std::fs::remove_file(&script_path);
            return;
        }
    };

    let mut stdin = child.stdin.take().expect("Failed to get stdin");
    let stdout = child.stdout.take().expect("Failed to get stdout");
    let mut reader = BufReader::new(stdout);

    // 处理所有任务
    while let Ok(task) = task_queue.recv() {
        let single_task = SingleTask {
            python_code: python_code.clone(),
            task,
        };

        // 序列化任务
        let packed_data = match rmp_serde::to_vec_named(&single_task) {
            Ok(data) => data,
            Err(_) => continue,
        };

        let length = packed_data.len() as u32;
        let length_bytes = length.to_le_bytes();

        // 发送任务
        if stdin.write_all(&length_bytes).is_err() {
            break;
        }
        if stdin.write_all(&packed_data).is_err() {
            break;
        }
        if stdin.flush().is_err() {
            break;
        }

        // 等待Python子进程完成任务并读取确认信号
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            break;
        }

        // 只有收到确认信号后才通知主线程完成任务
        if line.trim() == "DONE" {
            let _ = completion_sender.send(());
        } else {
            // 如果没有收到正确的确认信号，跳过这个任务
            continue;
        }
    }

    // 发送终止信号
    let _ = stdin.write_all(&[0u8; 4]);
    let _ = stdin.flush();

    // 等待进程结束
    let _ = child.wait();

    // 清理脚本
    let _ = std::fs::remove_file(&script_path);
}

// ============================================================================
// 主函数
// ============================================================================

/// 极简版并行计算函数 - 只执行不返回
#[pyfunction]
#[pyo3(signature = (python_function, args, n_jobs))]
pub fn run_pools_simple(python_function: PyObject, args: &PyList, n_jobs: usize) -> PyResult<()> {
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

        let handle = thread::spawn(move || {
            run_simple_worker(
                i,
                worker_task_receiver,
                worker_python_code,
                worker_python_path,
                worker_completion_sender,
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
