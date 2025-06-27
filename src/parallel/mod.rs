use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::backup::BackupManager;
use numpy::{PyArray2, ToPyArray};
use ndarray::Array2;
use crate::multiprocess::{MultiProcessExecutor, MultiProcessConfig};

/// 通过Python print函数打印信息，确保在Jupyter中可见
fn py_print(py: Python, message: &str) {
    if let Ok(builtins) = py.import("builtins") {
        let _ = builtins.call_method1("print", (message,));
    }
}

/// 计算结果
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ComputeResult {
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

impl ComputeResult {
    pub fn to_return_result(&self) -> ReturnResult {
        ReturnResult {
            date: self.date,
            code: self.code.clone(),
            facs: self.facs.clone(),
        }
    }
}

/// Python接口函数
#[pyfunction]
#[pyo3(signature = (
    func,
    args,
    go_class=None,
    num_threads=None,
    backup_file=None,
    backup_batch_size=1000,
    storage_format="binary",
    resume_from_backup=false,
    progress_callback=None,
))]
pub fn run_pools<'py>(
    py: Python<'py>,
    func: &PyAny,
    args: &PyList,
    go_class: Option<&PyAny>,
    num_threads: Option<usize>,
    backup_file: Option<String>,
    backup_batch_size: usize,
    storage_format: &str,
    resume_from_backup: bool,
    progress_callback: Option<&PyAny>,
) -> PyResult<&'py PyArray2<PyObject>> {
    
    py_print(py, "🔄 [run_pools] 函数开始执行");
    py_print(py, &format!("🔄 [run_pools] 参数信息 - args数量: {}, num_threads: {:?}, backup_file: {:?}", args.len(), num_threads, backup_file));
    py_print(py, &format!("🔄 [run_pools] 参数信息 - backup_batch_size: {}, storage_format: {}, resume_from_backup: {}", backup_batch_size, storage_format, resume_from_backup));
    
    // --- 多进程模式 ---
    py_print(py, "🔄 [run_pools] 调度到Rust原生多进程执行...");
    
    py_print(py, "🔄 [run_pools] 开始解析参数...");
    let parsed_args: Vec<(i32, String)> = args
        .iter()
        .map(|item| {
            let list: &PyList = item.downcast()?;
            let date: i32 = list.get_item(0)?.extract()?;
            let code: String = list.get_item(1)?.extract()?;
            Ok((date, code))
        })
        .collect::<PyResult<Vec<_>>>()?;
    py_print(py, &format!("✅ [run_pools] 参数解析完成，共 {} 个任务", parsed_args.len()));

    py_print(py, "🔄 [run_pools] 创建多进程配置...");
    let multiprocess_config = MultiProcessConfig {
        num_processes: num_threads,
        backup_batch_size,
        backup_file: backup_file.clone(),
        storage_format: storage_format.to_string(),
        resume_from_backup,
        ..Default::default()
    };
    py_print(py, "✅ [run_pools] 多进程配置创建完成");
    
    // 执行多进程任务
    py_print(py, "🔄 [run_pools] 创建多进程执行器...");
    let mut multiprocess_executor = MultiProcessExecutor::new(multiprocess_config)?;
    py_print(py, "✅ [run_pools] 多进程执行器创建完成");
    
    py_print(py, "🔄 [run_pools] 开始执行多进程任务...");
    let multiprocess_results = multiprocess_executor.run_multiprocess(py, func, parsed_args.clone(), go_class, progress_callback)?; // chunk_size在异步模式下不使用
    py_print(py, &format!("✅ [run_pools] 多进程任务执行完成，返回结果数量: {}", multiprocess_results.len()));
    
    // 输出收集的日志到Python
    crate::multiprocess::flush_logs_to_python(py);

    // 🔍 步骤1: 检查是否从备份恢复且所有任务已完成（此时应使用完整的备份格式）
    py_print(py, &format!("🔍 步骤1: 检查备份恢复条件 - resume_from_backup={}, multiprocess_results数量={}", resume_from_backup, multiprocess_results.len()));
    
    if resume_from_backup && !multiprocess_results.is_empty() {
        py_print(py, "✅ 满足备份恢复条件：resume_from_backup=true 且 multiprocess_results非空");
        
        // 🔍 步骤2: 检查备份文件中是否已包含所有请求的任务
        if let Some(ref backup_file_path) = backup_file {
            py_print(py, &format!("📁 步骤2: 检查备份文件 - 文件路径: {}", backup_file_path));
            py_print(py, &format!("📊 步骤2: 请求的任务总数: {}", parsed_args.len()));
            
            // 显示前几个请求的任务
            for (i, task) in parsed_args.iter().take(5).enumerate() {
                py_print(py, &format!("📋 请求任务{}：date={}, code={}", i+1, task.0, task.1));
            }
            if parsed_args.len() > 5 {
                py_print(py, &format!("📋 ... 还有 {} 个任务未显示", parsed_args.len() - 5));
            }
            
            py_print(py, "🔄 步骤3: 创建备份管理器...");
            let backup_manager = BackupManager::new(backup_file_path, &storage_format)?;
            py_print(py, "✅ 备份管理器创建成功");
            
            py_print(py, "🔄 步骤4: 查询备份数据...");
            let backup_compute_results = backup_manager.query_results(None, None)?;
            py_print(py, &format!("📋 步骤4完成: 从备份文件读取到 {} 条ComputeResult记录", backup_compute_results.len()));
            
            // 显示备份数据的前几条记录
            for (i, result) in backup_compute_results.iter().take(3).enumerate() {
                py_print(py, &format!("📋 备份记录{}：date={}, code={}, timestamp={}, 因子数={}", 
                    i+1, result.date, result.code, result.timestamp, result.facs.len()));
            }
            if backup_compute_results.len() > 3 {
                py_print(py, &format!("📋 ... 备份中还有 {} 条记录未显示", backup_compute_results.len() - 3));
            }
            
            // 🔍 步骤5: 比较备份数据数量与请求任务数量
            py_print(py, &format!("🔍 步骤5: 比较数量 - 备份记录={}, 请求任务={}", backup_compute_results.len(), parsed_args.len()));
            
            if backup_compute_results.len() >= parsed_args.len() {
                py_print(py, &format!("✅ 步骤5: 检测到完整备份恢复 - 备份中有{}条记录 >= 请求{}个任务", backup_compute_results.len(), parsed_args.len()));
                py_print(py, "🔄 步骤6: 开始使用备份文件的完整格式返回数据（包含timestamp）...");
                
                // 🔍 步骤7: 检查重复性和数据质量
                py_print(py, "🔄 步骤7: 检查数据质量和任务覆盖率...");
                let mut date_code_set = std::collections::HashSet::new();
                let mut duplicate_count = 0;
                let mut task_coverage = std::collections::HashSet::new();
                
                for compute_result in &backup_compute_results {
                    let key = (compute_result.date, compute_result.code.clone());
                    if !date_code_set.insert(key.clone()) {
                        duplicate_count += 1;
                    }
                    task_coverage.insert(key);
                }
                
                py_print(py, &format!("📊 步骤7: 备份数据质量 - 总记录={}, 唯一记录={}, 重复记录={}", 
                    backup_compute_results.len(), date_code_set.len(), duplicate_count));
                
                // 检查任务覆盖率
                let mut requested_tasks = std::collections::HashSet::new();
                for task in &parsed_args {
                    requested_tasks.insert((task.0, task.1.clone()));
                }
                
                let covered_tasks = requested_tasks.intersection(&task_coverage).count();
                let coverage_rate = (covered_tasks as f64 / requested_tasks.len() as f64) * 100.0;
                py_print(py, &format!("📈 步骤7: 任务覆盖率 - {}/{} = {:.1}%", covered_tasks, requested_tasks.len(), coverage_rate));
                
                if coverage_rate < 100.0 {
                    let missing_tasks: Vec<_> = requested_tasks.difference(&task_coverage).collect();
                    py_print(py, &format!("⚠️ 步骤7: 发现 {} 个缺失任务", missing_tasks.len()));
                    for (i, missing) in missing_tasks.iter().take(3).enumerate() {
                        py_print(py, &format!("❌ 缺失任务{}: date={}, code={}", i+1, missing.0, missing.1));
                    }
                    if missing_tasks.len() > 3 {
                        py_print(py, &format!("❌ ... 还有 {} 个缺失任务未显示", missing_tasks.len() - 3));
                    }
                }
                
                // 🔄 步骤8: 进行去重处理
                py_print(py, "🔄 步骤8: 开始去重处理...");
                let mut dedup_map: std::collections::HashMap<(i32, String), ComputeResult> = std::collections::HashMap::new();
                
                for compute_result in backup_compute_results {
                    let key = (compute_result.date, compute_result.code.clone());
                    dedup_map.insert(key, compute_result);
                }
                
                let final_backup_results: Vec<ComputeResult> = dedup_map.into_values().collect();
                py_print(py, &format!("✅ 步骤8: 去重完成 - {} -> {} 条记录", date_code_set.len(), final_backup_results.len()));
                
                // 🔍 步骤9: 计算数组形状
                py_print(py, "🔄 步骤9: 计算数组形状...");
                let num_rows = final_backup_results.len();
                let num_cols = if num_rows > 0 {
                    let facs_len = final_backup_results[0].facs.len();
                    py_print(py, &format!("📊 步骤9: 第一条记录的因子数量: {}", facs_len));
                    3 + facs_len // date, code, timestamp + facs
                } else {
                    py_print(py, "⚠️ 步骤9: 没有数据，使用最小列数");
                    3 // 最小列数
                };
                
                py_print(py, &format!("📐 步骤9: 计算的数组形状 - {} 行 × {} 列", num_rows, num_cols));
                
                // 🔍 步骤10: 内存限制检查
                py_print(py, "🔄 步骤10: 检查内存使用...");
                let expected_total_elements = num_rows * num_cols;
                let estimated_memory_gb = (expected_total_elements * 28) as f64 / (1024.0 * 1024.0 * 1024.0);
                py_print(py, &format!("💾 步骤10: 预期内存 - {:.2}GB ({}行 × {}列 × 28字节)", estimated_memory_gb, num_rows, num_cols));
                
                if estimated_memory_gb > 64.0 {
                    py_print(py, &format!("❌ 步骤10: 内存超限 - 需要{:.1}GB > 64GB限制", estimated_memory_gb));
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!(
                            "备份数据集过大: 预计需要 {:.1}GB 内存，超过限制 64.0GB。\n\
                            当前数据: {}行 × {}列\n\
                            建议使用query_backup的分块读取功能",
                            estimated_memory_gb, num_rows, num_cols
                        )
                    ));
                }
                py_print(py, "✅ 步骤10: 内存检查通过");
                
                // 🔄 步骤11: 分配内存并填充数据
                py_print(py, "🔄 步骤11: 分配内存并开始填充数据...");
                let mut data = Vec::with_capacity(expected_total_elements);
                
                for (i, result) in final_backup_results.iter().enumerate() {
                    data.push(result.date.to_object(py));
                    data.push(result.code.to_object(py));
                    data.push(result.timestamp.to_object(py)); // 包含timestamp
                    for fac in &result.facs {
                        data.push(fac.to_object(py));
                    }
                    
                    if (i + 1) % 50000 == 0 {
                        py_print(py, &format!("📈 步骤11: 已处理 {}/{} 行 ({:.1}%)", 
                            i + 1, num_rows, (i + 1) as f64 / num_rows as f64 * 100.0));
                    }
                }
                
                py_print(py, &format!("✅ 步骤11: 数据填充完成 - 最终数据长度: {} (期望: {})", data.len(), expected_total_elements));
                
                // 🔍 步骤12: 验证数据长度
                py_print(py, "🔄 步骤12: 验证数据长度...");
                if data.len() != expected_total_elements {
                    py_print(py, &format!("❌ 步骤12: 数据长度不匹配 - 期望: {}, 实际: {}", expected_total_elements, data.len()));
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("❌ 备份数据长度不匹配: 期望 {}, 实际 {}", expected_total_elements, data.len())
                    ));
                }
                py_print(py, "✅ 步骤12: 数据长度验证通过");
                
                // 🔄 步骤13: 创建NDArray
                py_print(py, "🔄 步骤13: 创建NDArray...");
                let array = Array2::from_shape_vec((num_rows, num_cols), data)
                    .map_err(|e| {
                        py_print(py, &format!("❌ 步骤13: NDArray创建失败 - {}", e));
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("❌ 创建备份NDArray失败: {}", e)
                        )
                    })?;
                
                py_print(py, &format!("🎉 步骤13完成: 成功从备份恢复 {} 条记录（完整格式，包含timestamp列）", num_rows));
                py_print(py, &format!("🎉 最终结果: 返回{}行×{}列的数组，第3列为timestamp", num_rows, num_cols));
                return Ok(array.to_pyarray(py));
            } else {
                py_print(py, &format!("⚠️ 步骤5: 备份数据不完整 - 备份{}条 < 请求{}个任务，将继续正常流程", 
                    backup_compute_results.len(), parsed_args.len()));
            }
        } else {
            py_print(py, "⚠️ 步骤2: 备份文件路径为空，继续正常流程");
        }
    } else {
        py_print(py, &format!("ℹ️ 不满足备份恢复条件 - resume_from_backup={}, multiprocess_results.is_empty()={}", 
            resume_from_backup, multiprocess_results.is_empty()));
    }

    // 转换为PyArray
    py_print(py, "🔄 [run_pools] 开始转换为PyArray");
    py_print(py, &format!("🔄 [run_pools] multiprocess_results数量: {}", multiprocess_results.len()));
    
    if multiprocess_results.is_empty() {
        py_print(py, "🔄 [run_pools] multiprocess_results为空，进入流式处理模式");
        // 流式处理模式：结果为空说明数据在备份文件中，尝试从备份文件读取
        py_print(py, "🔄 [run_pools] 进入流式处理模式：multiprocess_results为空，从备份文件读取...");
        if let Some(ref backup_file_path) = backup_file {
            py_print(py, &format!("🔄 [run_pools] 备份文件路径: {}", backup_file_path));
            py_print(py, "🔄 [run_pools] 流式处理模式：创建备份管理器...");
            let backup_manager = BackupManager::new(backup_file_path, &storage_format)?;
            py_print(py, "✅ [run_pools] 备份管理器创建成功");
            
            py_print(py, "🔄 [run_pools] 从备份文件读取结果...");
            let backup_compute_results = backup_manager.query_results(None, None)?;
            py_print(py, &format!("✅ [run_pools] 从备份文件读取完成，结果数量: {}", backup_compute_results.len()));
            
            py_print(py, &format!("从备份管理器读取到 {} 条ComputeResult记录", backup_compute_results.len()));
            
            if backup_compute_results.is_empty() {
                py_print(py, "警告：备份文件为空或无法解析");
                let empty_array = Array2::<PyObject>::from_shape_vec((0, 0), vec![])
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("无法创建空NDArray: {}", e)
                    ))?;
                return Ok(empty_array.to_pyarray(py));
            }
            
            // 将ComputeResult转换为ReturnResult，去除timestamp字段，并进行去重
            let mut dedup_map: std::collections::HashMap<(i32, String), ReturnResult> = std::collections::HashMap::new();
            
            for compute_result in backup_compute_results {
                let return_result = compute_result.to_return_result();
                let key = (return_result.date, return_result.code.clone());
                // 插入新记录，如果key已存在会覆盖旧记录，实现保留后出现的记录
                dedup_map.insert(key, return_result);
            }
            
            let backup_results: Vec<ReturnResult> = dedup_map.into_values().collect();
            
            py_print(py, &format!("备份数据转换完成，记录数: {}", backup_results.len()));
            
            // 安全的数组形状计算
            let num_rows = backup_results.len();
            let num_cols = if num_rows > 0 {
                let facs_len = backup_results[0].facs.len();
                py_print(py, &format!("第一条记录的因子数量: {}", facs_len));
                2 + facs_len // date, code + facs (不包含timestamp)
            } else {
                2 // 最小列数
            };
            
            py_print(py, &format!("计算的数组形状: {} 行 × {} 列", num_rows, num_cols));
            
            // 智能内存限制检查（与query_backup保持一致）
            let expected_total_elements = num_rows * num_cols;
            let estimated_memory_gb = (expected_total_elements * 28) as f64 / (1024.0 * 1024.0 * 1024.0);
            
            py_print(py, &format!("预期总元素数: {}, 预计内存: {:.2}GB", expected_total_elements, estimated_memory_gb));
            
            if estimated_memory_gb > 64.0 { // 64GB限制，与query_backup一致
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!(
                        "数据集过大: 预计需要 {:.1}GB 内存，超过限制 64.0GB。\n\
                        当前数据: {}行 × {}列\n\
                        建议使用query_backup的分块读取功能",
                        estimated_memory_gb, num_rows, num_cols
                    )
                ));
            }
            
            let mut data = Vec::with_capacity(expected_total_elements);
            
            py_print(py, "开始填充数据...");
            
            for (i, result) in backup_results.iter().enumerate() {
                data.push(result.date.to_object(py));
                data.push(result.code.to_object(py));
                // timestamp已在to_return_result()转换中被移除
                for fac in &result.facs {
                    // 保持NaN值，让Python接收到实际的NaN
                    data.push(fac.to_object(py));
                }
                
                // 每处理10万行报告一次进度
                if (i + 1) % 100000 == 0 {
                    py_print(py, &format!("已处理 {} 行，当前数据长度: {}", i + 1, data.len()));
                }
            }
            
            py_print(py, &format!("数据填充完成，最终数据长度: {}", data.len()));
            
            // 验证数据长度
            if data.len() != expected_total_elements {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("数据长度不匹配: 期望 {}, 实际 {}", expected_total_elements, data.len())
                ));
            }
            
            let array = Array2::from_shape_vec((num_rows, num_cols), data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("创建NDArray失败: {}", e)
                ))?;
            
            py_print(py, &format!("✅ [run_pools] 成功从备份读取 {} 条记录", num_rows));
            py_print(py, "🔄 [run_pools] 流式处理模式：返回备份文件数据");
            return Ok(array.to_pyarray(py));
        } else {
            // 没有备份文件，返回空数组
            py_print(py, "⚠️ [run_pools] 没有备份文件，返回空数组");
            let empty_array = Array2::<PyObject>::from_shape_vec((0, 0), vec![])
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("无法创建空NDArray: {}", e)
                ))?;
            py_print(py, "✅ [run_pools] 空数组创建完成，函数返回");
            return Ok(empty_array.to_pyarray(py));
        }
    }
    
    py_print(py, "🔄 [run_pools] 处理非空的multiprocess_results...");
    
    // 对multiprocess_results进行去重处理
    py_print(py, &format!("去重前multiprocess_results数量: {}", multiprocess_results.len()));
    let mut dedup_map: std::collections::HashMap<(i32, String), crate::multiprocess::ReturnResult> = std::collections::HashMap::new();
    
    for result in multiprocess_results {
        let key = (result.date, result.code.clone());
        // 插入新记录，如果key已存在会覆盖旧记录，实现保留后出现的记录
        dedup_map.insert(key, result);
    }
    
    let multiprocess_results: Vec<crate::multiprocess::ReturnResult> = dedup_map.into_values().collect();
    py_print(py, &format!("去重后multiprocess_results数量: {}", multiprocess_results.len()));
    
    let num_rows = multiprocess_results.len();
    
    // 计算最大因子数量，处理不一致的数据
    let max_facs_len = multiprocess_results.iter().map(|r| r.facs.len()).max().unwrap_or(0);
    let num_cols = 2 + max_facs_len; // date, code + facs
    
    py_print(py, &format!("multiprocess_results数组形状: {} 行 × {} 列 (最大因子数: {})", num_rows, num_cols, max_facs_len));
    
    let mut data = Vec::with_capacity(num_rows * num_cols);
    
    for (i, result) in multiprocess_results.iter().enumerate() {
        data.push(result.date.to_object(py));
        data.push(result.code.to_object(py));
        
        // 处理不一致的因子数量
        for j in 0..max_facs_len {
            if j < result.facs.len() {
                data.push(result.facs[j].to_object(py));
            } else {
                // 用NaN填充缺失的因子
                data.push(f64::NAN.to_object(py));
            }
        }
        
        // 每处理10万行检查一次数据长度
        if (i + 1) % 100000 == 0 {
            let expected_len = (i + 1) * num_cols;
            if data.len() != expected_len {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("multiprocess_results数据长度不匹配在第{}行: 期望 {}, 实际 {}", i + 1, expected_len, data.len())
                ));
            }
        }
    }
    
    py_print(py, &format!("multiprocess_results数据填充完成，数据长度: {}", data.len()));
    
    py_print(py, "🔄 [run_pools] 创建NDArray...");
    let array = Array2::from_shape_vec((num_rows, num_cols), data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("创建NDArray失败: {}", e)
        ))?;
    
    py_print(py, "✅ [run_pools] multiprocess_results NDArray创建成功");
    py_print(py, "✅ [run_pools] 函数执行完成，正常返回");
    Ok(array.to_pyarray(py))
}

/// 查询备份数据（增强版：支持大文件和分块读取）
#[pyfunction]
#[pyo3(signature = (
    backup_file,
    date_range=None,
    codes=None,
    storage_format="binary",
    row_range=None,
    max_rows=None,
    memory_limit_gb=None
))]
pub fn query_backup<'py>(
    py: Python<'py>,
    backup_file: &str,
    date_range: Option<(i32, i32)>,
    codes: Option<Vec<String>>,
    storage_format: &str,
    row_range: Option<(usize, usize)>,
    max_rows: Option<usize>,
    memory_limit_gb: Option<f64>,
) -> PyResult<&'py PyArray2<PyObject>> {
    let backup_manager = BackupManager::new(backup_file, storage_format)?;
    
    // 使用增强的查询方法支持分块读取
    let raw_results = backup_manager.query_results_chunked(date_range, codes, row_range, max_rows)?;

    // 对查询结果进行去重处理
    let mut dedup_map: std::collections::HashMap<(i32, String), ComputeResult> = std::collections::HashMap::new();
    
    for result in raw_results {
        let key = (result.date, result.code.clone());
        // 插入新记录，如果key已存在会覆盖旧记录，实现保留后出现的记录
        dedup_map.insert(key, result);
    }
    
    let results: Vec<ComputeResult> = dedup_map.into_values().collect();

    // 转换为NDArray
    if results.is_empty() {
        let empty_array = Array2::<PyObject>::from_shape_vec((0, 0), vec![])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("无法创建空NDArray: {}", e)
            ))?;
        return Ok(empty_array.to_pyarray(py));
    }
    
    let num_rows = results.len();
    
    // 安全地确定最大因子数量，处理不一致的数据
    let max_facs_len = results.iter().map(|r| r.facs.len()).max().unwrap_or(0);
    let num_cols = 3 + max_facs_len; // date, code, timestamp + facs
    
    // 智能内存限制检查
    let expected_total_elements = num_rows * num_cols;
    let estimated_memory_gb = (expected_total_elements * 28) as f64 / (1024.0 * 1024.0 * 1024.0);
    let memory_limit = memory_limit_gb.unwrap_or(64.0); // 默认64GB限制

    if estimated_memory_gb > memory_limit {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!(
                "数据集过大: 预计需要 {:.1}GB 内存，超过限制 {:.1}GB。\n\
                当前数据: {}行 × {}列\n\
                建议使用分块读取：\n\
                - 使用 row_range=(start, end) 读取部分行，如 row_range=(0, 100000)\n\
                - 使用 max_rows=N 限制读取行数，如 max_rows=50000\n\
                - 或增加 memory_limit_gb 参数，如 memory_limit_gb=16.0",
                estimated_memory_gb, memory_limit, num_rows, num_cols
            )
        ));
    }
    
    let mut data = Vec::with_capacity(expected_total_elements);
    
    // 检查数据一致性并填充
    for (i, result) in results.iter().enumerate() {
        data.push(result.date.to_object(py));
        data.push(result.code.to_object(py));
        data.push(result.timestamp.to_object(py));
        
        // 处理不一致的因子数量
        for j in 0..max_facs_len {
            if j < result.facs.len() {
                // 保持NaN值，让Python接收到实际的NaN
                data.push(result.facs[j].to_object(py));
            } else {
                // 用NaN填充缺失的因子
                data.push(f64::NAN.to_object(py));
            }
        }
        
        // 每处理100行检查一次数据长度
        if (i + 1) % 100 == 0 {
            let expected_len = (i + 1) * num_cols;
            if data.len() != expected_len {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("数据长度不匹配在第{}行: 期望 {}, 实际 {}", i + 1, expected_len, data.len())
                ));
            }
        }
    }
    
    // 最终验证数据长度
    if data.len() != expected_total_elements {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("最终数据长度不匹配: 期望 {}, 实际 {}", expected_total_elements, data.len())
        ));
    }
    
    let array = Array2::from_shape_vec((num_rows, num_cols), data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("创建NDArray失败: {}", e)
        ))?;
    
    Ok(array.to_pyarray(py))
}

/// 删除备份文件
#[pyfunction]
pub fn delete_backup(backup_file: &str, storage_format: &str) -> PyResult<()> {
    let backup_manager = BackupManager::new(backup_file, storage_format)?;
    backup_manager.delete_backup()
}
/// 检查备份文件是否存在
#[pyfunction]
pub fn backup_exists(backup_file: &str, storage_format: &str) -> PyResult<bool> {
    let backup_manager = BackupManager::new(backup_file, storage_format)?;
    Ok(backup_manager.backup_exists())
}

/// 获取备份文件信息
#[pyfunction]
pub fn get_backup_info(backup_file: &str, storage_format: &str) -> PyResult<(u64, String)> {
    let backup_manager = BackupManager::new(backup_file, storage_format)?;
    backup_manager.get_backup_info()
}

/// 获取备份文件详细信息（增强版）
/// 返回：(行数, 列数, 估算内存GB, 最早日期, 最新日期)
#[pyfunction]
pub fn get_backup_dataset_info(backup_file: &str, storage_format: &str) -> PyResult<(usize, usize, f64, String, String)> {
    crate::backup::get_backup_info_detailed(backup_file, storage_format)
}
