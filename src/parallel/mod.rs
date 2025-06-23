use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::backup::BackupManager;
use numpy::{PyArray2, ToPyArray};
use ndarray::Array2;
use crate::multiprocess::{MultiProcessExecutor, MultiProcessConfig};

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
    chunk_size=None
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
    chunk_size: Option<usize>,
) -> PyResult<&'py PyArray2<PyObject>> {
    
    // --- 多进程模式 ---
    println!("调度到Rust原生多进程执行...");
    
    let parsed_args: Vec<(i32, String)> = args
        .iter()
        .map(|item| {
            let list: &PyList = item.downcast()?;
            let date: i32 = list.get_item(0)?.extract()?;
            let code: String = list.get_item(1)?.extract()?;
            Ok((date, code))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let multiprocess_config = MultiProcessConfig {
        num_processes: num_threads,
        backup_batch_size,
        backup_file,
        storage_format: storage_format.to_string(),
        resume_from_backup,
        ..Default::default()
    };
    
    // 执行多进程任务
    let mut multiprocess_executor = MultiProcessExecutor::new(multiprocess_config)?;
    let multiprocess_results = multiprocess_executor.run_multiprocess(py, func, parsed_args, go_class, progress_callback, chunk_size)?;

    // 转换为PyArray
    if multiprocess_results.is_empty() {
        let empty_array = Array2::<PyObject>::from_shape_vec((0, 0), vec![])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("无法创建空NDArray: {}", e)
            ))?;
        return Ok(empty_array.to_pyarray(py));
    }
    
    let num_rows = multiprocess_results.len();
    let num_cols = 2 + multiprocess_results[0].facs.len(); // date, code, + facs
    
    let mut data = Vec::with_capacity(num_rows * num_cols);
    
    for result in multiprocess_results {
        data.push(result.date.to_object(py));
        data.push(result.code.to_object(py));
        for fac in result.facs {
            data.push(fac.to_object(py));
        }
    }
    
    let array = Array2::from_shape_vec((num_rows, num_cols), data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("无法创建NDArray: {}", e)
        ))?;
    
    Ok(array.to_pyarray(py))
}

/// 查询备份数据
#[pyfunction]
#[pyo3(signature = (
    backup_file,
    date_range=None,
    codes=None,
    storage_format="json"
))]
pub fn query_backup<'py>(
    py: Python<'py>,
    backup_file: &str,
    date_range: Option<(i32, i32)>,
    codes: Option<Vec<String>>,
    storage_format: &str,
) -> PyResult<&'py PyArray2<PyObject>> {
    let backup_manager = BackupManager::new(backup_file, storage_format)?;
    let results = backup_manager.query_results(date_range, codes)?;

    // 转换为NDArray
    if results.is_empty() {
        let empty_array = Array2::<PyObject>::from_shape_vec((0, 0), vec![])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("无法创建空NDArray: {}", e)
            ))?;
        return Ok(empty_array.to_pyarray(py));
    }
    
    let num_rows = results.len();
    let num_cols = 3 + results[0].facs.len(); // date, code, timestamp + facs
    
    let mut data = Vec::with_capacity(num_rows * num_cols);
    
    for result in results {
        data.push(result.date.to_object(py));
        data.push(result.code.to_object(py));
        data.push(result.timestamp.to_object(py));
        for fac in result.facs {
            data.push(fac.to_object(py));
        }
    }
    
    let array = Array2::from_shape_vec((num_rows, num_cols), data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("无法创建NDArray: {}", e)
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
