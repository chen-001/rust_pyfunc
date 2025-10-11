use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::Arc;

// 从 parallel_computing.rs 迁移的数据结构定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub date: i64,
    pub code: String,
    pub timestamp: i64,
    pub facs: Vec<f64>,
}

// 文件格式相关常量
const HEADER_SIZE: usize = 64; // 文件头64字节
const MAX_FACTORS: usize = 256; // 临时向后兼容常量
const RECORD_SIZE: usize = 2116; // 临时向后兼容常量（对应256个因子）

// 动态计算记录大小
pub fn calculate_record_size(factor_count: usize) -> usize {
    8 +        // date: i64
    8 +        // code_hash: u64 
    8 +        // timestamp: i64
    4 +        // factor_count: u32
    4 +        // code_len: u32
    32 +       // code_bytes: [u8; 32]
    factor_count * 8 +  // factors: [f64; factor_count]
    4 // checksum: u32
}

#[repr(C, packed)]
pub struct FileHeader {
    pub magic: [u8; 8],     // 魔数 "RPBACKUP"
    pub version: u32,       // 版本号
    pub record_count: u64,  // 记录总数
    pub record_size: u32,   // 单条记录大小
    pub factor_count: u32,  // 因子数量
    pub reserved: [u8; 36], // 保留字段
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
pub struct DynamicRecord {
    date: i64,
    code_hash: u64,
    timestamp: i64,
    factor_count: u32,
    code_len: u32,
    code_bytes: [u8; 32],
    factors: Vec<f64>, // 动态大小的因子数组
    checksum: u32,
}

impl DynamicRecord {
    pub fn from_task_result(result: &TaskResult) -> Self {
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
    pub fn to_bytes(&self) -> Vec<u8> {
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
    fn from_bytes(
        bytes: &[u8],
        expected_factor_count: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < calculate_record_size(expected_factor_count) {
            return Err("Insufficient bytes for record".into());
        }

        let mut offset = 0;

        let date = i64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;

        let code_hash = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;

        let timestamp = i64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;

        let factor_count = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;

        let code_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;

        let mut code_bytes = [0u8; 32];
        code_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;

        let mut factors = Vec::with_capacity(expected_factor_count);
        for _ in 0..expected_factor_count {
            let factor = f64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
            factors.push(factor);
            offset += 8;
        }

        let checksum = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);

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

pub fn calculate_hash(s: &str) -> u64 {
    // 简单的哈希函数
    let mut hash = 0u64;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    hash
}

pub fn read_existing_backup(
    file_path: &str,
) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
    read_existing_backup_with_filter(file_path, None)
}

pub fn read_existing_backup_with_filter(
    file_path: &str,
    date_filter: Option<&HashSet<i64>>,
) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
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
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

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

fn read_existing_backup_legacy(
    file_path: &str,
) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
    read_existing_backup_legacy_with_filter(file_path, None)
}

fn read_existing_backup_legacy_with_filter(
    file_path: &str,
    date_filter: Option<&HashSet<i64>>,
) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
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
            size_bytes[0],
            size_bytes[1],
            size_bytes[2],
            size_bytes[3],
            size_bytes[4],
            size_bytes[5],
            size_bytes[6],
            size_bytes[7],
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

#[pyfunction]
#[pyo3(signature = (backup_file,))]
pub fn query_backup(backup_file: String) -> PyResult<PyObject> {
    read_backup_results(&backup_file)
}

/// 高速并行备份查询函数，专门优化大文件读取
#[pyfunction]
#[pyo3(signature = (backup_file, num_threads=None, dates=None, codes=None))]
pub fn query_backup_fast(
    backup_file: String,
    num_threads: Option<usize>,
    dates: Option<Vec<i64>>,
    codes: Option<Vec<String>>,
) -> PyResult<PyObject> {
    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());

    // 使用自定义线程池而不是全局线程池
    if let Some(threads) = num_threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create thread pool: {}",
                    e
                ))
            })?;

        pool.install(|| {
            read_backup_results_ultra_fast_v4_with_filter(
                &backup_file,
                date_filter.as_ref(),
                code_filter.as_ref(),
            )
        })
    } else {
        read_backup_results_ultra_fast_v4_with_filter(
            &backup_file,
            date_filter.as_ref(),
            code_filter.as_ref(),
        )
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
pub fn query_backup_single_column_with_filter(
    backup_file: String,
    column_index: usize,
    dates: Option<Vec<i64>>,
    codes: Option<Vec<String>>,
) -> PyResult<PyObject> {
    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());

    read_backup_results_single_column_with_filter(
        &backup_file,
        column_index,
        date_filter.as_ref(),
        code_filter.as_ref(),
    )
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
pub fn query_backup_columns_range_with_filter(
    backup_file: String,
    column_start: usize,
    column_end: usize,
    dates: Option<Vec<i64>>,
    codes: Option<Vec<String>>,
) -> PyResult<PyObject> {
    // 检查参数有效性
    if column_start > column_end {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "column_start must be <= column_end",
        ));
    }

    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());

    read_backup_results_columns_range_with_filter(
        &backup_file,
        column_start,
        column_end,
        date_filter.as_ref(),
        code_filter.as_ref(),
    )
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
pub fn query_backup_factor_only_with_filter(
    backup_file: String,
    column_index: usize,
    dates: Option<Vec<i64>>,
    codes: Option<Vec<String>>,
) -> PyResult<PyObject> {
    // 将Vec转换为HashSet以提高查找性能
    let date_filter: Option<HashSet<i64>> = dates.map(|v| v.into_iter().collect());
    let code_filter: Option<HashSet<String>> = codes.map(|v| v.into_iter().collect());

    read_backup_results_factor_only_with_filter(
        &backup_file,
        column_index,
        date_filter.as_ref(),
        code_filter.as_ref(),
    )
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
pub fn query_backup_factor_only_ultra_fast(
    backup_file: String,
    column_index: usize,
) -> PyResult<PyObject> {
    read_backup_results_factor_only_ultra_fast(&backup_file, column_index)
}

pub fn read_backup_results(file_path: &str) -> PyResult<PyObject> {
    read_backup_results_with_filter(file_path, None, None)
}

pub fn read_backup_results_with_filter(
    file_path: &str,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Python::with_gil(|py| Ok(py.None()));
    }

    let file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open backup file: {}", e))
    })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to get file metadata: {}",
                e
            ))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        // 尝试旧格式的回退处理
        return read_legacy_backup_results_with_filter(file_path, date_filter, code_filter);
    }

    // 使用内存映射进行超高速读取
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to memory map file: {}",
                e
            ))
        })?
    };
    let mmap = Arc::new(mmap);

    // 读取文件头
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

    // 验证魔数
    if &header.magic != b"RPBACKUP" {
        // 不是新格式，尝试旧格式
        return read_legacy_backup_results_with_filter(file_path, date_filter, code_filter);
    }

    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No records found in backup file",
        ));
    }

    // 检查版本并计算预期文件大小
    let factor_count = header.factor_count as usize;
    let record_size = if header.version == 2 {
        calculate_record_size(factor_count)
    } else {
        RECORD_SIZE // 旧格式使用固定大小
    };

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Backup file appears to be truncated",
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
            .chunks(std::cmp::max(
                64,
                record_count / rayon::current_num_threads(),
            ))
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
            .chunks(std::cmp::max(
                64,
                record_count / rayon::current_num_threads(),
            ))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
            .par_iter()
            .map(|chunk| {
                let mut chunk_data = Vec::with_capacity(chunk.len() * num_cols);
                let records_start = HEADER_SIZE;

                for &i in chunk {
                    let record_offset = records_start + i * RECORD_SIZE;
                    let record =
                        unsafe { &*(mmap.as_ptr().add(record_offset) as *const FixedRecord) };

                    // 检查日期过滤器
                    if let Some(date_filter) = date_filter {
                        let date = record.date; // 复制到本地变量避免unaligned reference
                        if !date_filter.contains(&date) {
                            continue;
                        }
                    }

                    // 检查代码过滤器
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code_str =
                        unsafe { std::str::from_utf8_unchecked(&record.code_bytes[..code_len]) };
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
                        factor_count,
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
fn read_legacy_backup_results(file_path: &str) -> PyResult<PyObject> {
    read_legacy_backup_results_with_filter(file_path, None, None)
}

fn read_legacy_backup_results_with_filter(
    file_path: &str,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    let mut file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open backup file: {}", e))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read backup file: {}", e))
    })?;

    if buffer.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Backup file is empty",
        ));
    }

    let mut all_results = Vec::new();
    let mut cursor = 0;

    // 尝试新的批次格式（带大小头）
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0],
            size_bytes[1],
            size_bytes[2],
            size_bytes[3],
            size_bytes[4],
            size_bytes[5],
            size_bytes[6],
            size_bytes[7],
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
                    let batch_size = bincode::serialized_size(&batch).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Serialization error: {}",
                            e
                        ))
                    })? as usize;
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
            "No valid results found in backup file",
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
            let factors_reshaped =
                factors_array.call_method1("reshape", ((num_rows, factor_count),))?;
            dict.set_item("factors", factors_reshaped)?;
        } else {
            dict.set_item("factors", factors_array)?;
        }

        Ok(dict.into())
    })
}

/// 终极版本：线程安全的并行+零分配+缓存优化
pub fn read_backup_results_ultra_fast_v4_with_filter(
    file_path: &str,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "Backup file not found",
        ));
    }

    let file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open backup file: {}", e))
    })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to get file metadata: {}",
                e
            ))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_with_filter(file_path, date_filter, code_filter);
    }

    // 内存映射
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to memory map file: {}",
                e
            ))
        })?
    };
    let mmap = Arc::new(mmap);

    // 读取文件头
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Record size mismatch: header says {}, calculated {}. File may be corrupt.",
            record_size, calculated_record_size
        )));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Backup file appears to be truncated",
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
            let factors_reshaped =
                factors_array.call_method1("reshape", ((num_rows, factor_count),))?;
            dict.set_item("factors", factors_reshaped)?;
        } else {
            dict.set_item("factors", factors_array)?;
        }

        Ok(dict.into())
    })
}

/// 单列读取函数
pub fn read_backup_results_single_column(
    file_path: &str,
    column_index: usize,
) -> PyResult<PyObject> {
    read_backup_results_single_column_with_filter(file_path, column_index, None, None)
}

pub fn read_backup_results_single_column_with_filter(
    file_path: &str,
    column_index: usize,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在",
        ));
    }

    let file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_single_column_with_filter(
            file_path,
            column_index,
            date_filter,
            code_filter,
        );
    }

    // 内存映射
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e))
        })?
    };
    let mmap = Arc::new(mmap);

    // 读取文件头
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

    if &header.magic != b"RPBACKUP" {
        return read_legacy_backup_results_single_column_with_filter(
            file_path,
            column_index,
            date_filter,
            code_filter,
        );
    }

    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item(
                "code",
                numpy.call_method1("array", (Vec::<String>::new(),))?,
            )?;
            dict.set_item("factor", numpy.call_method1("array", (Vec::<f64>::new(),))?)?;
            Ok(dict.into())
        });
    }

    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;

    // 检查列索引是否有效
    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列索引 {} 超出范围，因子列数为 {}",
            column_index, factor_count
        )));
    }

    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了",
        ));
    }

    // 使用自定义线程池并直接从mmap读取，避免大量内存复制
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 8))
        .build()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
        })?;

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
                        let code =
                            String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();

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

pub fn read_backup_results_columns_range_with_filter(
    file_path: &str,
    column_start: usize,
    column_end: usize,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在",
        ));
    }

    let file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_columns_range_with_filter(
            file_path,
            column_start,
            column_end,
            date_filter,
            code_filter,
        );
    }

    // 内存映射
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e))
        })?
    };
    let mmap = Arc::new(mmap);

    // 读取文件头
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

    if &header.magic != b"RPBACKUP" {
        return read_legacy_backup_results_columns_range_with_filter(
            file_path,
            column_start,
            column_end,
            date_filter,
            code_filter,
        );
    }

    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item(
                "code",
                numpy.call_method1("array", (Vec::<String>::new(),))?,
            )?;
            dict.set_item(
                "factors",
                numpy.call_method1("array", (Vec::<Vec<f64>>::new(),))?,
            )?;
            Ok(dict.into())
        });
    }

    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;

    // 检查列索引是否有效
    if column_start >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "起始列索引 {} 超出范围，因子列数为 {}",
            column_start, factor_count
        )));
    }

    if column_end >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "结束列索引 {} 超出范围，因子列数为 {}",
            column_end, factor_count
        )));
    }

    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了",
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

fn read_legacy_backup_results_columns_range_with_filter(
    file_path: &str,
    column_start: usize,
    column_end: usize,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    let mut file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取备份文件: {}", e))
    })?;

    if buffer.is_empty() {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item(
                "code",
                numpy.call_method1("array", (Vec::<String>::new(),))?,
            )?;
            dict.set_item(
                "factors",
                numpy.call_method1("array", (Vec::<Vec<f64>>::new(),))?,
            )?;
            Ok(dict.into())
        });
    }

    let mut cursor = 0;
    let mut all_results = Vec::new();

    // 尝试新的批次格式
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0],
            size_bytes[1],
            size_bytes[2],
            size_bytes[3],
            size_bytes[4],
            size_bytes[5],
            size_bytes[6],
            size_bytes[7],
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
fn read_legacy_backup_results_single_column_with_filter(
    file_path: &str,
    column_index: usize,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    let mut file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取备份文件: {}", e))
    })?;

    if buffer.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件为空",
        ));
    }

    let mut all_results = Vec::new();
    let mut cursor = 0;

    // 尝试新的批次格式（带大小头）
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0],
            size_bytes[1],
            size_bytes[2],
            size_bytes[3],
            size_bytes[4],
            size_bytes[5],
            size_bytes[6],
            size_bytes[7],
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
                    let batch_size = bincode::serialized_size(&batch).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "序列化错误: {}",
                            e
                        ))
                    })? as usize;
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
            "备份文件中没有找到有效结果",
        ));
    }

    // 检查列索引是否有效
    if let Some(first_result) = all_results.first() {
        if column_index >= first_result.facs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "列索引 {} 超出范围，因子列数为 {}",
                column_index,
                first_result.facs.len()
            )));
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

/// 读取备份文件中的指定列因子值（纯因子值数组）
pub fn read_backup_results_factor_only(file_path: &str, column_index: usize) -> PyResult<PyObject> {
    read_backup_results_factor_only_with_filter(file_path, column_index, None, None)
}

pub fn read_backup_results_factor_only_with_filter(
    file_path: &str,
    column_index: usize,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在",
        ));
    }

    let file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        return read_legacy_backup_results_factor_only_with_filter(
            file_path,
            column_index,
            date_filter,
            code_filter,
        );
    }

    // 内存映射
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e))
        })?
    };
    let mmap = Arc::new(mmap);

    // 读取文件头
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

    if &header.magic != b"RPBACKUP" {
        return read_legacy_backup_results_factor_only_with_filter(
            file_path,
            column_index,
            date_filter,
            code_filter,
        );
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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列索引 {} 超出范围，因子列数为 {}",
            column_index, factor_count
        )));
    }

    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了",
        ));
    }

    // 使用自定义线程池避免资源竞争和泄漏
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 8))
        .build()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
        })?;

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
                            let code =
                                String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();

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
fn read_legacy_backup_results_factor_only_with_filter(
    file_path: &str,
    column_index: usize,
    date_filter: Option<&HashSet<i64>>,
    code_filter: Option<&HashSet<String>>,
) -> PyResult<PyObject> {
    let mut file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取备份文件: {}", e))
    })?;

    if buffer.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件为空",
        ));
    }

    let mut all_results = Vec::new();
    let mut cursor = 0;

    // 尝试新的批次格式（带大小头）
    while cursor + 8 <= buffer.len() {
        let size_bytes = &buffer[cursor..cursor + 8];
        let batch_size = u64::from_le_bytes([
            size_bytes[0],
            size_bytes[1],
            size_bytes[2],
            size_bytes[3],
            size_bytes[4],
            size_bytes[5],
            size_bytes[6],
            size_bytes[7],
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
                    let batch_size = bincode::serialized_size(&batch).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "序列化错误: {}",
                            e
                        ))
                    })? as usize;
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
            "备份文件中没有找到有效结果",
        ));
    }

    // 检查列索引是否有效
    if let Some(first_result) = all_results.first() {
        if column_index >= first_result.facs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "列索引 {} 超出范围，因子列数为 {}",
                column_index,
                first_result.facs.len()
            )));
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

/// 超高速因子值读取（直接字节偏移版本）
pub fn read_backup_results_factor_only_ultra_fast(
    file_path: &str,
    column_index: usize,
) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在",
        ));
    }

    let file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        return read_backup_results_factor_only(&file_path, column_index);
    }

    // 内存映射
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e))
        })?
    };

    #[cfg(target_family = "unix")]
    unsafe {
        // 提示内核按顺序访问，增加预读窗口
        let _ = libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            file_len,
            libc::MADV_SEQUENTIAL,
        );
    }

    // 读取文件头
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

    if &header.magic != b"RPBACKUP" {
        return read_backup_results_factor_only(&file_path, column_index);
    }

    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| Ok(PyArray1::<f64>::from_vec(py, Vec::new()).into_py(py)));
    }

    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;

    // 检查列索引是否有效
    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列索引 {} 超出范围，因子列数为 {}",
            column_index, factor_count
        )));
    }

    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了",
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
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
        })?;

    // 并行读取所有因子值
    let mut factors = vec![0f64; record_count];
    pool.install(|| {
        factors
            .par_iter_mut()
            .enumerate()
            .with_min_len(4096)
            .for_each(|(i, slot)| {
                let record_offset = records_start + i * record_size;

                // 直接读取因子值，完全跳过其他字段的解析
                unsafe {
                    let factor_ptr = mmap.as_ptr().add(record_offset + factor_offset) as *const f64;
                    *slot = *factor_ptr;
                }
            });
    });

    // 显式释放mmap
    drop(mmap);

    // 创建numpy数组
    Python::with_gil(|py| Ok(PyArray1::from_vec(py, factors).into_py(py)))
}

/// 超高速查询备份文件中的指定列（完整版本v2）
/// 直接字节偏移读取，避免完整记录解析
pub fn read_backup_results_single_column_ultra_fast_v2(
    file_path: &str,
    column_index: usize,
) -> PyResult<PyObject> {
    if !Path::new(file_path).exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在",
        ));
    }

    let file = File::open(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
    })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        return read_backup_results_single_column(&file_path, column_index);
    }

    // 内存映射
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e))
        })?
    };

    // 读取文件头
    let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };

    if &header.magic != b"RPBACKUP" {
        return read_backup_results_single_column(&file_path, column_index);
    }

    let record_count = header.record_count as usize;
    if record_count == 0 {
        return Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("date", numpy.call_method1("array", (Vec::<i64>::new(),))?)?;
            dict.set_item(
                "code",
                numpy.call_method1("array", (Vec::<String>::new(),))?,
            )?;
            dict.set_item("factor", numpy.call_method1("array", (Vec::<f64>::new(),))?)?;
            Ok(dict.into())
        });
    }

    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;

    // 检查列索引是否有效
    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列索引 {} 超出范围，因子列数为 {}",
            column_index, factor_count
        )));
    }

    let calculated_record_size = calculate_record_size(factor_count);
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let expected_size = HEADER_SIZE + record_count * record_size;
    if file_len < expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件似乎被截断了",
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
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
        })?;

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
                        let code_len_ptr =
                            mmap.as_ptr().add(record_offset + code_len_offset) as *const u32;
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
                        let factor_ptr =
                            mmap.as_ptr().add(record_offset + factor_offset) as *const f64;
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
