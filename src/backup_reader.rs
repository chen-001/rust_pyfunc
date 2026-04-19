use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::ptr;
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

// v3 格式常量
const V3_CODE_OFFSET: usize = 8;
const V3_CODE_SIZE: usize = 16;
const V3_TIMESTAMP_OFFSET: usize = 24;
const V3_FACTOR_BASE_OFFSET: usize = 32;
const V3_FACTOR_SIZE: usize = 4; // f32

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

// ==================== v4 格式辅助 ====================

/// v4 chunk 索引条目
#[derive(Debug, Clone)]
struct ChunkInfo {
    compressed_size: usize,
    record_count: usize,
    data_offset: usize, // chunk 数据在 mmap 中的起始偏移
}

/// 扫描 v4 文件的数据区，构建 chunk 索引
fn build_chunk_index(mmap: &[u8], header_version: u32) -> Result<Vec<ChunkInfo>, String> {
    if header_version != 4 {
        return Err(format!("build_chunk_index 仅支持 v4, 当前版本 {}", header_version));
    }
    let chunk_count = u32::from_le_bytes(mmap[28..32].try_into().unwrap()) as usize;
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut offset = HEADER_SIZE;

    for _ in 0..chunk_count {
        if offset + 8 > mmap.len() {
            return Err(format!("chunk 头部越界: offset={}, len={}", offset, mmap.len()));
        }
        let compressed_size = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
        let record_count = u32::from_le_bytes(mmap[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let data_offset = offset + 8;
        if data_offset + compressed_size > mmap.len() {
            return Err(format!(
                "chunk 数据越界: offset={}, compressed_size={}, len={}",
                data_offset, compressed_size, mmap.len()
            ));
        }
        chunks.push(ChunkInfo {
            compressed_size,
            record_count,
            data_offset,
        });
        offset = data_offset + compressed_size;
    }
    Ok(chunks)
}

/// 将 v4 文件所有 chunk 解压，拼接为连续的 v3 格式记录字节
/// 返回 (解压后的连续字节, 总记录数, record_size)
fn decompress_all_chunks_v4(mmap: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let record_size = u32::from_le_bytes(mmap[20..24].try_into().unwrap()) as usize;
    let record_count = u64::from_le_bytes(mmap[12..20].try_into().unwrap()) as usize;
    let chunks = build_chunk_index(mmap, 4)?;

    let mut buf = Vec::with_capacity(record_count * record_size);
    for chunk in &chunks {
        let compressed_data = &mmap[chunk.data_offset..chunk.data_offset + chunk.compressed_size];
        let decompressed = zstd::decode_all(compressed_data)
            .map_err(|e| format!("zstd 解压失败: {}", e))?;
        buf.extend_from_slice(&decompressed);
    }

    // 校验解压后大小
    if buf.len() != record_count * record_size {
        return Err(format!(
            "解压后大小不匹配: 期望 {} ({} * {}), 实际 {}",
            record_count * record_size, record_count, record_size, buf.len()
        ));
    }

    Ok((buf, record_count, record_size))
}

// ==================== v3 格式辅助函数 ====================

/// 计算 v3 记录大小
pub fn calculate_v3_record_size(factor_count: usize) -> usize {
    8 + V3_CODE_SIZE + 8 + factor_count * V3_FACTOR_SIZE + 4
}

/// 从 v3 记录字节读取 null-terminated code 字符串
fn read_v3_code(bytes: &[u8]) -> String {
    let code_end = bytes[V3_CODE_OFFSET..V3_CODE_OFFSET + V3_CODE_SIZE]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(V3_CODE_SIZE);
    String::from_utf8_lossy(&bytes[V3_CODE_OFFSET..V3_CODE_OFFSET + code_end]).to_string()
}

/// 从 v3 记录字节读取单个 f32 因子并转为 f64
fn read_v3_factor_f64(bytes: &[u8], column_index: usize) -> f64 {
    let offset = V3_FACTOR_BASE_OFFSET + column_index * V3_FACTOR_SIZE;
    let bits = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    f32::from_bits(bits) as f64
}

/// 从 v3 记录字节读取日期
fn read_v3_date(bytes: &[u8]) -> i64 {
    i64::from_le_bytes(bytes[0..8].try_into().unwrap())
}

/// 从 v3 记录字节读取时间戳
fn read_v3_timestamp(bytes: &[u8]) -> i64 {
    i64::from_le_bytes(bytes[V3_TIMESTAMP_OFFSET..V3_TIMESTAMP_OFFSET + 8].try_into().unwrap())
}

/// 从 v3 记录字节读取所有因子（转为 f64）
fn read_v3_all_factors_f64(bytes: &[u8], factor_count: usize) -> Vec<f64> {
    (0..factor_count)
        .map(|i| read_v3_factor_f64(bytes, i))
        .collect()
}

/// 从 v3 记录字节读取指定范围的因子（转为 f64）
fn read_v3_factors_range_f64(bytes: &[u8], start: usize, end: usize) -> Vec<f64> {
    (start..=end).map(|i| read_v3_factor_f64(bytes, i)).collect()
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
    let records_start = HEADER_SIZE;

    // 检查版本号
    if header.version == 4 {
        // v4 分块压缩格式
        let (decompressed, total_count, record_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| Box::<dyn std::error::Error>::from(e))?;
        for i in 0..total_count {
            let record_offset = i * record_size;
            if record_offset + record_size > decompressed.len() {
                break;
            }
            let record_bytes = &decompressed[record_offset..record_offset + record_size];
            let date = read_v3_date(record_bytes);
            if let Some(filter) = date_filter {
                if !filter.contains(&date) {
                    continue;
                }
            }
            let code = read_v3_code(record_bytes);
            existing_tasks.insert((date, code));
        }
    } else if header.version == 3 {
        // v3 格式
        let record_size = calculate_v3_record_size(factor_count);
        for i in 0..record_count {
            let record_offset = records_start + i * record_size;
            if record_offset + record_size > mmap.len() {
                break;
            }
            let record_bytes = &mmap[record_offset..record_offset + record_size];
            let date = read_v3_date(record_bytes);
            if let Some(filter) = date_filter {
                if !filter.contains(&date) {
                    continue;
                }
            }
            let code = read_v3_code(record_bytes);
            existing_tasks.insert((date, code));
        }
    } else if header.version == 2 {
        // v2 动态格式
        let record_size = calculate_record_size(factor_count);
        for i in 0..record_count {
            let record_offset = records_start + i * record_size;
            let record_bytes = &mmap[record_offset..record_offset + record_size];

            match DynamicRecord::from_bytes(record_bytes, factor_count) {
                Ok(record) => {
                    if let Some(filter) = date_filter {
                        if !filter.contains(&record.date) {
                            continue;
                        }
                    }
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                    existing_tasks.insert((record.date, code));
                }
                Err(_) => continue,
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
/// - use_single_thread: 是否使用单线程读取
///
/// 返回:
/// 包含三个numpy数组的字典: {"date": 日期数组, "code": 代码数组, "factor": 指定列的因子值数组}
#[pyfunction]
#[pyo3(signature = (backup_file, column_index, use_single_thread=false))]
pub fn query_backup_single_column(
    backup_file: String,
    column_index: usize,
    use_single_thread: bool,
) -> PyResult<PyObject> {
    if use_single_thread {
        read_backup_results_single_column_ultra_fast_v2_single_thread(&backup_file, column_index)
    } else {
        // 优先使用超高速版本
        read_backup_results_single_column_ultra_fast_v2(&backup_file, column_index)
    }
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
    let version = header.version;
    let record_size = if version == 3 || version == 4 {
        calculate_v3_record_size(factor_count)
    } else if version == 2 {
        calculate_record_size(factor_count)
    } else {
        RECORD_SIZE // 旧格式使用固定大小
    };

    // v4 不做简单的 expected_size 检查（压缩后大小不等于 record_count * record_size）
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Backup file appears to be truncated",
            ));
        }
    }

    // 预计算矩阵维度
    let num_cols = 3 + factor_count;

    // 根据版本选择不同的读取方式
    let parallel_results: Result<Vec<_>, _> = if version == 4 {
        // v4 分块压缩格式：先解压到连续缓冲区，然后像 v3 一样解析
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        let decompressed = Arc::new(decompressed);

        (0..total_count)
            .collect::<Vec<_>>()
            .chunks(std::cmp::max(64, total_count / rayon::current_num_threads()))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
            .par_iter()
            .map(|chunk| {
                let mut chunk_data = Vec::with_capacity(chunk.len() * num_cols);
                for &i in chunk {
                    let record_offset = i * rec_size;
                    let record_bytes = &decompressed[record_offset..record_offset + rec_size];

                    let date = read_v3_date(record_bytes);
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&date) {
                            continue;
                        }
                    }

                    let code_str = read_v3_code(record_bytes);
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&code_str) {
                            continue;
                        }
                    }

                    let timestamp = read_v3_timestamp(record_bytes);

                    chunk_data.push(date as f64);
                    let code_num = if let Ok(num) = code_str.parse::<f64>() {
                        num
                    } else {
                        f64::NAN
                    };
                    chunk_data.push(code_num);
                    chunk_data.push(timestamp as f64);

                    for j in 0..factor_count {
                        chunk_data.push(read_v3_factor_f64(record_bytes, j));
                    }
                }
                Ok(chunk_data)
            })
            .collect()
    } else if header.version == 3 {
        // v3 格式读取
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

                    let date = read_v3_date(record_bytes);
                    if let Some(date_filter) = date_filter {
                        if !date_filter.contains(&date) {
                            continue;
                        }
                    }

                    let code_str = read_v3_code(record_bytes);
                    if let Some(code_filter) = code_filter {
                        if !code_filter.contains(&code_str) {
                            continue;
                        }
                    }

                    let timestamp = read_v3_timestamp(record_bytes);

                    chunk_data.push(date as f64);
                    let code_num = if let Ok(num) = code_str.parse::<f64>() {
                        num
                    } else {
                        f64::NAN
                    };
                    chunk_data.push(code_num);
                    chunk_data.push(timestamp as f64);

                    for j in 0..factor_count {
                        chunk_data.push(read_v3_factor_f64(record_bytes, j));
                    }
                }

                Ok(chunk_data)
            })
            .collect()
    } else if header.version == 2 {
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

    // --- 使用文件头中的 record_size ---
    let record_size = header.record_size as usize;
    let factor_count = header.factor_count as usize;
    let version = header.version;

    // 根据版本校验 record_size
    let calculated_record_size = if version == 3 || version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_record_size(factor_count)
    };
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Record size mismatch: header says {}, calculated {}. File may be corrupt.",
            record_size, calculated_record_size
        )));
    }

    // v4 不做简单的 expected_size 检查
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Backup file appears to be truncated",
            ));
        }
    }

    // --- 并行收集为元组，再转换为Python字典 ---
    let records_start = HEADER_SIZE;
    let results: Vec<_> = if version == 4 {
        // v4 分块压缩格式：先解压到连续缓冲区
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        (0..total_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = i * rec_size;
                let record_bytes = &decompressed[record_offset..record_offset + rec_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                let code = read_v3_code(record_bytes);
                if let Some(code_filter) = code_filter {
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                let timestamp = read_v3_timestamp(record_bytes);
                let factors = read_v3_all_factors_f64(record_bytes, factor_count);
                Some((date, code, timestamp, factors))
            })
            .collect()
    } else if version == 3 {
        // v3 格式解析
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                let code = read_v3_code(record_bytes);
                if let Some(code_filter) = code_filter {
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                let timestamp = read_v3_timestamp(record_bytes);
                let factors = read_v3_all_factors_f64(record_bytes, factor_count);
                Some((date, code, timestamp, factors))
            })
            .collect()
    } else {
        // v2 格式解析
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];

                match DynamicRecord::from_bytes(record_bytes, factor_count) {
                    Ok(record) => {
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&record.date) {
                                return None;
                            }
                        }

                        let code_len = std::cmp::min(record.code_len as usize, 32);
                        let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();

                        if let Some(code_filter) = code_filter {
                            if !code_filter.contains(&code) {
                                return None;
                            }
                        }

                        Some((record.date, code, record.timestamp, record.factors))
                    }
                    Err(_) => None
                }
            })
            .collect()
    };

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

    let calculated_record_size = if header.version == 3 || header.version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_record_size(factor_count)
    };
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let version = header.version;

    // v4 不做简单的 expected_size 检查
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "备份文件似乎被截断了",
            ));
        }
    }

    let records_start = HEADER_SIZE;

    let results: Vec<_> = if version == 4 {
        // v4 分块压缩格式
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        (0..total_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = i * rec_size;
                let record_bytes = &decompressed[record_offset..record_offset + rec_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                let code = read_v3_code(record_bytes);
                if let Some(code_filter) = code_filter {
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                let factor_value = read_v3_factor_f64(record_bytes, column_index);
                Some((date, code, factor_value))
            })
            .collect()
    } else if version == 3 {
        // v3 格式单列读取
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                let code = read_v3_code(record_bytes);
                if let Some(code_filter) = code_filter {
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                let factor_value = read_v3_factor_f64(record_bytes, column_index);
                Some((date, code, factor_value))
            })
            .collect()
    } else {
        // v2 格式
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(std::cmp::min(rayon::current_num_threads(), 8))
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
            })?;

        pool.install(|| {
            (0..record_count)
                .into_par_iter()
                .filter_map(|i| {
                    let record_offset = records_start + i * record_size;
                    let record_bytes = &mmap[record_offset..record_offset + record_size];

                    match DynamicRecord::from_bytes(record_bytes, factor_count) {
                        Ok(record) => {
                            if let Some(date_filter) = date_filter {
                                if !date_filter.contains(&record.date) {
                                    return None;
                                }
                            }

                            let code_len = std::cmp::min(record.code_len as usize, 32);
                            let code =
                                String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();

                            if let Some(code_filter) = code_filter {
                                if !code_filter.contains(&code) {
                                    return None;
                                }
                            }

                            let factor_value = if column_index < record.factors.len() {
                                record.factors[column_index]
                            } else {
                                f64::NAN
                            };

                            Some((record.date, code, factor_value))
                        }
                        Err(_) => None
                    }
                })
                .collect()
        })
    };

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

    let calculated_record_size = if header.version == 3 || header.version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_record_size(factor_count)
    };
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let version = header.version;

    // v4 不做简单的 expected_size 检查
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "备份文件似乎被截断了",
            ));
        }
    }

    // 并行读取指定列范围
    let records_start = HEADER_SIZE;
    let num_columns = column_end - column_start + 1;

    let results: Vec<_> = if version == 4 {
        // v4 分块压缩格式
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        (0..total_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = i * rec_size;
                let record_bytes = &decompressed[record_offset..record_offset + rec_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                let code = read_v3_code(record_bytes);
                if let Some(code_filter) = code_filter {
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                let factor_values = read_v3_factors_range_f64(record_bytes, column_start, column_end);
                Some((date, code, factor_values))
            })
            .collect()
    } else if version == 3 {
        // v3 格式列范围读取
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                let code = read_v3_code(record_bytes);
                if let Some(code_filter) = code_filter {
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                let factor_values = read_v3_factors_range_f64(record_bytes, column_start, column_end);
                Some((date, code, factor_values))
            })
            .collect()
    } else {
        // v2 格式列范围读取
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];

                match DynamicRecord::from_bytes(record_bytes, factor_count) {
                    Ok(record) => {
                        if let Some(date_filter) = date_filter {
                            if !date_filter.contains(&record.date) {
                                return None;
                            }
                        }

                        let code_len = std::cmp::min(record.code_len as usize, 32);
                        let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();

                        if let Some(code_filter) = code_filter {
                            if !code_filter.contains(&code) {
                                return None;
                            }
                        }

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
                    Err(_) => None
                }
            })
            .collect()
    };

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

    let calculated_record_size = if header.version == 3 || header.version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_record_size(factor_count)
    };
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let version = header.version;

    // v4 不做简单的 expected_size 检查
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "备份文件似乎被截断了",
            ));
        }
    }

    let records_start = HEADER_SIZE;

    let factors: Vec<f64> = if version == 4 {
        // v4 分块压缩格式
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        (0..total_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = i * rec_size;
                let record_bytes = &decompressed[record_offset..record_offset + rec_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                if let Some(code_filter) = code_filter {
                    let code = read_v3_code(record_bytes);
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                Some(read_v3_factor_f64(record_bytes, column_index))
            })
            .collect()
    } else if version == 3 {
        // v3 格式因子读取
        (0..record_count)
            .into_par_iter()
            .filter_map(|i| {
                let record_offset = records_start + i * record_size;
                let record_bytes = &mmap[record_offset..record_offset + record_size];

                let date = read_v3_date(record_bytes);
                if let Some(date_filter) = date_filter {
                    if !date_filter.contains(&date) {
                        return None;
                    }
                }

                if let Some(code_filter) = code_filter {
                    let code = read_v3_code(record_bytes);
                    if !code_filter.contains(&code) {
                        return None;
                    }
                }

                Some(read_v3_factor_f64(record_bytes, column_index))
            })
            .collect()
    } else {
        // v2 格式因子读取
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(std::cmp::min(rayon::current_num_threads(), 8))
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
            })?;

        pool.install(|| {
            (0..record_count)
                .into_par_iter()
                .filter_map(|i| {
                    let record_offset = records_start + i * record_size;
                    let record_bytes = &mmap[record_offset..record_offset + record_size];

                    match DynamicRecord::from_bytes(record_bytes, factor_count) {
                        Ok(record) => {
                            if let Some(date_filter) = date_filter {
                                if !date_filter.contains(&record.date) {
                                    return None;
                                }
                            }

                            if let Some(code_filter) = code_filter {
                                let code_len = std::cmp::min(record.code_len as usize, 32);
                                let code =
                                    String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                                if !code_filter.contains(&code) {
                                    return None;
                                }
                            }

                            Some(if column_index < record.factors.len() {
                                record.factors[column_index]
                            } else {
                                f64::NAN
                            })
                        }
                        Err(_) => Some(f64::NAN)
                    }
                })
                .collect()
        })
    };

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

    let calculated_record_size = if header.version == 3 || header.version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_record_size(factor_count)
    };
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    // v4 不做简单的 expected_size 检查
    let version = header.version;
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "备份文件似乎被截断了",
            ));
        }
    }

    // 直接偏移读取因子值
    let records_start = HEADER_SIZE;

    if version == 4 {
        // v4 分块压缩格式：解压后直接偏移读取
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        let factor_offset = V3_FACTOR_BASE_OFFSET + column_index * V3_FACTOR_SIZE;

        let mut factors = vec![0f64; total_count];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(std::cmp::min(rayon::current_num_threads(), 16))
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
            })?;

        pool.install(|| {
            factors
                .par_iter_mut()
                .enumerate()
                .with_min_len(4096)
                .for_each(|(i, slot)| {
                    let record_offset = i * rec_size;
                    let ptr = decompressed.as_ptr().wrapping_add(record_offset + factor_offset);
                    unsafe {
                        let bits = u32::from_le(ptr::read_unaligned(ptr as *const u32));
                        *slot = f32::from_bits(bits) as f64;
                    }
                });
        });

        drop(mmap);
        return Python::with_gil(|py| Ok(PyArray1::from_vec(py, factors).into_py(py)));
    }

    if version == 3 {
        // v3: factor_base_offset = 32, factor_size = 4 (f32)
        let factor_offset = V3_FACTOR_BASE_OFFSET + column_index * V3_FACTOR_SIZE;

        let mut factors = vec![0f64; record_count];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(std::cmp::min(rayon::current_num_threads(), 16))
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
            })?;

        pool.install(|| {
            factors
                .par_iter_mut()
                .enumerate()
                .with_min_len(4096)
                .for_each(|(i, slot)| {
                    let record_offset = records_start + i * record_size;
                    unsafe {
                        let ptr = mmap.as_ptr().add(record_offset + factor_offset);
                        let bits = u32::from_le(ptr::read_unaligned(ptr as *const u32));
                        *slot = f32::from_bits(bits) as f64;
                    }
                });
        });

        drop(mmap);
        return Python::with_gil(|py| Ok(PyArray1::from_vec(py, factors).into_py(py)));
    }

    // v2: 原有逻辑
    let factor_base_offset = 8 + 8 + 8 + 4 + 4 + 32;
    let factor_offset = factor_base_offset + column_index * 8;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 16))
        .build()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
        })?;

    let mut factors = vec![0f64; record_count];
    pool.install(|| {
        factors
            .par_iter_mut()
            .enumerate()
            .with_min_len(4096)
            .for_each(|(i, slot)| {
                let record_offset = records_start + i * record_size;
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

/// 超高速查询备份文件中的指定列（单线程版本v2）
/// 直接字节偏移读取，避免完整记录解析
pub fn read_backup_results_single_column_ultra_fast_v2_single_thread(
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

    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射文件到内存: {}", e))
        })?
    };

    #[cfg(target_family = "unix")]
    unsafe {
        // 单线程顺序读取时提示内核扩大预读窗口，降低缺页开销
        let _ = libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            file_len,
            libc::MADV_SEQUENTIAL,
        );
    }

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

    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列索引 {} 超出范围，因子列数为 {}",
            column_index, factor_count
        )));
    }

    let calculated_record_size = if header.version == 3 || header.version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_record_size(factor_count)
    };
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let version = header.version;

    // v4 不做简单的 expected_size 检查
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "备份文件似乎被截断了",
            ));
        }
    }

    let records_start = HEADER_SIZE;
    let mut dates = Vec::with_capacity(record_count);
    let mut codes = Vec::with_capacity(record_count);
    let mut factors = Vec::with_capacity(record_count);

    if version == 4 {
        // v4 分块压缩格式：解压后顺序读取
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        let factor_offset = V3_FACTOR_BASE_OFFSET + column_index * V3_FACTOR_SIZE;
        for i in 0..total_count {
            let record_offset = i * rec_size;
            let date = i64::from_le_bytes(decompressed[record_offset..record_offset + 8].try_into().unwrap());
            let code = {
                let code_start = record_offset + V3_CODE_OFFSET;
                let code_end = (0..V3_CODE_SIZE).position(|j| decompressed[code_start + j] == 0).unwrap_or(V3_CODE_SIZE);
                String::from_utf8_lossy(&decompressed[code_start..code_start + code_end]).into_owned()
            };
            let factor = {
                let off = record_offset + factor_offset;
                let bits = u32::from_le_bytes(decompressed[off..off + 4].try_into().unwrap());
                f32::from_bits(bits) as f64
            };
            dates.push(date);
            codes.push(code);
            factors.push(factor);
        }
    } else if header.version == 3 {
        // v3: date(0) + code(8,16) + timestamp(24) + factors(32) - null-terminated code, f32 factors
        let factor_offset = V3_FACTOR_BASE_OFFSET + column_index * V3_FACTOR_SIZE;
        for i in 0..record_count {
            let record_offset = records_start + i * record_size;
            let date = unsafe {
                let ptr = mmap.as_ptr().add(record_offset) as *const i64;
                i64::from_le(ptr::read_unaligned(ptr))
            };
            let code = unsafe {
                let code_ptr = mmap.as_ptr().add(record_offset + V3_CODE_OFFSET);
                let code_end = (0..V3_CODE_SIZE).position(|j| *code_ptr.add(j) == 0).unwrap_or(V3_CODE_SIZE);
                let slice = std::slice::from_raw_parts(code_ptr, code_end);
                String::from_utf8_lossy(slice).into_owned()
            };
            let factor = unsafe {
                let ptr = mmap.as_ptr().add(record_offset + factor_offset);
                let bits = u32::from_le(ptr::read_unaligned(ptr as *const u32));
                f32::from_bits(bits) as f64
            };
            dates.push(date);
            codes.push(code);
            factors.push(factor);
        }
    } else {
        // v2: 原有逻辑
        let code_len_offset = 8 + 8 + 8 + 4;
        let code_bytes_offset = code_len_offset + 4;
        let factor_base_offset = 8 + 8 + 8 + 4 + 4 + 32;
        let factor_offset = factor_base_offset + column_index * 8;

        for i in 0..record_count {
            let record_offset = records_start + i * record_size;
            let date = unsafe {
                let date_ptr = mmap.as_ptr().add(record_offset) as *const i64;
                i64::from_le(ptr::read_unaligned(date_ptr))
            };
            let code_len = unsafe {
                let code_len_ptr = mmap.as_ptr().add(record_offset + code_len_offset) as *const u32;
                std::cmp::min(u32::from_le(ptr::read_unaligned(code_len_ptr)) as usize, 32)
            };
            let code = unsafe {
                let code_bytes_ptr = mmap.as_ptr().add(record_offset + code_bytes_offset);
                let code_slice = std::slice::from_raw_parts(code_bytes_ptr, code_len);
                String::from_utf8_lossy(code_slice).into_owned()
            };
            let factor = unsafe {
                let factor_bits_ptr = mmap.as_ptr().add(record_offset + factor_offset) as *const u64;
                f64::from_bits(u64::from_le(ptr::read_unaligned(factor_bits_ptr)))
            };
            dates.push(date);
            codes.push(code);
            factors.push(factor);
        }
    }

    drop(mmap);

    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("date", PyArray1::from_vec(py, dates))?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("factor", PyArray1::from_vec(py, factors))?;
        Ok(dict.into())
    })
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

    let calculated_record_size = if header.version == 3 || header.version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_record_size(factor_count)
    };
    if record_size != calculated_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 文件头显示 {}, 计算得到 {}. 文件可能损坏.",
            record_size, calculated_record_size
        )));
    }

    let version = header.version;

    // v4 不做简单的 expected_size 检查
    if version != 4 {
        let expected_size = HEADER_SIZE + record_count * record_size;
        if file_len < expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "备份文件似乎被截断了",
            ));
        }
    }

    let records_start = HEADER_SIZE;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(rayon::current_num_threads(), 16))
        .build()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("创建线程池失败: {}", e))
        })?;

    let mut dates = Vec::with_capacity(record_count);
    let mut codes = Vec::with_capacity(record_count);
    let mut factors = Vec::with_capacity(record_count);

    const BATCH_SIZE: usize = 10000;
    let num_batches = (record_count + BATCH_SIZE - 1) / BATCH_SIZE;

    if version == 4 {
        // v4 分块压缩格式：解压后并行读取
        let (decompressed, total_count, rec_size) = decompress_all_chunks_v4(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        let factor_offset = V3_FACTOR_BASE_OFFSET + column_index * V3_FACTOR_SIZE;
        let v4_num_batches = (total_count + BATCH_SIZE - 1) / BATCH_SIZE;

        pool.install(|| {
            let batch_results: Vec<Vec<(i64, String, f64)>> = (0..v4_num_batches)
                .into_par_iter()
                .map(|batch_idx| {
                    let start_idx = batch_idx * BATCH_SIZE;
                    let end_idx = std::cmp::min(start_idx + BATCH_SIZE, total_count);
                    let mut batch_data = Vec::with_capacity(end_idx - start_idx);

                    for i in start_idx..end_idx {
                        let record_offset = i * rec_size;
                        let date = i64::from_le_bytes(
                            decompressed[record_offset..record_offset + 8].try_into().unwrap()
                        );
                        let code = {
                            let code_start = record_offset + V3_CODE_OFFSET;
                            let code_end = (0..V3_CODE_SIZE)
                                .position(|j| decompressed[code_start + j] == 0)
                                .unwrap_or(V3_CODE_SIZE);
                            String::from_utf8_lossy(
                                &decompressed[code_start..code_start + code_end]
                            ).into_owned()
                        };
                        let factor = {
                            let off = record_offset + factor_offset;
                            let bits = u32::from_le_bytes(
                                decompressed[off..off + 4].try_into().unwrap()
                            );
                            f32::from_bits(bits) as f64
                        };
                        batch_data.push((date, code, factor));
                    }
                    batch_data
                })
                .collect();

            for batch in batch_results {
                for (date, code, factor) in batch {
                    dates.push(date);
                    codes.push(code);
                    factors.push(factor);
                }
            }
        });
    } else if version == 3 {
        // v3 并行读取
        let factor_offset = V3_FACTOR_BASE_OFFSET + column_index * V3_FACTOR_SIZE;

        pool.install(|| {
            let batch_results: Vec<Vec<(i64, String, f64)>> = (0..num_batches)
                .into_par_iter()
                .map(|batch_idx| {
                    let start_idx = batch_idx * BATCH_SIZE;
                    let end_idx = std::cmp::min(start_idx + BATCH_SIZE, record_count);
                    let mut batch_data = Vec::with_capacity(end_idx - start_idx);

                    for i in start_idx..end_idx {
                        let record_offset = records_start + i * record_size;
                        let date = unsafe {
                            let ptr = mmap.as_ptr().add(record_offset) as *const i64;
                            i64::from_le(ptr::read_unaligned(ptr))
                        };
                        let code = unsafe {
                            let code_ptr = mmap.as_ptr().add(record_offset + V3_CODE_OFFSET);
                            let code_end = (0..V3_CODE_SIZE).position(|j| *code_ptr.add(j) == 0).unwrap_or(V3_CODE_SIZE);
                            let slice = std::slice::from_raw_parts(code_ptr, code_end);
                            String::from_utf8_lossy(slice).into_owned()
                        };
                        let factor = unsafe {
                            let ptr = mmap.as_ptr().add(record_offset + factor_offset);
                            let bits = u32::from_le(ptr::read_unaligned(ptr as *const u32));
                            f32::from_bits(bits) as f64
                        };
                        batch_data.push((date, code, factor));
                    }
                    batch_data
                })
                .collect();

            for batch in batch_results {
                for (date, code, factor) in batch {
                    dates.push(date);
                    codes.push(code);
                    factors.push(factor);
                }
            }
        });
    } else {
        // v2 并行读取
        let code_len_offset = 8 + 8 + 8 + 4;
        let code_bytes_offset = code_len_offset + 4;
        let factor_base_offset = 8 + 8 + 8 + 4 + 4 + 32;
        let factor_offset = factor_base_offset + column_index * 8;

        pool.install(|| {
            let batch_results: Vec<Vec<(i64, String, f64)>> = (0..num_batches)
                .into_par_iter()
                .map(|batch_idx| {
                    let start_idx = batch_idx * BATCH_SIZE;
                    let end_idx = std::cmp::min(start_idx + BATCH_SIZE, record_count);
                    let mut batch_data = Vec::with_capacity(end_idx - start_idx);

                    for i in start_idx..end_idx {
                        let record_offset = records_start + i * record_size;
                        let date = unsafe {
                            let date_ptr = mmap.as_ptr().add(record_offset) as *const i64;
                            *date_ptr
                        };
                        let code_len = unsafe {
                            let code_len_ptr = mmap.as_ptr().add(record_offset + code_len_offset) as *const u32;
                            std::cmp::min(*code_len_ptr as usize, 32)
                        };
                        let code = unsafe {
                            let code_bytes_ptr = mmap.as_ptr().add(record_offset + code_bytes_offset);
                            let code_slice = std::slice::from_raw_parts(code_bytes_ptr, code_len);
                            String::from_utf8_lossy(code_slice).into_owned()
                        };
                        let factor = unsafe {
                            let factor_ptr = mmap.as_ptr().add(record_offset + factor_offset) as *const f64;
                            *factor_ptr
                        };
                        batch_data.push((date, code, factor));
                    }
                    batch_data
                })
                .collect();

            for batch in batch_results {
                for (date, code, factor) in batch {
                    dates.push(date);
                    codes.push(code);
                    factors.push(factor);
                }
            }
        });
    }

    drop(mmap);

    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("date", numpy.call_method1("array", (dates,))?)?;
        dict.set_item("code", numpy.call_method1("array", (codes,))?)?;
        dict.set_item("factor", numpy.call_method1("array", (factors,))?)?;
        Ok(dict.into())
    })
}

// ==================== v2 → v3 原地转换 ====================

/// 从原始字节计算 v3 校验和（避免依赖 backup_writer 模块）
fn calc_v3_checksum_raw(
    date: i64,
    code_bytes: &[u8], // 16 bytes
    timestamp: i64,
    factor_bytes: &[u8], // F*4 bytes
    factor_count: usize,
) -> u32 {
    let mut sum = 0u32;
    sum = sum.wrapping_add(date as u32);
    sum = sum.wrapping_add((date >> 32) as u32);
    for &b in code_bytes {
        sum = sum.wrapping_add(b as u32);
    }
    sum = sum.wrapping_add(timestamp as u32);
    sum = sum.wrapping_add((timestamp >> 32) as u32);
    for i in 0..factor_count {
        let off = i * 4;
        let bits = u32::from_le_bytes(factor_bytes[off..off + 4].try_into().unwrap());
        sum = sum.wrapping_add(bits);
    }
    sum
}

/// 将 v2 格式备份文件原地转换为 v3 格式
///
/// v3 记录大小恰好是 v2 的 50%，因此写指针永远落后于读指针，可安全原地覆写。
///
/// 参数:
/// - backup_file: v2 格式备份文件路径
/// - batch_size: 每批处理的记录数（默认 5000，约 1 GB 内存）
///
/// 返回: 成功转换的记录数
#[pyfunction]
#[pyo3(signature = (backup_file, batch_size=None))]
pub fn convert_backup_v2_to_v3_inplace(
    backup_file: String,
    batch_size: Option<usize>,
) -> PyResult<u64> {
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};

    let batch_size = batch_size.unwrap_or(5000);
    if batch_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "batch_size 必须大于 0",
        ));
    }

    // 1. 打开文件（读写模式）
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&backup_file)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {}", e))
        })?;

    let file_len = file
        .metadata()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取文件元数据: {}", e))
        })?
        .len() as usize;

    if file_len < HEADER_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "文件太小，不是有效的备份文件",
        ));
    }

    // 2. 读取并校验 v2 文件头
    let mut header_bytes = [0u8; HEADER_SIZE];
    file.read_exact(&mut header_bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取文件头: {}", e))
    })?;

    if &header_bytes[0..8] != b"RPBACKUP" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "无效的备份文件：魔数不匹配",
        ));
    }

    let version = u32::from_le_bytes(header_bytes[8..12].try_into().unwrap());
    if version != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "备份文件版本为 {}，仅支持版本 2 的原地转换",
            version
        )));
    }

    let record_count = u64::from_le_bytes(header_bytes[12..20].try_into().unwrap()) as usize;
    let v2_record_size = u32::from_le_bytes(header_bytes[20..24].try_into().unwrap()) as usize;
    let factor_count = u32::from_le_bytes(header_bytes[24..28].try_into().unwrap()) as usize;
    let v3_record_size = calculate_v3_record_size(factor_count);

    // 验证 v2 record size
    let expected_v2_record_size = calculate_record_size(factor_count);
    if v2_record_size != expected_v2_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "v2 记录大小不匹配: 文件头 {}, 计算 {}",
            v2_record_size, expected_v2_record_size
        )));
    }

    // 验证文件大小
    let expected_file_size = HEADER_SIZE + record_count * v2_record_size;
    if file_len < expected_file_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "文件被截断: 期望 {} 字节，实际 {} 字节",
            expected_file_size, file_len
        )));
    }

    let v3_total_size = HEADER_SIZE + record_count * v3_record_size;
    let gb = |n: usize| n as f64 / (1024.0 * 1024.0 * 1024.0);

    eprintln!("开始 v2 -> v3 原地转换:");
    eprintln!("  记录数: {}", record_count);
    eprintln!("  因子数: {}", factor_count);
    eprintln!("  v2 记录大小: {} 字节", v2_record_size);
    eprintln!("  v3 记录大小: {} 字节", v3_record_size);
    eprintln!("  源文件大小: {:.2} GB", gb(file_len));
    eprintln!("  目标文件大小: {:.2} GB", gb(v3_total_size));
    eprintln!("  节省空间: {:.2} GB", gb(file_len - v3_total_size));

    // 3. 分配缓冲区
    let actual_batch = std::cmp::min(batch_size, record_count);
    let read_buf_size = actual_batch * v2_record_size;
    let write_buf_size = actual_batch * v3_record_size;

    eprintln!(
        "  批次大小: {} 条/批 (读缓冲 {:.0} MB, 写缓冲 {:.0} MB)",
        actual_batch,
        read_buf_size as f64 / (1024.0 * 1024.0),
        write_buf_size as f64 / (1024.0 * 1024.0),
    );

    let mut read_buf = vec![0u8; read_buf_size];
    let mut write_buf = vec![0u8; write_buf_size];

    let mut converted_count: usize = 0;
    let mut skipped_count: usize = 0;
    let start_time = std::time::Instant::now();

    // 4. 循环处理（每次 batch 条记录）
    let num_batches = (record_count + actual_batch - 1) / actual_batch;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * actual_batch;
        let batch_end = std::cmp::min(batch_start + actual_batch, record_count);
        let this_batch = batch_end - batch_start;

        // a. 从文件读取 this_batch 条 v2 记录到 read_buf
        let read_file_pos = HEADER_SIZE + batch_start * v2_record_size;
        file.seek(SeekFrom::Start(read_file_pos as u64))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("定位失败: {}", e))
            })?;
        file.read_exact(&mut read_buf[..this_batch * v2_record_size])
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取失败: {}", e))
            })?;

        // b. 逐条解析 v2 记录 → 序列化为 v3 格式到 write_buf
        let mut write_used = 0usize;
        let mut batch_skipped = 0usize;

        for i in 0..this_batch {
            let v2_off = i * v2_record_size;
            let v2 = &read_buf[v2_off..v2_off + v2_record_size];

            // 解析 v2 关键字段
            let date = i64::from_le_bytes(v2[0..8].try_into().unwrap());
            let timestamp = i64::from_le_bytes(v2[16..24].try_into().unwrap());
            let code_len = std::cmp::min(
                u32::from_le_bytes(v2[28..32].try_into().unwrap()) as usize,
                32,
            );

            // v3 的 code 字段只有 16 字节（15 字节 + null），超过则跳过
            if code_len > 15 {
                eprintln!(
                    "WARNING: 跳过记录 #{} (date={}, code_len={}), code 超过 15 字节限制",
                    batch_start + i,
                    date,
                    code_len
                );
                batch_skipped += 1;
                continue;
            }

            // c. 序列化为 v3 格式
            let v3 = &mut write_buf[write_used..write_used + v3_record_size];

            // date (0..8)
            v3[0..8].copy_from_slice(&date.to_le_bytes());

            // code (8..24), null-terminated, zero-padded
            v3[8..24].fill(0);
            v3[8..8 + code_len].copy_from_slice(&v2[32..32 + code_len]);

            // timestamp (24..32)
            v3[24..32].copy_from_slice(&timestamp.to_le_bytes());

            // factors f64→f32 (32..32+F*4)
            for j in 0..factor_count {
                let f64_src = 64 + j * 8;
                let f64_val = f64::from_le_bytes(v2[f64_src..f64_src + 8].try_into().unwrap());
                let f32_val = f64_val as f32;
                let f32_dst = 32 + j * 4;
                v3[f32_dst..f32_dst + 4].copy_from_slice(&f32_val.to_le_bytes());
            }

            // checksum (32+F*4..32+F*4+4)
            let ck_off = 32 + factor_count * 4;
            let cksum = calc_v3_checksum_raw(
                date,
                &v3[8..24],
                timestamp,
                &v3[32..32 + factor_count * 4],
                factor_count,
            );
            v3[ck_off..ck_off + 4].copy_from_slice(&cksum.to_le_bytes());

            write_used += v3_record_size;
        }

        // d. 将 write_buf 写回文件对应位置
        if write_used > 0 {
            let write_file_pos = HEADER_SIZE + converted_count * v3_record_size;
            file.seek(SeekFrom::Start(write_file_pos as u64))
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入定位失败: {}", e))
                })?;
            file.write_all(&write_buf[..write_used])
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入失败: {}", e))
                })?;
        }

        converted_count += this_batch - batch_skipped;
        skipped_count += batch_skipped;

        // 进度信息（每 10 个 batch 或最后一个 batch 打印一次）
        if batch_idx % 10 == 0 || batch_idx == num_batches - 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (batch_end as f64 / record_count as f64) * 100.0;
            let speed = if elapsed > 0.0 {
                (batch_end as f64 / elapsed).round() as usize
            } else {
                0
            };
            let remaining_records = record_count - batch_end;
            let remaining_secs = if speed > 0 {
                remaining_records as f64 / speed as f64
            } else {
                0.0
            };
            eprintln!(
                "  [{:.1}%] 已处理 {}/{} | {} 条/秒 | 剩余 ~{:.0}s",
                progress, batch_end, record_count, speed, remaining_secs
            );
        }
    }

    // 确保所有数据已刷盘
    file.flush().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("刷盘失败: {}", e))
    })?;

    // 5. 写入 v3 文件头（覆盖 v2 头部）
    let mut v3_header = [0u8; HEADER_SIZE];
    v3_header[0..8].copy_from_slice(b"RPBACKUP");
    v3_header[8..12].copy_from_slice(&3u32.to_le_bytes());
    v3_header[12..20].copy_from_slice(&(converted_count as u64).to_le_bytes());
    v3_header[20..24].copy_from_slice(&(v3_record_size as u32).to_le_bytes());
    v3_header[24..28].copy_from_slice(&(factor_count as u32).to_le_bytes());
    v3_header[28..32].copy_from_slice(&16u32.to_le_bytes()); // code_size

    file.seek(SeekFrom::Start(0)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("文件头定位失败: {}", e))
    })?;
    file.write_all(&v3_header).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("文件头写入失败: {}", e))
    })?;
    file.flush().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("文件头刷盘失败: {}", e))
    })?;

    // 6. 截断文件到 v3 大小
    let final_size = HEADER_SIZE + converted_count * v3_record_size;
    file.set_len(final_size as u64).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("截断文件失败: {}", e))
    })?;

    let total_elapsed = start_time.elapsed().as_secs_f64();
    eprintln!("转换完成!");
    eprintln!("  转换记录数: {} / {}", converted_count, record_count);
    if skipped_count > 0 {
        eprintln!("  跳过记录数: {}", skipped_count);
    }
    eprintln!(
        "  总耗时: {:.1}s ({:.1} 分钟)",
        total_elapsed,
        total_elapsed / 60.0
    );
    eprintln!("  最终文件大小: {:.2} GB", gb(final_size));

    Ok(converted_count as u64)
}

/// 将 v3 格式备份文件转换为 v4 分块压缩格式（新文件输出）
///
/// v3 与 v4 的记录格式完全一致，转换过程无需逐条解析字段，
/// 直接将 batch_size 条 v3 原始字节整体 zstd 压缩后写入一个 chunk。
///
/// 参数:
/// - backup_file: 源 v3 格式备份文件路径
/// - output_file: 目标 v4 格式文件路径（新文件）
/// - batch_size: 每个 chunk 包含的记录数（默认 5000）
///
/// 返回: 成功转换的记录数
#[pyfunction]
#[pyo3(signature = (backup_file, output_file, batch_size=None))]
pub fn convert_backup_v3_to_v4(
    backup_file: String,
    output_file: String,
    batch_size: Option<usize>,
) -> PyResult<u64> {
    use std::fs::OpenOptions;
    use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

    let batch_size = batch_size.unwrap_or(5000);
    if batch_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "batch_size 必须大于 0",
        ));
    }

    // 1. 打开 v3 源文件
    let src_file = File::open(&backup_file).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开源文件: {}", e))
    })?;
    let src_len = src_file.metadata().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法获取源文件元数据: {}", e))
    })?.len() as usize;

    if src_len < HEADER_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "文件太小，不是有效的备份文件",
        ));
    }

    // 2. 读取并校验 v3 文件头
    let mut src = BufReader::new(src_file);
    let mut header_bytes = [0u8; HEADER_SIZE];
    src.read_exact(&mut header_bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法读取文件头: {}", e))
    })?;

    if &header_bytes[0..8] != b"RPBACKUP" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "无效的备份文件：魔数不匹配",
        ));
    }

    let version = u32::from_le_bytes(header_bytes[8..12].try_into().unwrap());
    if version != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "源文件版本为 {}，仅支持 v3 格式转换",
            version
        )));
    }

    let record_count = u64::from_le_bytes(header_bytes[12..20].try_into().unwrap()) as usize;
    let record_size = u32::from_le_bytes(header_bytes[20..24].try_into().unwrap()) as usize;
    let factor_count = u32::from_le_bytes(header_bytes[24..28].try_into().unwrap()) as usize;

    // 验证 record_size
    let expected_record_size = calculate_v3_record_size(factor_count);
    if record_size != expected_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "v3 记录大小不匹配: 文件头 {}, 计算 {}",
            record_size, expected_record_size
        )));
    }

    // 验证文件大小
    let expected_file_size = HEADER_SIZE + record_count * record_size;
    if src_len < expected_file_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "文件被截断: 期望 {} 字节，实际 {} 字节",
            expected_file_size, src_len
        )));
    }

    let gb = |n: usize| n as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("开始 v3 -> v4 转换:");
    eprintln!("  记录数: {}", record_count);
    eprintln!("  因子数: {}", factor_count);
    eprintln!("  记录大小: {} 字节", record_size);
    eprintln!("  源文件大小: {:.2} GB", gb(src_len));
    eprintln!("  batch_size: {} 条/chunk", batch_size);

    // 3. 创建输出文件，写入占位 v4 头部
    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output_file)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法创建输出文件: {}", e))
        })?;
    let mut out = BufWriter::new(out_file);

    // 写入占位 header（后续回写）
    let mut v4_header = [0u8; HEADER_SIZE];
    v4_header[0..8].copy_from_slice(b"RPBACKUP");
    v4_header[8..12].copy_from_slice(&4u32.to_le_bytes()); // version = 4
    v4_header[12..20].copy_from_slice(&(record_count as u64).to_le_bytes());
    v4_header[20..24].copy_from_slice(&(record_size as u32).to_le_bytes());
    v4_header[24..28].copy_from_slice(&(factor_count as u32).to_le_bytes());
    // chunk_count 暂时为 0，后续回写
    out.write_all(&v4_header).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入头部失败: {}", e))
    })?;

    // 4. 分批循环：读取 → zstd 压缩 → 写入 chunk
    let actual_batch = std::cmp::min(batch_size, record_count);
    let mut read_buf = vec![0u8; actual_batch * record_size];
    let mut chunk_count: u32 = 0;
    let mut total_converted: usize = 0;
    let start_time = std::time::Instant::now();
    let num_batches = (record_count + actual_batch - 1) / actual_batch;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * actual_batch;
        let batch_end = std::cmp::min(batch_start + actual_batch, record_count);
        let this_batch = batch_end - batch_start;
        let read_size = this_batch * record_size;

        // a. 从源文件读取 this_batch 条 v3 原始记录字节
        src.read_exact(&mut read_buf[..read_size]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取源文件失败: {}", e))
        })?;

        // b. zstd 压缩（level 9）
        let compressed = zstd::encode_all(&read_buf[..read_size], 9).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("zstd 压缩失败: {}", e))
        })?;

        // c. 写入 chunk: [u32 compressed_size][u32 record_count][compressed_data]
        out.write_all(&(compressed.len() as u32).to_le_bytes()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入 chunk 头失败: {}", e))
        })?;
        out.write_all(&(this_batch as u32).to_le_bytes()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入 chunk 记录数失败: {}", e))
        })?;
        out.write_all(&compressed).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入 chunk 数据失败: {}", e))
        })?;

        chunk_count += 1;
        total_converted += this_batch;

        // 打印进度
        if batch_idx % 10 == 0 || batch_idx == num_batches - 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (batch_end as f64 / record_count as f64) * 100.0;
            let speed = if elapsed > 0.0 {
                (batch_end as f64 / elapsed).round() as usize
            } else {
                0
            };
            let remaining_records = record_count - batch_end;
            let remaining_secs = if speed > 0 {
                remaining_records as f64 / speed as f64
            } else {
                0.0
            };
            eprintln!(
                "  [{:.1}%] 已处理 {}/{} | {} 条/秒 | 剩余 ~{:.0}s",
                progress, batch_end, record_count, speed, remaining_secs
            );
        }
    }

    out.flush().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("输出文件刷盘失败: {}", e))
    })?;

    // 5. 回写 v4 文件头的 chunk_count
    let out_file = out.into_inner().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("BufWriter 刷盘失败: {}", e))
    })?;
    let mut out_file = out_file;
    out_file.seek(SeekFrom::Start(28)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("定位 chunk_count 失败: {}", e))
    })?;
    out_file.write_all(&chunk_count.to_le_bytes()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入 chunk_count 失败: {}", e))
    })?;
    out_file.flush().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("文件头刷盘失败: {}", e))
    })?;

    // 6. 打印完成摘要
    let out_len = out_file.metadata().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("获取输出文件元数据失败: {}", e))
    })?.len() as usize;

    let total_elapsed = start_time.elapsed().as_secs_f64();
    let compression_ratio = src_len as f64 / out_len as f64;
    eprintln!("转换完成!");
    eprintln!("  转换记录数: {}", total_converted);
    eprintln!("  chunk 数: {}", chunk_count);
    eprintln!(
        "  总耗时: {:.1}s ({:.1} 分钟)",
        total_elapsed,
        total_elapsed / 60.0
    );
    eprintln!("  源文件大小: {:.2} GB", gb(src_len));
    eprintln!("  目标文件大小: {:.2} GB", gb(out_len));
    eprintln!("  压缩率: {:.2}x (节省 {:.1}%)", compression_ratio, (1.0 - out_len as f64 / src_len as f64) * 100.0);

    Ok(total_converted as u64)
}
