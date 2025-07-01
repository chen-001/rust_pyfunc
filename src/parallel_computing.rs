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
use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};

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

fn read_existing_backup(file_path: &str) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
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
        return read_existing_backup_legacy(file_path);
    }
    
    Ok(existing_tasks)
}

fn read_existing_backup_legacy(file_path: &str) -> Result<HashSet<(i64, String)>, Box<dyn std::error::Error>> {
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
            # 确保返回浮点数类型
            return [float(x) if x is not None else float('nan') for x in result]
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
        return read_legacy_backup_results(file_path);
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
        return read_legacy_backup_results(file_path);
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
                            chunk_data.push(record.date as f64);
                            
                            // 安全的code转换
                            let code_len = std::cmp::min(record.code_len as usize, 32);
                            let code_str = String::from_utf8_lossy(&record.code_bytes[..code_len]);
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
                    
                    // 直接复制数据到输出数组
                    chunk_data.push(record.date as f64);
                    
                    // 安全的code转换 - 防止越界访问
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code_str = unsafe {
                        std::str::from_utf8_unchecked(&record.code_bytes[..code_len])
                    };
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
    
    // 超高速转换：直接从内存映射创建numpy数组
    Python::with_gil(|py| {
        let numpy = py.import("numpy")?;
        
        // 创建numpy数组并reshape（这是最快的方式）
        let array = numpy.call_method1("array", (flat_data,))?;
        let reshaped = array.call_method1("reshape", ((record_count, num_cols),))?;
        
        Ok(reshaped.into())
    })
}

// 向后兼容的旧格式读取函数
fn read_legacy_backup_results(file_path: &str) -> PyResult<PyObject> {
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
                all_results.extend(batch);
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
                    all_results.extend(batch);
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
#[pyo3(signature = (backup_file, num_threads=None))]
pub fn query_backup_fast(backup_file: String, num_threads: Option<usize>) -> PyResult<PyObject> {
    // 使用自定义线程池而不是全局线程池
    if let Some(threads) = num_threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create thread pool: {}", e)))?;
        
        pool.install(|| read_backup_results_ultra_fast_v4(&backup_file))
    } else {
        read_backup_results_ultra_fast_v4(&backup_file)
    }
}




/// 终极版本：线程安全的并行+零分配+缓存优化
fn read_backup_results_ultra_fast_v4(file_path: &str) -> PyResult<PyObject> {
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
        return read_legacy_backup_results(file_path);
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
        return read_legacy_backup_results(file_path);
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
        .map(|i| {
            let record_offset = records_start + i * record_size;
            let record_bytes = &mmap[record_offset..record_offset + record_size];
            
            match DynamicRecord::from_bytes(record_bytes, factor_count) {
                Ok(record) => {
                    let code_len = std::cmp::min(record.code_len as usize, 32);
                    let code = String::from_utf8_lossy(&record.code_bytes[..code_len]).to_string();
                    (record.date, code, record.timestamp, record.factors)
                }
                Err(_) => {
                    // 记录损坏，返回默认值
                    (0i64, String::new(), 0i64, vec![f64::NAN; factor_count])
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
) {
    loop { // 循环以支持worker重启
        if restart_flag.compare_exchange(true, false, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
            println!("🔄 Worker {} 检测到重启信号，正在重启...", worker_id);
        }

        // println!("🚀 Persistent Worker {} 启动，创建持久Python进程", worker_id);
        
        let script_content = create_persistent_worker_script();
        let script_path = format!("/tmp/persistent_worker_{}.py", worker_id);
        
        // 创建worker脚本
        if let Err(e) = std::fs::write(&script_path, script_content) {
            eprintln!("❌ Worker {} 创建脚本失败: {}", worker_id, e);
            return;
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
                return;
            }
        };
        
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
                    eprintln!("❌ Worker {} 任务 #{} 序列化失败: {}", worker_id, task_count, e);
                    continue;
                }
            };
            
            // 发送任务到Python进程（带长度前缀）
            let length = packed_data.len() as u32;
            let length_bytes = length.to_le_bytes();
            
            if let Err(e) = stdin.write_all(&length_bytes) {
                eprintln!("❌ Worker {} 发送长度前缀失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            if let Err(e) = stdin.write_all(&packed_data) {
                eprintln!("❌ Worker {} 发送任务数据失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            if let Err(e) = stdin.flush() {
                eprintln!("❌ Worker {} flush失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            // 读取结果（带长度前缀）
            let mut length_bytes = [0u8; 4];
            if let Err(e) = stdout.read_exact(&mut length_bytes) {
                eprintln!("❌ Worker {} 读取结果长度失败: {}", worker_id, e);
                needs_restart = true;
                break;
            }
            
            let length = u32::from_le_bytes(length_bytes) as usize;
            let mut result_data = vec![0u8; length];
            
            if let Err(e) = stdout.read_exact(&mut result_data) {
                eprintln!("❌ Worker {} 读取结果数据失败: {}", worker_id, e);
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
                        needs_restart = true;
                        break;
                    }
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
                        needs_restart = true;
                        break;
                    }
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
}


#[pyfunction]
#[pyo3(signature = (python_function, args, n_jobs, backup_file, expected_result_length))]
pub fn run_pools_queue(
    python_function: PyObject,
    args: &PyList,
    n_jobs: usize,
    backup_file: String,
    expected_result_length: usize,
) -> PyResult<PyObject> {
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
    
    // 读取现有备份，过滤已完成的任务
    let existing_tasks = read_existing_backup(&backup_file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read backup: {}", e)))?;
    
    let pending_tasks: Vec<TaskParam> = all_tasks
        .into_iter()
        .filter(|task| !existing_tasks.contains(&(task.date, task.code.clone())))
        .collect();
    
    if pending_tasks.is_empty() {
        // 所有任务都已完成，直接返回结果
        return read_backup_results(&backup_file);
    }
    
    let start_time = Instant::now();
    println!("📋 总任务数: {}, 待处理: {}, 已完成: {}", 
             pending_tasks.len() + existing_tasks.len(), pending_tasks.len(), existing_tasks.len());
    
    // 提取Python函数代码
    let python_code = extract_python_function_code(&python_function)?;
    
    // 获取Python解释器路径
    let python_path = env::var("PYTHON_PATH")
        .unwrap_or_else(|_| "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python".to_string());
    
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
    println!("🚀 启动 {} 个worker处理 {} 个任务", n_jobs, pending_tasks.len());
    
    // 启动worker线程
    let mut worker_handles = Vec::new();
    for i in 0..n_jobs {
        let worker_task_receiver = task_receiver.clone();
        let worker_python_code = python_code.clone();
        let worker_python_path = python_path.clone();
        let worker_result_sender = result_sender.clone();
        let worker_restart_flag = restart_flag.clone();
        
        let handle = thread::spawn(move || {
            run_persistent_task_worker(
                i,
                worker_task_receiver,
                worker_python_code,
                expected_result_length,
                worker_python_path,
                worker_result_sender,
                worker_restart_flag,
            );
        });
        
        worker_handles.push(handle);
    }
    
    // 关闭主线程的result_sender
    drop(result_sender);
    
    // 启动结果收集器
    let backup_file_clone = backup_file.clone();
    let expected_result_length_clone = expected_result_length;
    let pending_tasks_len = pending_tasks.len();
    let collector_restart_flag = restart_flag.clone();
    let collector_handle = thread::spawn(move || {
        let mut batch_results = Vec::new();
        let mut total_collected = 0;
        let mut batch_count = 0;
        let mut batch_count_this_chunk = 0;
        let total_batches = (pending_tasks_len + 999) / 1000;
        
        println!("🔄 结果收集器启动，等待worker结果...");
        
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

                print!("\r💾 第 {}/{} 次备份，存{}个结果。已用{}小时{}分钟{}秒，预余{}小时{}分钟{}秒", 
                       batch_count, total_batches, batch_results.len(),
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

                if batch_count_this_chunk >= 500 {
                    println!("\n🔄 达到500次备份，触发 workers 重启...");
                    collector_restart_flag.store(true, Ordering::SeqCst);
                    batch_count_this_chunk = 0;
                }
            }
        }
        
        // 保存剩余结果
        if !batch_results.is_empty() {
            batch_count += 1;
            println!("💾 保存最终剩余结果: {} 个", batch_results.len());
            
            match save_results_to_backup(&batch_results, &backup_file_clone, expected_result_length_clone) {
                Ok(()) => {
                    println!("✅ 最终备份成功！");
                }
                Err(e) => {
                    eprintln!("❌ 最终备份失败: {}", e);
                }
            }
        }
        
        println!("📊 收集器统计: 总收集{}个结果，进行了{}次备份", total_collected, batch_count);
    });
    
    // 等待所有worker完成
    println!("⏳ 等待所有worker完成...");
    for (i, handle) in worker_handles.into_iter().enumerate() {
        match handle.join() {
            Ok(()) => {},
            Err(e) => eprintln!("❌ Worker {} 异常: {:?}", i, e),
        }
    }
    
    // 等待收集器完成
    println!("⏳ 等待结果收集器完成...");
    match collector_handle.join() {
        Ok(()) => println!("✅ 结果收集器已完成"),
        Err(e) => eprintln!("❌ 结果收集器异常: {:?}", e),
    }
    
    // 读取并返回最终结果
    println!("📖 读取最终备份结果...");
    read_backup_results(&backup_file)
}