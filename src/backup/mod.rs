use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::parallel::ComputeResult;

/// 备份数据结构
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackupData {
    pub date: i32,
    pub code: String,
    pub timestamp: i64,
    pub facs: Vec<f64>,
}

impl From<ComputeResult> for BackupData {
    fn from(result: ComputeResult) -> Self {
        Self {
            date: result.date,
            code: result.code,
            timestamp: result.timestamp,
            facs: result.facs,
        }
    }
}

impl From<BackupData> for ComputeResult {
    fn from(data: BackupData) -> Self {
        Self {
            date: data.date,
            code: data.code,
            timestamp: data.timestamp,
            facs: data.facs,
        }
    }
}

/// 备份管理器
#[derive(Clone)]
pub struct BackupManager {
    file_path: String,
    storage_format: String,
}

impl BackupManager {
    pub fn new(file_path: &str, storage_format: &str) -> PyResult<Self> {
        Ok(Self {
            file_path: file_path.to_string(),
            storage_format: storage_format.to_string(),
        })
    }

    /// 保存一批数据
    pub fn save_batch(&mut self, results: &[ComputeResult]) -> PyResult<()> {
        match self.storage_format.as_str() {
            "json" => self.save_batch_json(results),
            "binary" => self.save_batch_binary(results),
            "memory_map" => self.save_batch_memory_map(results),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("不支持的存储格式: {}，支持的格式: json, binary, memory_map", self.storage_format)
            )),
        }
    }

    /// JSON格式保存
    fn save_batch_json(&mut self, results: &[ComputeResult]) -> PyResult<()> {
        let backup_data: Vec<BackupData> = results.iter().map(|r| r.clone().into()).collect();
        
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法打开文件 {}: {}", self.file_path, e)
            ))?;
        
        let mut writer = BufWriter::new(file);
        
        for data in backup_data {
            let json_line = serde_json::to_string(&data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("JSON序列化失败: {}", e)
                ))?;
            writeln!(writer, "{}", json_line)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("写入文件失败: {}", e)
                ))?;
        }
        
        writer.flush()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("刷新缓冲区失败: {}", e)
            ))?;
        
        Ok(())
    }

    /// 高性能二进制格式保存（追加模式）
    fn save_batch_binary(&mut self, results: &[ComputeResult]) -> PyResult<()> {
        // 直接序列化当前批次数据
        let serialized = bincode::serialize(results)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("二进制序列化失败: {}", e)
            ))?;

        // 追加到文件末尾
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法打开文件 {}: {}", self.file_path, e)
            ))?;

        // 写入数据长度（4字节）然后写入数据
        let len = serialized.len() as u32;
        file.write_all(&len.to_le_bytes())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("写入长度失败: {}", e)
            ))?;
        
        file.write_all(&serialized)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("二进制写入文件失败: {}", e)
            ))?;

        file.flush()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("刷新缓冲区失败: {}", e)
            ))?;

        Ok(())
    }

    /// 内存映射格式保存（使用与二进制相同的追加模式）
    fn save_batch_memory_map(&mut self, results: &[ComputeResult]) -> PyResult<()> {
        // 与二进制存储使用相同的高效追加方法
        self.save_batch_binary(results)
    }


    /// 加载已有结果
    pub fn load_existing_results(&self, requested_args: &[(i32, String)]) -> PyResult<Vec<ComputeResult>> {
        if !Path::new(&self.file_path).exists() {
            return Ok(Vec::new());
        }

        match self.storage_format.as_str() {
            "json" => self.load_existing_json(requested_args),
            "binary" => self.load_existing_binary(requested_args),
            "memory_map" => self.load_existing_memory_map(requested_args),
            _ => Ok(Vec::new()),
        }
    }

    /// 从JSON文件加载
    fn load_existing_json(&self, requested_args: &[(i32, String)]) -> PyResult<Vec<ComputeResult>> {
        let file = File::open(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法打开文件 {}: {}", self.file_path, e)
            ))?;
        
        let reader = BufReader::new(file);
        let mut existing_results = Vec::new();
        let requested_set: std::collections::HashSet<(i32, String)> = 
            requested_args.iter().cloned().collect();
        
        for line in std::io::BufRead::lines(reader) {
            let line = line.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("读取文件行失败: {}", e)
            ))?;
            
            if let Ok(backup_data) = serde_json::from_str::<BackupData>(&line) {
                let key = (backup_data.date, backup_data.code.clone());
                if requested_set.contains(&key) {
                    existing_results.push(backup_data.into());
                }
            }
        }
        
        Ok(existing_results)
    }

    /// 从二进制文件加载
    fn load_existing_binary(&self, requested_args: &[(i32, String)]) -> PyResult<Vec<ComputeResult>> {
        let all_data = self.load_all_binary_raw()?;
        let requested_set: std::collections::HashSet<(i32, String)> = 
            requested_args.iter().cloned().collect();
        
        let filtered_results = all_data.into_iter()
            .filter(|result| requested_set.contains(&(result.date, result.code.clone())))
            .collect();

        Ok(filtered_results)
    }

    /// 从内存映射文件加载
    fn load_existing_memory_map(&self, requested_args: &[(i32, String)]) -> PyResult<Vec<ComputeResult>> {
        let all_data = self.load_all_memory_map_raw()?;
        let requested_set: std::collections::HashSet<(i32, String)> = 
            requested_args.iter().cloned().collect();
        
        let filtered_results = all_data.into_iter()
            .filter(|result| requested_set.contains(&(result.date, result.code.clone())))
            .collect();

        Ok(filtered_results)
    }


    /// 查询结果
    pub fn query_results(
        &self,
        date_range: Option<(i32, i32)>,
        codes: Option<Vec<String>>,
    ) -> PyResult<Vec<ComputeResult>> {
        if !Path::new(&self.file_path).exists() {
            return Ok(Vec::new());
        }

        let all_results = match self.storage_format.as_str() {
            "json" => self.load_all_json()?,
            "binary" => self.load_all_binary()?,
            "memory_map" => self.load_all_memory_map()?,
            _ => return Ok(Vec::new()),
        };

        let filtered_results = all_results.into_iter()
            .filter(|result| {
                // 日期范围过滤
                if let Some((start_date, end_date)) = date_range {
                    if result.date < start_date || result.date > end_date {
                        return false;
                    }
                }
                
                // 代码过滤
                if let Some(ref code_list) = codes {
                    if !code_list.contains(&result.code) {
                        return false;
                    }
                }
                
                true
            })
            .collect();

        Ok(filtered_results)
    }

    /// 加载所有JSON数据
    fn load_all_json(&self) -> PyResult<Vec<ComputeResult>> {
        let file = File::open(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法打开文件 {}: {}", self.file_path, e)
            ))?;
        
        let reader = BufReader::new(file);
        let mut results = Vec::new();
        
        for line in std::io::BufRead::lines(reader) {
            let line = line.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("读取文件行失败: {}", e)
            ))?;
            
            if let Ok(backup_data) = serde_json::from_str::<BackupData>(&line) {
                results.push(backup_data.into());
            }
        }
        
        Ok(results)
    }

    /// 加载所有二进制数据
    fn load_all_binary(&self) -> PyResult<Vec<ComputeResult>> {
        self.load_all_binary_raw()
    }

    /// 原始二进制数据加载（支持追加格式）
    fn load_all_binary_raw(&self) -> PyResult<Vec<ComputeResult>> {
        if !Path::new(&self.file_path).exists() {
            return Ok(Vec::new());
        }

        let data = std::fs::read(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("二进制读取文件失败: {}", e)
            ))?;

        let mut results = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            if offset + 4 > data.len() {
                break; // 不足4字节，无法读取长度
            }

            // 读取数据长度
            let len_bytes = &data[offset..offset + 4];
            let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
            offset += 4;

            if offset + len > data.len() {
                break; // 数据长度超出文件大小
            }

            // 读取并反序列化数据
            let chunk = &data[offset..offset + len];
            match bincode::deserialize::<Vec<ComputeResult>>(chunk) {
                Ok(mut batch_results) => {
                    results.append(&mut batch_results);
                }
                Err(e) => {
                    eprintln!("反序列化批次数据失败: {}", e);
                }
            }
            offset += len;
        }

        Ok(results)
    }

    /// 加载所有内存映射数据
    fn load_all_memory_map(&self) -> PyResult<Vec<ComputeResult>> {
        self.load_all_memory_map_raw()
    }

    /// 原始内存映射数据加载（使用与二进制相同的方法）
    fn load_all_memory_map_raw(&self) -> PyResult<Vec<ComputeResult>> {
        // 与二进制加载使用相同的方法
        self.load_all_binary_raw()
    }

    /// 删除备份文件
    pub fn delete_backup(&self) -> PyResult<()> {
        if Path::new(&self.file_path).exists() {
            std::fs::remove_file(&self.file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("删除备份文件失败: {}", e)
                ))?;
        }
        Ok(())
    }

    /// 检查备份文件是否存在
    pub fn backup_exists(&self) -> bool {
        Path::new(&self.file_path).exists()
    }

    /// 获取备份文件信息
    pub fn get_backup_info(&self) -> PyResult<(u64, String)> {
        if !Path::new(&self.file_path).exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                "备份文件不存在"
            ));
        }

        let metadata = std::fs::metadata(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法读取文件信息: {}", e)
            ))?;

        let size = metadata.len();
        let modified = metadata.modified()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法读取修改时间: {}", e)
            ))?;

        let datetime_str = if let Ok(duration) = modified.duration_since(std::time::SystemTime::UNIX_EPOCH) {
            let timestamp = duration.as_secs();
            format!("{}时间戳", timestamp) // 简化版本，避免依赖chrono
        } else {
            "未知时间".to_string()
        };

        Ok((size, datetime_str))
    }
}