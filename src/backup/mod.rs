use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::parallel::ComputeResult;

/// 数据集信息结构体
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub num_rows: usize,
    pub num_cols: usize,
    pub estimated_memory_gb: f64,
    pub min_date: Option<i32>,
    pub max_date: Option<i32>,
    pub file_size_bytes: u64,
}

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
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法打开文件 {}: {}", self.file_path, e)
            ))?;
        
        let mut writer = BufWriter::new(file);

        for result in results {
            let serialized = bincode::serialize(result)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("二进制序列化失败: {}", e)
                ))?;

            let len = serialized.len() as u32;
            writer.write_all(&len.to_le_bytes())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("写入长度失败: {}", e)
                ))?;
            
            writer.write_all(&serialized)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("二进制写入文件失败: {}", e)
                ))?;
        }

        writer.flush()
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

    /// 分块查询结果（增强版：支持大文件读取）
    pub fn query_results_chunked(
        &self,
        date_range: Option<(i32, i32)>,
        codes: Option<Vec<String>>,
        row_range: Option<(usize, usize)>,
        max_rows: Option<usize>,
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

        // 应用日期和代码过滤
        let mut filtered_results: Vec<ComputeResult> = all_results.into_iter()
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

        // 应用行范围限制
        if let Some((start_row, end_row)) = row_range {
            let total_rows = filtered_results.len();
            if start_row >= total_rows {
                return Ok(Vec::new());
            }
            
            let actual_end = std::cmp::min(end_row, total_rows);
            if start_row < actual_end {
                filtered_results = filtered_results.into_iter()
                    .skip(start_row)
                    .take(actual_end - start_row)
                    .collect();
            } else {
                return Ok(Vec::new());
            }
        }

        // 应用最大行数限制
        if let Some(max_rows_limit) = max_rows {
            if filtered_results.len() > max_rows_limit {
                filtered_results.truncate(max_rows_limit);
            }
        }

        Ok(filtered_results)
    }

    /// 获取数据集基本信息（不加载全部数据）
    pub fn get_dataset_info(&self) -> PyResult<DatasetInfo> {
        if !Path::new(&self.file_path).exists() {
            return Ok(DatasetInfo {
                num_rows: 0,
                num_cols: 0,
                estimated_memory_gb: 0.0,
                min_date: None,
                max_date: None,
                file_size_bytes: 0,
            });
        }

        // 获取文件大小
        let file_size = std::fs::metadata(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法读取文件信息: {}", e)
            ))?
            .len();

        // 根据存储格式快速获取基本信息
        match self.storage_format.as_str() {
            "binary" => self.get_binary_dataset_info(file_size),
            _ => {
                // 对于其他格式，加载全部数据来获取信息（暂时）
                let all_results = self.query_results(None, None)?;
                let num_rows = all_results.len();
                let num_cols = if num_rows > 0 {
                    3 + all_results.iter().map(|r| r.facs.len()).max().unwrap_or(0)
                } else {
                    0
                };

                let (min_date, max_date) = if num_rows > 0 {
                    let dates: Vec<i32> = all_results.iter().map(|r| r.date).collect();
                    (dates.iter().min().copied(), dates.iter().max().copied())
                } else {
                    (None, None)
                };

                let estimated_memory_gb = (num_rows * num_cols * 28) as f64 / (1024.0 * 1024.0 * 1024.0);

                Ok(DatasetInfo {
                    num_rows,
                    num_cols,
                    estimated_memory_gb,
                    min_date,
                    max_date,
                    file_size_bytes: file_size,
                })
            }
        }
    }

    /// 快速获取二进制文件的数据集信息
    fn get_binary_dataset_info(&self, file_size: u64) -> PyResult<DatasetInfo> {
        // 尝试读取前几个记录来推断结构
        let sample_results = self.load_binary_sample(100)?; // 采样前100条记录
        
        if sample_results.is_empty() {
            return Ok(DatasetInfo {
                num_rows: 0,
                num_cols: 0,
                estimated_memory_gb: 0.0,
                min_date: None,
                max_date: None,
                file_size_bytes: file_size,
            });
        }

        // 从采样数据推断总体结构
        let sample_size = sample_results.len();
        let num_cols = 3 + sample_results.iter().map(|r| r.facs.len()).max().unwrap_or(0);
        
        // 估算总行数（基于采样的平均记录大小）
        let sample_bytes = self.estimate_sample_size_bytes(sample_size)?;
        let avg_bytes_per_record = if sample_size > 0 { sample_bytes / sample_size } else { 100 };
        let estimated_num_rows = (file_size as usize) / avg_bytes_per_record;

        // 获取日期范围
        let (min_date, max_date) = if !sample_results.is_empty() {
            let dates: Vec<i32> = sample_results.iter().map(|r| r.date).collect();
            (dates.iter().min().copied(), dates.iter().max().copied())
        } else {
            (None, None)
        };

        let estimated_memory_gb = (estimated_num_rows * num_cols * 28) as f64 / (1024.0 * 1024.0 * 1024.0);

        Ok(DatasetInfo {
            num_rows: estimated_num_rows,
            num_cols,
            estimated_memory_gb,
            min_date,
            max_date,
            file_size_bytes: file_size,
        })
    }

    /// 加载二进制文件的采样数据
    fn load_binary_sample(&self, max_records: usize) -> PyResult<Vec<ComputeResult>> {
        if !Path::new(&self.file_path).exists() {
            return Ok(Vec::new());
        }

        let data = std::fs::read(&self.file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("二进制读取文件失败: {}", e)
            ))?;

        let mut results = Vec::new();
        let mut offset = 0;

        while offset < data.len() && results.len() < max_records {
            if offset + 4 > data.len() {
                break;
            }

            let len_bytes = &data[offset..offset + 4];
            let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
            offset += 4;

            if len == 0 || len > 100_000_000 || offset + len > data.len() {
                break;
            }

            let chunk = &data[offset..offset + len];
            match bincode::deserialize::<Vec<ComputeResult>>(chunk) {
                Ok(mut batch_results) => {
                    for result in batch_results.drain(..) {
                        results.push(result);
                        if results.len() >= max_records {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
            offset += len;
        }

        Ok(results)
    }

    /// 估算采样数据的字节大小
    fn estimate_sample_size_bytes(&self, sample_size: usize) -> PyResult<usize> {
        // 这是一个简化的估算，基于已知的二进制格式结构
        // 实际实现中可能需要更精确的计算
        Ok(sample_size * 150) // 假设每条记录平均150字节
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

    /// 原始二进制数据加载（增强容错版本）
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
        let mut corrupted_blocks = 0;

        while offset < data.len() {
            if offset + 4 > data.len() {
                eprintln!("警告: 文件末尾数据不足4字节，跳过剩余 {} 字节", data.len() - offset);
                break;
            }

            let len_bytes = &data[offset..offset + 4];
            let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
            offset += 4;

            if len == 0 {
                continue;
            }
            
            if offset + len > data.len() {
                eprintln!("警告: 数据块长度 {} 超出文件范围，文件可能已损坏。停止解析。", len);
                break;
            }

            let chunk = &data[offset..offset + len];

            // 尝试将块反序列化为 Vec<ComputeResult> (旧的批处理格式)
            if let Ok(batch_results) = bincode::deserialize::<Vec<ComputeResult>>(chunk) {
                results.extend(batch_results);
            } 
            // 如果失败，尝试反序列化为单个 ComputeResult (新的逐条记录格式)
            else if let Ok(record) = bincode::deserialize::<ComputeResult>(chunk) {
                results.push(record);
            } 
            // 如果两种方式都失败，则块已损坏
            else {
                corrupted_blocks += 1;
                eprintln!("警告: 无法在偏移量 {} 处反序列化长度为 {} 的数据块", offset, len);
            }
            
            offset += len;
        }

        if corrupted_blocks > 0 {
            eprintln!("解析完成，有 {} 个损坏的数据块被跳过。", corrupted_blocks);
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

/// 获取备份文件详细信息（公共函数）
/// 返回：(行数, 列数, 估算内存GB, 最早日期, 最新日期)
pub fn get_backup_info_detailed(
    backup_file: &str,
    storage_format: &str,
) -> PyResult<(usize, usize, f64, String, String)> {
    let backup_manager = BackupManager::new(backup_file, storage_format)?;
    let dataset_info = backup_manager.get_dataset_info()?;
    
    let min_date_str = dataset_info.min_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "N/A".to_string());
    
    let max_date_str = dataset_info.max_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "N/A".to_string());
    
    Ok((
        dataset_info.num_rows,
        dataset_info.num_cols,
        dataset_info.estimated_memory_gb,
        min_date_str,
        max_date_str,
    ))
}