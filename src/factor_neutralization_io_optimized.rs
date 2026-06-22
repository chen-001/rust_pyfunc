use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray};
use chrono::Local;
use nalgebra::{DMatrix, DVector};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

/// I/O优化的风格数据结构
pub struct IOOptimizedStyleData {
    pub data_by_date: HashMap<i64, IOOptimizedStyleDayData>,
    // 预加载的文件内容缓存
    pub file_cache: Arc<Mutex<Vec<u8>>>,
}

/// I/O优化的单日风格数据
pub struct IOOptimizedStyleDayData {
    pub stocks: Vec<String>,
    pub style_matrix: DMatrix<f64>,
    pub regression_matrix: Option<Arc<DMatrix<f64>>>,
    pub stock_index_map: HashMap<String, usize>,
}

/// I/O优化的因子数据结构 - 支持流式读取
pub struct IOOptimizedFactorData {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    pub values: DMatrix<f64>,
    pub stock_index_map: HashMap<String, usize>,
    // 文件元数据用于快速访问
    pub file_metadata: FactorFileMetadata,
}

/// 因子文件元数据
pub struct FactorFileMetadata {
    pub file_size: u64,
    pub row_count: usize,
    pub col_count: usize,
    pub has_nan_values: bool,
}

/// I/O优化的文件批量读取器
pub struct BatchFileReader {
    // 预分配的缓冲区
    buffer_pool: Vec<Vec<u8>>,
    // 当前可用缓冲区索引
    available_buffers: Vec<usize>,
}

impl BatchFileReader {
    pub fn new(buffer_count: usize, buffer_size: usize) -> Self {
        let mut buffer_pool = Vec::with_capacity(buffer_count);
        let mut available_buffers = Vec::with_capacity(buffer_count);

        for i in 0..buffer_count {
            buffer_pool.push(vec![0u8; buffer_size]);
            available_buffers.push(i);
        }

        Self {
            buffer_pool,
            available_buffers,
        }
    }

    pub fn get_buffer(&mut self) -> Option<usize> {
        self.available_buffers.pop()
    }

    pub fn return_buffer(&mut self, index: usize) {
        if index < self.buffer_pool.len() {
            self.available_buffers.push(index);
        }
    }
}

impl IOOptimizedStyleData {
    /// I/O优化的风格数据加载 - 使用缓冲读取和预分配
    pub fn load_from_parquet_io_optimized(path: &str) -> PyResult<Self> {
        let start_time = Instant::now();
        println!("🔄 开始I/O优化版风格数据加载...");

        // 获取文件元数据以预分配内存
        let file_metadata = fs::metadata(path)
            .map_err(|e| PyRuntimeError::new_err(format!("获取文件元数据失败: {}", e)))?;
        let file_size = file_metadata.len();

        println!("📁 文件大小: {:.2}MB", file_size as f64 / 1024.0 / 1024.0);

        // 使用标准文件读取（parquet库优化的读取方式）
        let file = File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("打开风格数据文件失败: {}", e)))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| PyRuntimeError::new_err(format!("创建parquet读取器失败: {}", e)))?;

        // 优化批处理大小 - 根据文件大小动态调整
        let optimal_batch_size = if file_size > 100 * 1024 * 1024 {
            // > 100MB
            32768 // 大文件使用更大的批处理
        } else if file_size > 10 * 1024 * 1024 {
            // > 10MB
            16384 // 中等文件
        } else {
            8192 // 小文件
        };

        let reader = builder
            .with_batch_size(optimal_batch_size)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("构建记录批次读取器失败: {}", e)))?;

        // 预分配数据结构以减少内存分配开销
        let mut all_data = Vec::new();
        let mut total_rows = 0;

        // 批量读取所有数据
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| PyRuntimeError::new_err(format!("读取记录批次失败: {}", e)))?;
            total_rows += batch.num_rows();
            all_data.push(batch);
        }

        println!(
            "📊 读取完成: {}行数据, {}个批次",
            total_rows,
            all_data.len()
        );

        // 预分配HashMap以减少重新哈希
        let estimated_dates = (total_rows / 1000).max(100); // 估算日期数量
        let mut data_by_date: HashMap<i64, Vec<(String, Vec<f64>)>> =
            HashMap::with_capacity(estimated_dates);

        // 使用向量化处理优化数据解析
        let parse_start = Instant::now();
        for batch in all_data {
            Self::process_batch_vectorized(&batch, &mut data_by_date)?;
        }
        let parse_time = parse_start.elapsed();

        println!("⚡ 数据解析耗时: {:.3}s", parse_time.as_secs_f64());

        // 批量转换为最终数据结构
        let convert_start = Instant::now();
        let mut final_data_by_date = HashMap::with_capacity(data_by_date.len());
        let mut total_stocks_processed = 0;

        // 并行处理日期数据（如果日期数量足够多）
        if data_by_date.len() > 10 {
            // 大量日期时使用并行处理
            let date_results: Vec<_> = data_by_date
                .into_par_iter()
                .filter_map(|(date, stock_data)| {
                    if stock_data.len() >= 12 {
                        if let Ok(day_data) = Self::convert_date_data_optimized(date, stock_data) {
                            Some((date, day_data))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            for (date, day_data) in date_results {
                total_stocks_processed += day_data.stocks.len();
                final_data_by_date.insert(date, day_data);
            }
        } else {
            // 少量日期时使用串行处理
            for (date, stock_data) in data_by_date {
                if stock_data.len() >= 12 {
                    match Self::convert_date_data_optimized(date, stock_data) {
                        Ok(day_data) => {
                            total_stocks_processed += day_data.stocks.len();
                            final_data_by_date.insert(date, day_data);
                        }
                        Err(_) => {
                            println!("警告: 日期{}数据转换失败", date);
                        }
                    }
                }
            }
        }

        let convert_time = convert_start.elapsed();
        println!("🔄 数据转换耗时: {:.3}s", convert_time.as_secs_f64());

        if final_data_by_date.is_empty() {
            return Err(PyRuntimeError::new_err(
                "风格数据为空或所有日期的股票数量都少于12只",
            ));
        }

        let total_time = start_time.elapsed();
        println!("✅ I/O优化版风格数据加载完成!");
        println!(
            "   📊 统计: {}个交易日, {}只股票",
            final_data_by_date.len(),
            total_stocks_processed
        );
        println!("   ⏱️  总耗时: {:.3}s", total_time.as_secs_f64());
        println!(
            "   🚀 I/O速度: {:.1}MB/s",
            file_size as f64 / 1024.0 / 1024.0 / total_time.as_secs_f64()
        );

        Ok(IOOptimizedStyleData {
            data_by_date: final_data_by_date,
            file_cache: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// 向量化批处理数据解析
    fn process_batch_vectorized(
        batch: &arrow::record_batch::RecordBatch,
        data_by_date: &mut HashMap<i64, Vec<(String, Vec<f64>)>>,
    ) -> PyResult<()> {
        let date_column = batch.column(0);

        // 批量解析日期列
        let batch_dates: Vec<i64> =
            if let Some(date_array_i64) = date_column.as_any().downcast_ref::<Int64Array>() {
                (0..date_array_i64.len())
                    .map(|i| date_array_i64.value(i))
                    .collect()
            } else if let Some(date_array_i32) = date_column.as_any().downcast_ref::<Int32Array>() {
                (0..date_array_i32.len())
                    .map(|i| date_array_i32.value(i) as i64)
                    .collect()
            } else {
                return Err(PyRuntimeError::new_err(
                    "日期列类型错误：期望Int64或Int32类型",
                ));
            };

        // 支持StringArray和LargeStringArray两种类型
        let stock_column = batch.column(1);
        let get_stock_value = |row_idx: usize| -> String {
            if let Some(string_array) = stock_column.as_any().downcast_ref::<StringArray>() {
                string_array.value(row_idx).to_string()
            } else if let Some(large_string_array) =
                stock_column.as_any().downcast_ref::<LargeStringArray>()
            {
                large_string_array.value(row_idx).to_string()
            } else {
                panic!("股票代码列类型错误：期望StringArray或LargeStringArray类型");
            }
        };

        // 批量提取风格因子列取值闭包: value_0~value_9(Float64) + ind_1~ind_31(Float32) = 41列
        let style_getters: Vec<Box<dyn Fn(usize) -> f64 + Send + Sync>> = (2..43)
            .map(|i| {
                let col = batch.column(i);
                let as_any = col.as_any();
                let getter: Box<dyn Fn(usize) -> f64 + Send + Sync> = if let Some(arr) =
                    as_any.downcast_ref::<Float64Array>()
                {
                    Box::new(move |row_idx: usize| {
                        if arr.is_null(row_idx) {
                            f64::NAN
                        } else {
                            arr.value(row_idx)
                        }
                    })
                } else if let Some(arr) = as_any.downcast_ref::<Float32Array>() {
                    Box::new(move |row_idx: usize| {
                        if arr.is_null(row_idx) {
                            f64::NAN
                        } else {
                            arr.value(row_idx) as f64
                        }
                    })
                } else {
                    return Err(PyRuntimeError::new_err(format!(
                        "风格因子列{}类型错误：期望Float64或Float32",
                        i - 2
                    )));
                };
                Ok(getter)
            })
            .collect::<PyResult<Vec<_>>>()?;

        // 向量化处理每一行
        for row_idx in 0..batch.num_rows() {
            let date = batch_dates[row_idx];
            let stock = get_stock_value(row_idx);

            // 使用迭代器和collect优化风格值提取
            let style_values: Vec<f64> = style_getters.iter().map(|g| g(row_idx)).collect();

            data_by_date
                .entry(date)
                .or_insert_with(Vec::new)
                .push((stock, style_values));
        }

        Ok(())
    }

    /// 优化的单日数据转换
    fn convert_date_data_optimized(
        _date: i64,
        stock_data: Vec<(String, Vec<f64>)>,
    ) -> PyResult<IOOptimizedStyleDayData> {
        let n_stocks = stock_data.len();

        // 预分配所有数据结构
        let mut stocks = Vec::with_capacity(n_stocks);
        let mut stock_index_map = HashMap::with_capacity(n_stocks);
        let mut style_matrix = DMatrix::zeros(n_stocks, 41);

        // 单次遍历填充所有数据结构（41维风格: 10 barra + 31 行业哑变量, 行业和=1已含截距, 不再加常数项）
        for (i, (stock, style_values)) in stock_data.into_iter().enumerate() {
            stock_index_map.insert(stock.clone(), i);
            stocks.push(stock);

            // 直接写入矩阵（避免边界检查）
            unsafe {
                for j in 0..41 {
                    *style_matrix.get_unchecked_mut((i, j)) = style_values[j];
                }
            }
        }

        // 预计算回归矩阵
        let regression_matrix = compute_regression_matrix_io_optimized(&style_matrix)?;

        Ok(IOOptimizedStyleDayData {
            stocks,
            style_matrix,
            regression_matrix: Some(Arc::new(regression_matrix)),
            stock_index_map,
        })
    }
}

/// I/O优化的回归矩阵计算
fn compute_regression_matrix_io_optimized(style_matrix: &DMatrix<f64>) -> PyResult<DMatrix<f64>> {
    let xt = style_matrix.transpose();
    let xtx = &xt * style_matrix;

    let xtx_inv = xtx
        .try_inverse()
        .ok_or_else(|| PyRuntimeError::new_err("风格因子矩阵不可逆，可能存在多重共线性"))?;

    Ok(xtx_inv * xt)
}

/// I/O优化的因子文件加载
fn load_factor_file_io_optimized(
    file_path: &Path,
    log_detailed: bool,
) -> PyResult<IOOptimizedFactorData> {
    let start_time = Instant::now();

    // 获取文件元数据
    let file_metadata = fs::metadata(file_path)
        .map_err(|e| PyRuntimeError::new_err(format!("获取文件元数据失败: {}", e)))?;
    let file_size = file_metadata.len();

    let file = File::open(file_path).map_err(|e| {
        PyRuntimeError::new_err(format!("打开因子文件失败 {}: {}", file_path.display(), e))
    })?;

    // 准备使用I/O优化的parquet读取

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyRuntimeError::new_err(format!("创建parquet读取器失败: {}", e)))?;

    // 自适应批处理大小
    let batch_size = if file_size > 100 * 1024 * 1024 {
        32768
    } else if file_size > 10 * 1024 * 1024 {
        16384
    } else {
        8192
    };

    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("构建记录批次读取器失败: {}", e)))?;

    // 预加载所有批次
    let mut all_batches = Vec::new();
    let mut total_rows = 0;
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| PyRuntimeError::new_err(format!("读取记录批次失败: {}", e)))?;
        total_rows += batch.num_rows();
        all_batches.push(batch);
    }

    if all_batches.is_empty() {
        return Err(PyRuntimeError::new_err("因子文件为空"));
    }

    // 解析schema和列映射
    let schema = all_batches[0].schema();
    let total_columns = schema.fields().len();
    let last_field = &schema.fields()[total_columns - 1];

    let (date_col_idx, stocks) = if last_field.name() == "date" {
        let stocks: Vec<String> = schema
            .fields()
            .iter()
            .take(total_columns - 1)
            .map(|f| f.name().clone())
            .collect();
        (total_columns - 1, stocks)
    } else {
        let stocks: Vec<String> = schema
            .fields()
            .iter()
            .skip(1)
            .map(|f| f.name().clone())
            .collect();
        (0, stocks)
    };

    let n_stocks = stocks.len();

    // 预分配结果数据结构
    let mut all_data = Vec::with_capacity(total_rows);
    let mut dates = Vec::with_capacity(total_rows);
    let mut has_nan = false;

    // 创建股票索引映射
    let stock_index_map: HashMap<String, usize> = stocks
        .iter()
        .enumerate()
        .map(|(idx, stock)| (stock.clone(), idx))
        .collect();

    // 预构建列映射
    let stock_col_map: HashMap<usize, usize> = (0..n_stocks)
        .filter_map(|stock_idx| {
            schema
                .fields()
                .iter()
                .position(|f| f.name() == &stocks[stock_idx])
                .map(|col_idx| (stock_idx, col_idx))
        })
        .collect();

    // 并行处理批次数据（如果批次数量足够多）
    if all_batches.len() > 4 {
        // 使用并行处理
        let batch_results: Vec<_> = all_batches
            .into_par_iter()
            .map(|batch| {
                process_factor_batch_optimized(&batch, date_col_idx, &stock_col_map, n_stocks)
            })
            .collect();

        // 合并结果
        for result in batch_results {
            let (batch_data, batch_dates, batch_has_nan) = result?;
            all_data.extend(batch_data);
            dates.extend(batch_dates);
            has_nan = has_nan || batch_has_nan;
        }
    } else {
        // 使用串行处理
        for batch in all_batches {
            let (batch_data, batch_dates, batch_has_nan) =
                process_factor_batch_optimized(&batch, date_col_idx, &stock_col_map, n_stocks)?;
            all_data.extend(batch_data);
            dates.extend(batch_dates);
            has_nan = has_nan || batch_has_nan;
        }
    }

    // 构建最终矩阵
    let n_dates = dates.len();
    let mut values = DMatrix::zeros(n_dates, n_stocks);

    for (date_idx, row_values) in all_data.into_iter().enumerate() {
        for (stock_idx, value) in row_values.into_iter().enumerate() {
            values[(date_idx, stock_idx)] = value;
        }
    }

    let load_time = start_time.elapsed();
    let mb_per_sec = (file_size as f64 / 1024.0 / 1024.0) / load_time.as_secs_f64();

    // 根据log_detailed参数决定是否输出详细日志
    if log_detailed {
        println!(
            "✅ I/O优化因子文件加载: {}, {}行x{}列, {:.3}s, {:.1}MB/s",
            file_path.file_name().unwrap().to_string_lossy(),
            n_dates,
            n_stocks,
            load_time.as_secs_f64(),
            mb_per_sec
        );
    }

    Ok(IOOptimizedFactorData {
        dates,
        stocks,
        values,
        stock_index_map,
        file_metadata: FactorFileMetadata {
            file_size,
            row_count: n_dates,
            col_count: n_stocks,
            has_nan_values: has_nan,
        },
    })
}

/// 优化的批次处理函数
fn process_factor_batch_optimized(
    batch: &arrow::record_batch::RecordBatch,
    date_col_idx: usize,
    stock_col_map: &HashMap<usize, usize>,
    n_stocks: usize,
) -> PyResult<(Vec<Vec<f64>>, Vec<i64>, bool)> {
    let date_column = batch.column(date_col_idx);

    let batch_dates: Vec<i64> =
        if let Some(date_array_i64) = date_column.as_any().downcast_ref::<Int64Array>() {
            (0..date_array_i64.len())
                .map(|i| date_array_i64.value(i))
                .collect()
        } else if let Some(date_array_i32) = date_column.as_any().downcast_ref::<Int32Array>() {
            (0..date_array_i32.len())
                .map(|i| date_array_i32.value(i) as i64)
                .collect()
        } else {
            return Err(PyRuntimeError::new_err(
                "日期列类型错误：期望Int64或Int32类型",
            ));
        };

    let num_rows = batch.num_rows();

    // 预获取所有股票列的引用
    let mut stock_arrays: Vec<(usize, &Float64Array)> = Vec::with_capacity(stock_col_map.len());
    for (&stock_idx, &col_idx) in stock_col_map.iter() {
        let array = batch.column(col_idx);
        if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
            stock_arrays.push((stock_idx, float_array));
        }
    }

    let mut batch_data = Vec::with_capacity(num_rows);
    let mut has_nan = false;

    for row_idx in 0..num_rows {
        let mut row_values = vec![f64::NAN; n_stocks];

        // 向量化处理行数据
        for &(stock_idx, float_array) in &stock_arrays {
            if !float_array.is_null(row_idx) {
                row_values[stock_idx] = float_array.value(row_idx);
            } else {
                has_nan = true;
            }
        }

        batch_data.push(row_values);
    }

    Ok((batch_data, batch_dates, has_nan))
}

/// I/O优化的截面排序
fn cross_section_rank_io_optimized(values: &[f64]) -> Vec<f64> {
    let n = values.len();

    // 预分配索引向量
    let mut indexed_values = Vec::with_capacity(n);
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() {
            indexed_values.push((i, v));
        }
    }

    // 使用不稳定排序提高性能
    indexed_values
        .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![f64::NAN; n];

    // 批量赋值ranks
    for (rank, &(original_idx, _)) in indexed_values.iter().enumerate() {
        ranks[original_idx] = (rank + 1) as f64;
    }

    ranks
}

/// 格式化持续时间为"几小时几分钟几秒"格式
fn format_duration(total_seconds: u64) -> String {
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    if hours > 0 {
        format!("{}小时{}分钟{}秒", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}分钟{}秒", minutes, seconds)
    } else {
        format!("{}秒", seconds)
    }
}

/// I/O优化的批量因子中性化函数
#[pyfunction]
pub fn batch_factor_neutralization_io_optimized(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: Option<usize>,
    log_detailed: Option<bool>,
) -> PyResult<()> {
    let start_time = Instant::now();
    println!("🚀 开始I/O优化版批量因子中性化处理...");

    // 使用I/O优化版本加载风格数据
    println!("📖 正在使用I/O优化加载风格数据...");
    let style_data = Arc::new(IOOptimizedStyleData::load_from_parquet_io_optimized(
        style_data_path,
    )?);

    // 获取所有因子文件并按大小排序以优化处理顺序
    let factor_dir = Path::new(factor_files_dir);
    let mut factor_files_with_size: Vec<(PathBuf, u64)> = fs::read_dir(factor_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("读取因子目录失败: {}", e)))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                if let Ok(metadata) = fs::metadata(&path) {
                    Some((path, metadata.len()))
                } else {
                    Some((path, 0))
                }
            } else {
                None
            }
        })
        .collect();

    // 按文件大小排序 - 先处理大文件，后处理小文件（更好的负载平衡）
    factor_files_with_size.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let factor_files: Vec<PathBuf> = factor_files_with_size
        .into_iter()
        .map(|(path, _)| path)
        .collect();

    let total_files = factor_files.len();
    println!("📁 找到{}个因子文件（已按大小排序）", total_files);

    if total_files == 0 {
        return Err(PyRuntimeError::new_err("未找到任何parquet因子文件"));
    }

    // 创建进度计数器
    let processed_files = Arc::new(AtomicUsize::new(0));
    let error_files = Arc::new(AtomicUsize::new(0));

    // 启动进度监控线程
    let progress_counter = Arc::clone(&processed_files);
    let error_counter = Arc::clone(&error_files);
    let monitor_start_time = start_time;
    let progress_handle = thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(60));
            let processed = progress_counter.load(Ordering::Relaxed);
            let errors = error_counter.load(Ordering::Relaxed);
            let elapsed = monitor_start_time.elapsed();

            if processed >= total_files {
                break;
            }

            let success_count = processed - errors;
            let progress_percent = (processed as f64 / total_files as f64) * 100.0;
            let elapsed_minutes = elapsed.as_secs_f64() / 60.0;

            let estimated_total_minutes = if progress_percent > 0.0 {
                elapsed_minutes * 100.0 / progress_percent
            } else {
                0.0
            };
            let estimated_remaining_minutes = estimated_total_minutes - elapsed_minutes;

            // 格式化已用时间
            let elapsed_seconds = elapsed.as_secs();
            let elapsed_time_str = format_duration(elapsed_seconds);

            // 格式化预计剩余时间
            let remaining_seconds = (estimated_remaining_minutes.max(0.0) * 60.0) as u64;
            let remaining_time_str = format_duration(remaining_seconds);

            // 显示进度：有处理进展或者已经运行超过5秒
            if processed > 0 || elapsed.as_secs() >= 5 {
                let current_time = Local::now().format("%Y-%m-%d %H:%M:%S");
                print!("\r[{}] 📊 处理进度: {}/{} ({:.1}%) - 成功: {}, 失败: {} - 已用时间: {} - 预计剩余: {}", current_time, processed, total_files, progress_percent, success_count, errors, elapsed_time_str, remaining_time_str);
                io::stdout().flush().unwrap();
            }
        }
    });

    // 创建输出目录
    fs::create_dir_all(output_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("创建输出目录失败: {}", e)))?;

    // 优化线程数配置
    let optimal_threads = if let Some(threads) = num_threads {
        threads
    } else {
        // 基于系统资源和文件数量自动选择线程数
        let cpu_threads = rayon::current_num_threads();
        let memory_gb = sys_info::mem_info()
            .map(|info| info.total / 1024 / 1024)
            .unwrap_or(8);
        let memory_based_threads = (memory_gb / 2).min(16).max(1) as usize; // 每2GB内存1个线程

        std::cmp::min(
            std::cmp::min(cpu_threads, memory_based_threads),
            total_files,
        )
    };

    println!("⚡ 使用{}个线程进行I/O优化并行处理", optimal_threads);

    // 创建I/O优化的线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(optimal_threads)
        .thread_name(|index| format!("io-optimized-worker-{}", index))
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;

    // 使用I/O优化版本并行处理所有文件
    let processed_counter = Arc::clone(&processed_files);
    let error_counter = Arc::clone(&error_files);

    let results: Vec<_> = pool.install(|| {
        factor_files
            .into_par_iter()
            .map(|file_path| {
                let style_data = Arc::clone(&style_data);
                let output_dir = Path::new(output_dir);
                let processed_counter = Arc::clone(&processed_counter);
                let error_counter = Arc::clone(&error_counter);

                let file_start_time = Instant::now();
                let result = (|| -> PyResult<()> {
                    // 使用I/O优化版本加载因子数据
                    let factor_data =
                        load_factor_file_io_optimized(&file_path, log_detailed.unwrap_or(false))?;

                    // 执行中性化处理
                    let neutralized_result =
                        neutralize_single_factor_io_optimized(factor_data, &style_data)?;

                    // 构建输出文件路径
                    let output_filename = file_path
                        .file_name()
                        .ok_or_else(|| PyRuntimeError::new_err("无效的文件名"))?;
                    let output_path = output_dir.join(output_filename);

                    // 保存结果
                    save_neutralized_result_io_optimized(neutralized_result, &output_path)?;

                    Ok(())
                })();

                // 条件化详细日志输出
                if log_detailed.unwrap_or(false) {
                    let file_time = file_start_time.elapsed();
                    if let Err(e) = &result {
                        eprintln!(
                            "❌ I/O优化处理失败: {} ({:.3}s) - {}",
                            file_path.file_name().unwrap().to_string_lossy(),
                            file_time.as_secs_f64(),
                            e
                        );
                    } else {
                        println!(
                            "✅ I/O优化完成: {} ({:.3}s)",
                            file_path.file_name().unwrap().to_string_lossy(),
                            file_time.as_secs_f64()
                        );
                    }
                }

                // 更新计数器
                processed_counter.fetch_add(1, Ordering::Relaxed);
                if result.is_err() {
                    error_counter.fetch_add(1, Ordering::Relaxed);
                }

                result
            })
            .collect()
    });

    // 等待监控线程结束
    progress_handle.join().expect("进度监控线程异常结束");

    // 统计处理结果
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let error_count = results.len() - success_count;

    let total_time = start_time.elapsed();
    println!("\n🎉 I/O优化版批量因子中性化处理完成!");
    println!("{}", "=".repeat(60));
    println!("📊 处理统计:");
    println!("   总文件数: {}", total_files);
    println!(
        "   成功处理: {} ({:.1}%)",
        success_count,
        success_count as f64 / total_files as f64 * 100.0
    );
    println!("   失败文件: {}", error_count);
    println!(
        "   总用时: {:.1}分钟 ({:.1}秒)",
        total_time.as_secs_f64() / 60.0,
        total_time.as_secs_f64()
    );
    println!(
        "   平均处理速度: {:.1} 文件/分钟",
        total_files as f64 / (total_time.as_secs_f64() / 60.0)
    );
    println!(
        "   平均单文件用时: {:.3}秒",
        total_time.as_secs_f64() / total_files as f64
    );
    println!("   I/O优化效果: ⚡ 缓冲读取 + 🔄 批处理优化 + 📊 自适应配置");

    if error_count > 0 {
        println!("⚠️  警告: {}个文件处理失败，请检查错误日志", error_count);
    }

    Ok(())
}

/// I/O优化的单因子中性化
fn neutralize_single_factor_io_optimized(
    factor_data: IOOptimizedFactorData,
    style_data: &IOOptimizedStyleData,
) -> PyResult<IOOptimizedNeutralizationResult> {
    // 使用原有的中性化逻辑，但应用I/O优化的数据结构
    let n_dates = factor_data.dates.len();

    if n_dates == 0 {
        return Err(PyRuntimeError::new_err("因子数据为空：没有日期数据"));
    }

    if factor_data.stocks.is_empty() {
        return Err(PyRuntimeError::new_err("因子数据为空：没有股票数据"));
    }

    // 获取股票交集
    let mut all_stocks_set = HashSet::new();
    for day_data in style_data.data_by_date.values() {
        for stock in &day_data.stocks {
            all_stocks_set.insert(stock.clone());
        }
    }

    let factor_stocks_set: HashSet<String> = factor_data.stocks.iter().cloned().collect();
    let mut union_stocks: Vec<String> = all_stocks_set
        .intersection(&factor_stocks_set)
        .cloned()
        .collect();
    union_stocks.sort_unstable();

    let n_union_stocks = union_stocks.len();
    let mut neutralized_values = DMatrix::from_element(n_dates, n_union_stocks, f64::NAN);

    // 处理每个日期的中性化
    for (date_idx, &date) in factor_data.dates.iter().enumerate() {
        if let Some(day_data) = style_data.data_by_date.get(&date) {
            if let Ok(day_values) =
                process_single_date_io_optimized(date_idx, &factor_data, day_data, &union_stocks)
            {
                for (union_idx, value) in day_values {
                    neutralized_values[(date_idx, union_idx)] = value;
                }
            }
        }
    }

    Ok(IOOptimizedNeutralizationResult {
        dates: factor_data.dates,
        stocks: union_stocks,
        neutralized_values,
    })
}

/// I/O优化的单日处理
fn process_single_date_io_optimized(
    date_idx: usize,
    factor_data: &IOOptimizedFactorData,
    day_data: &IOOptimizedStyleDayData,
    union_stocks: &[String],
) -> PyResult<Vec<(usize, f64)>> {
    let mut daily_factor_values = Vec::new();
    let mut valid_union_indices = Vec::new();
    let mut valid_style_indices = Vec::new();

    for (union_idx, union_stock) in union_stocks.iter().enumerate() {
        if let Some(&factor_stock_idx) = factor_data.stock_index_map.get(union_stock) {
            if let Some(&style_stock_idx) = day_data.stock_index_map.get(union_stock) {
                let value = factor_data.values[(date_idx, factor_stock_idx)];
                if !value.is_nan() {
                    daily_factor_values.push(value);
                    valid_union_indices.push(union_idx);
                    valid_style_indices.push(style_stock_idx);
                }
            }
        }
    }

    if daily_factor_values.len() < 12 {
        return Ok(Vec::new());
    }

    let ranked_values = cross_section_rank_io_optimized(&daily_factor_values);

    if let Some(regression_matrix) = &day_data.regression_matrix {
        let mut selected_regression_cols = Vec::with_capacity(valid_style_indices.len());
        for &style_idx in &valid_style_indices {
            selected_regression_cols.push(regression_matrix.column(style_idx).clone_owned());
        }

        let selected_regression_matrix = DMatrix::from_columns(&selected_regression_cols);
        let aligned_y_vector = DVector::from_vec(ranked_values.clone());

        let beta = &selected_regression_matrix * &aligned_y_vector;

        let n_features = day_data.style_matrix.ncols();
        let mut result_values = Vec::new();
        for (i, &union_idx) in valid_union_indices.iter().enumerate() {
            let style_idx = valid_style_indices[i];

            let mut predicted_value = 0.0;
            for j in 0..n_features {
                predicted_value += day_data.style_matrix[(style_idx, j)] * beta[j];
            }

            let residual = ranked_values[i] - predicted_value;
            result_values.push((union_idx, residual));
        }

        Ok(result_values)
    } else {
        Ok(Vec::new())
    }
}

/// I/O优化的中性化结果
pub struct IOOptimizedNeutralizationResult {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    pub neutralized_values: DMatrix<f64>,
}

/// I/O优化的结果保存
fn save_neutralized_result_io_optimized(
    result: IOOptimizedNeutralizationResult,
    output_path: &Path,
) -> PyResult<()> {
    use arrow::array::{ArrayRef, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, Encoding};
    use parquet::file::properties::WriterProperties;

    // 优化的Schema构建
    let mut fields = Vec::with_capacity(result.stocks.len() + 1);
    fields.push(Field::new("date", DataType::Int64, false));
    for stock in &result.stocks {
        fields.push(Field::new(stock, DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));

    // 优化的数组构建
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(result.stocks.len() + 1);

    // 日期数组
    arrays.push(Arc::new(Int64Array::from(result.dates.clone())));

    // 并行构建股票数据数组
    let stock_arrays: Vec<ArrayRef> = (0..result.stocks.len())
        .into_par_iter()
        .map(|stock_idx| {
            let column_data: Vec<Option<f64>> = (0..result.dates.len())
                .map(|date_idx| {
                    let value = result.neutralized_values[(date_idx, stock_idx)];
                    if value.is_nan() {
                        None
                    } else {
                        Some(value)
                    }
                })
                .collect();
            Arc::new(Float64Array::from(column_data)) as ArrayRef
        })
        .collect();

    arrays.extend(stock_arrays);

    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("创建RecordBatch失败: {}", e)))?;

    // I/O优化的写入配置
    let props = WriterProperties::builder()
        .set_compression(Compression::LZ4) // 使用更快的压缩算法
        .set_encoding(Encoding::PLAIN)
        .set_max_row_group_size(200000) // 更大的行组
        .set_write_batch_size(10000) // 优化写入批次大小
        .build();

    let file = File::create(output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("创建输出文件失败: {}", e)))?;

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| PyRuntimeError::new_err(format!("创建Arrow写入器失败: {}", e)))?;

    writer
        .write(&batch)
        .map_err(|e| PyRuntimeError::new_err(format!("写入数据失败: {}", e)))?;

    writer
        .close()
        .map_err(|e| PyRuntimeError::new_err(format!("关闭写入器失败: {}", e)))?;

    Ok(())
}
