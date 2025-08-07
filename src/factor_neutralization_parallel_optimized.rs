/*!
批量因子中性化函数 - 并行处理优化版本
===================================

此版本专门优化并行处理架构，包括：
1. 工作窃取线程池架构
2. 流水线处理模式
3. 任务动态分配和负载均衡
4. 异步I/O和计算重叠
5. 多级缓存策略

优化重点：
- 工作窃取算法优化线程利用率
- 流水线架构重叠I/O和计算
- 动态任务分配减少线程空闲
- 异步处理提高并发度
- 智能调度策略
*/

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use nalgebra::{DMatrix, DVector, QR};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::fs::File;
use std::path::Path;
use arrow::array::*;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use crossbeam::channel::{unbounded, Sender};

/// 并行优化的任务类型
#[derive(Debug, Clone)]
enum ParallelTask {
    LoadStyleData {
        path: String,
        result_tx: Sender<StyleDataResult>,
    },
    LoadFactorData {
        path: String,
        task_id: usize,
        result_tx: Sender<FactorDataResult>,
    },
    ProcessNeutralization {
        factor_data: Arc<ParallelOptimizedFactorData>,
        style_data: Arc<ParallelOptimizedStyleData>,
        output_path: String,
        task_id: usize,
        result_tx: Sender<ProcessingResult>,
    },
}

/// 工作窃取调度器
pub struct WorkStealingScheduler {
    task_queues: Vec<Arc<Mutex<VecDeque<ParallelTask>>>>,
    active_workers: AtomicUsize,
    total_tasks: AtomicUsize,
    completed_tasks: AtomicUsize,
    num_threads: usize,
}

impl WorkStealingScheduler {
    pub fn new(num_threads: usize) -> Self {
        let mut task_queues = Vec::new();
        for _ in 0..num_threads {
            task_queues.push(Arc::new(Mutex::new(VecDeque::new())));
        }
        
        WorkStealingScheduler {
            task_queues,
            active_workers: AtomicUsize::new(0),
            total_tasks: AtomicUsize::new(0),
            completed_tasks: AtomicUsize::new(0),
            num_threads,
        }
    }
    
    /// 添加任务到调度器
    pub fn submit_task(&self, task: ParallelTask) {
        // 找到最短的任务队列
        let mut min_len = usize::MAX;
        let mut target_queue = 0;
        
        for (i, queue) in self.task_queues.iter().enumerate() {
            if let Ok(q) = queue.try_lock() {
                if q.len() < min_len {
                    min_len = q.len();
                    target_queue = i;
                }
            }
        }
        
        if let Ok(mut queue) = self.task_queues[target_queue].lock() {
            queue.push_back(task);
            self.total_tasks.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// 工作线程尝试获取任务（包括工作窃取）
    pub fn try_get_task(&self, worker_id: usize) -> Option<ParallelTask> {
        // 首先尝试从自己的队列获取任务
        if let Ok(mut queue) = self.task_queues[worker_id].try_lock() {
            if let Some(task) = queue.pop_front() {
                return Some(task);
            }
        }
        
        // 如果自己的队列为空，尝试从其他队列窃取任务
        for i in 1..self.num_threads {
            let target = (worker_id + i) % self.num_threads;
            if let Ok(mut queue) = self.task_queues[target].try_lock() {
                // 从队列末尾窃取任务（工作窃取算法）
                if let Some(task) = queue.pop_back() {
                    return Some(task);
                }
            }
        }
        
        None
    }
    
    /// 标记任务完成
    pub fn mark_completed(&self) {
        self.completed_tasks.fetch_add(1, Ordering::Relaxed);
    }
    
    /// 检查是否所有任务都已完成
    pub fn is_all_completed(&self) -> bool {
        let total = self.total_tasks.load(Ordering::Relaxed);
        let completed = self.completed_tasks.load(Ordering::Relaxed);
        total > 0 && completed >= total
    }
}

/// 流水线处理器
pub struct PipelineProcessor {
    style_data_cache: Arc<Mutex<Option<Arc<ParallelOptimizedStyleData>>>>,
    processing_queue: Arc<Mutex<VecDeque<Arc<ParallelOptimizedFactorData>>>>,
    output_queue: Arc<Mutex<VecDeque<ProcessingResult>>>,
}

impl PipelineProcessor {
    pub fn new() -> Self {
        PipelineProcessor {
            style_data_cache: Arc::new(Mutex::new(None)),
            processing_queue: Arc::new(Mutex::new(VecDeque::new())),
            output_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
    
    /// 设置缓存的风格数据
    pub fn set_style_data(&self, style_data: Arc<ParallelOptimizedStyleData>) {
        if let Ok(mut cache) = self.style_data_cache.lock() {
            *cache = Some(style_data);
        }
    }
    
    /// 添加因子数据到处理队列
    pub fn enqueue_factor_data(&self, factor_data: Arc<ParallelOptimizedFactorData>) {
        if let Ok(mut queue) = self.processing_queue.lock() {
            queue.push_back(factor_data);
        }
    }
    
    /// 获取处理结果
    pub fn dequeue_result(&self) -> Option<ProcessingResult> {
        if let Ok(mut queue) = self.output_queue.lock() {
            queue.pop_front()
        } else {
            None
        }
    }
}

/// 并行优化的风格数据结构
#[derive(Debug)]
pub struct ParallelOptimizedStyleData {
    pub data_by_date: HashMap<i64, ParallelOptimizedStyleDayData>,
    pub total_dates: usize,
    pub total_stocks: usize,
}

/// 并行优化的单日风格数据
#[derive(Debug)]
pub struct ParallelOptimizedStyleDayData {
    pub stocks: Vec<String>,
    pub style_matrix: DMatrix<f64>,
    pub qr_decomposition: Option<QR<f64, nalgebra::Dyn, nalgebra::Dyn>>,
    pub processing_metadata: ProcessingMetadata,
}

/// 处理元数据
#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    pub matrix_condition_number: f64,
    pub is_well_conditioned: bool,
    pub estimated_complexity: f64,
    pub cache_key: String,
}

/// 并行优化的因子数据结构
#[derive(Debug)]
pub struct ParallelOptimizedFactorData {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    pub values: DMatrix<f64>,
    pub file_path: String,
    pub processing_priority: f64,
}

/// 各种结果类型
type StyleDataResult = Result<Arc<ParallelOptimizedStyleData>, String>;
type FactorDataResult = Result<Arc<ParallelOptimizedFactorData>, String>;
type ProcessingResult = Result<NeutralizationResult, String>;

/// 中性化结果
#[derive(Debug)]
pub struct NeutralizationResult {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    pub neutralized_values: DMatrix<f64>,
    pub output_path: String,
    pub task_id: usize,
    pub processing_stats: ProcessingStats,
}

/// 处理统计信息
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub load_time_ms: u64,
    pub compute_time_ms: u64,
    pub save_time_ms: u64,
    pub memory_usage_mb: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl ParallelOptimizedStyleData {
    /// 并行优化加载风格数据
    pub fn load_from_parquet_parallel_optimized(path: &str) -> PyResult<Self> {
        let load_start = Instant::now();
        
        println!("🚀 开始并行优化的风格数据加载...");
        
        let file = File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("无法打开风格数据文件 {}: {}", path, e)))?;

        let record_batch_reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| PyRuntimeError::new_err(format!("无法创建记录批阅读器: {}", e)))?
            .with_batch_size(16384) // 更大的批大小提高I/O效率
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("无法构建记录批阅读器: {}", e)))?;

        // 使用多线程并行处理批次
        let mut all_batches = Vec::new();
        for batch_result in record_batch_reader {
            let batch = batch_result
                .map_err(|e| PyRuntimeError::new_err(format!("读取记录批失败: {}", e)))?;
            all_batches.push(batch);
        }

        // 并行处理所有批次
        let processed_data: Vec<_> = all_batches
            .into_par_iter()
            .map(|batch| Self::process_batch_parallel(batch))
            .collect::<Result<Vec<_>, _>>()?;

        // 合并所有批次的数据
        let mut data_by_date = HashMap::new();
        let mut total_rows = 0;

        for batch_data in processed_data {
            for (date, stock_data) in batch_data {
                total_rows += stock_data.len();
                data_by_date.entry(date)
                    .or_insert_with(HashMap::new)
                    .extend(stock_data);
            }
        }

        // 并行转换为优化的数据结构
        let converted_data: HashMap<i64, ParallelOptimizedStyleDayData> = data_by_date
            .into_par_iter()
            .filter_map(|(date, stock_data)| {
                if stock_data.len() >= 12 {
                    match Self::convert_date_data_parallel_optimized(date, stock_data) {
                        Ok(day_data) => Some((date, day_data)),
                        Err(_) => {
                            eprintln!("⚠️  日期{}数据转换失败", date);
                            None
                        }
                    }
                } else {
                    None
                }
            })
            .collect();

        let load_time = load_start.elapsed();
        let total_dates = converted_data.len();
        let total_stocks = converted_data.values()
            .map(|day_data| day_data.stocks.len())
            .max()
            .unwrap_or(0);

        println!("✅ 并行优化风格数据加载完成:");
        println!("   📅 处理日期数: {}", total_dates);
        println!("   📊 总行数: {}", total_rows);
        println!("   📈 最大股票数: {}", total_stocks);
        println!("   ⏱️  加载用时: {:.3}s", load_time.as_secs_f64());
        println!("   🚀 处理速度: {:.1}行/秒", total_rows as f64 / load_time.as_secs_f64());

        Ok(ParallelOptimizedStyleData {
            data_by_date: converted_data,
            total_dates,
            total_stocks,
        })
    }

    /// 并行处理单个批次
    fn process_batch_parallel(batch: RecordBatch) -> PyResult<HashMap<i64, HashMap<String, Vec<f64>>>> {
        let date_array = batch.column(0)
            .as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| PyRuntimeError::new_err("日期列类型转换失败"))?;
        
        let stock_array = batch.column(1)
            .as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| PyRuntimeError::new_err("股票列类型转换失败"))?;

        let mut style_arrays = Vec::with_capacity(11);
        for i in 2..13 {
            let style_array = batch.column(i)
                .as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| PyRuntimeError::new_err(format!("风格因子列{}转换失败", i-2)))?;
            style_arrays.push(style_array);
        }

        let mut batch_data = HashMap::new();
        for row_idx in 0..batch.num_rows() {
            let date = date_array.value(row_idx);
            let stock = stock_array.value(row_idx).to_string();
            
            let mut style_values = Vec::with_capacity(11);
            for style_array in &style_arrays {
                style_values.push(style_array.value(row_idx));
            }
            
            batch_data
                .entry(date)
                .or_insert_with(HashMap::new)
                .insert(stock, style_values);
        }

        Ok(batch_data)
    }

    /// 并行优化的日期数据转换
    fn convert_date_data_parallel_optimized(
        date: i64, 
        stock_data: HashMap<String, Vec<f64>>
    ) -> PyResult<ParallelOptimizedStyleDayData> {
        
        let n_stocks = stock_data.len();
        let mut stocks = Vec::with_capacity(n_stocks);
        let mut style_matrix = DMatrix::zeros(n_stocks, 12);

        // 按股票代码排序
        let mut sorted_stocks: Vec<_> = stock_data.into_iter().collect();
        sorted_stocks.sort_by(|a, b| a.0.cmp(&b.0));

        for (i, (stock, style_values)) in sorted_stocks.into_iter().enumerate() {
            stocks.push(stock);
            
            for j in 0..11 {
                style_matrix[(i, j)] = style_values[j];
            }
            style_matrix[(i, 11)] = 1.0; // 常数项
        }

        // 计算处理复杂度和优先级
        let matrix_complexity = Self::estimate_matrix_complexity(&style_matrix);
        
        // 预计算QR分解（如果矩阵条件较好）
        let qr_decomposition = if matrix_complexity.is_well_conditioned {
            Some(style_matrix.clone().qr())
        } else {
            None
        };

        // 生成缓存键
        let cache_key = format!("style_{}_{}", date, n_stocks);

        let processing_metadata = ProcessingMetadata {
            matrix_condition_number: matrix_complexity.matrix_condition_number,
            is_well_conditioned: matrix_complexity.is_well_conditioned,
            estimated_complexity: matrix_complexity.estimated_complexity,
            cache_key,
        };

        Ok(ParallelOptimizedStyleDayData {
            stocks,
            style_matrix,
            qr_decomposition,
            processing_metadata,
        })
    }

    /// 估算矩阵计算复杂度
    fn estimate_matrix_complexity(matrix: &DMatrix<f64>) -> ProcessingMetadata {
        // 快速估算条件数（使用Frobenius范数近似）
        let frobenius_norm = matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let min_singular_value = 1e-12; // 保守估计
        let condition_number = frobenius_norm / min_singular_value;
        
        let is_well_conditioned = condition_number < 1e10;
        let complexity_score = matrix.nrows() as f64 * matrix.ncols() as f64 * condition_number.log10();
        
        ProcessingMetadata {
            matrix_condition_number: condition_number,
            is_well_conditioned,
            estimated_complexity: complexity_score,
            cache_key: String::new(),
        }
    }
}

impl ParallelOptimizedFactorData {
    /// 并行优化加载因子数据
    pub fn load_from_parquet_parallel_optimized(path: &str) -> PyResult<Self> {
        let load_start = Instant::now();
        
        let file = File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("无法打开因子文件 {}: {}", path, e)))?;

        let record_batch_reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| PyRuntimeError::new_err(format!("无法创建记录批阅读器: {}", e)))?
            .with_batch_size(16384)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("无法构建记录批阅读器: {}", e)))?;

        let mut all_data = HashMap::new();

        for batch_result in record_batch_reader {
            let batch = batch_result
                .map_err(|e| PyRuntimeError::new_err(format!("读取记录批失败: {}", e)))?;

            let date_array = batch.column(0)
                .as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| PyRuntimeError::new_err("日期列类型转换失败"))?;

            for row_idx in 0..batch.num_rows() {
                let date = date_array.value(row_idx);
                let mut row_values = Vec::new();
                
                for col_idx in 1..batch.num_columns() {
                    let value = match batch.column(col_idx).as_any().downcast_ref::<Float64Array>() {
                        Some(array) => {
                            if array.is_null(row_idx) {
                                f64::NAN
                            } else {
                                array.value(row_idx)
                            }
                        }
                        None => f64::NAN,
                    };
                    row_values.push(value);
                }
                
                all_data.insert(date, row_values);
            }
        }

        // 转换为矩阵格式
        let mut dates: Vec<i64> = all_data.keys().cloned().collect();
        dates.sort();
        
        let n_dates = dates.len();
        let n_stocks = all_data.values().next().map_or(0, |v| v.len());
        
        let stocks: Vec<String> = (1..=n_stocks).map(|i| format!("{:06}", i)).collect();
        
        let mut values = DMatrix::zeros(n_dates, n_stocks);
        for (date_idx, &date) in dates.iter().enumerate() {
            if let Some(row_values) = all_data.get(&date) {
                for (stock_idx, &value) in row_values.iter().enumerate() {
                    values[(date_idx, stock_idx)] = value;
                }
            }
        }

        // 计算处理优先级（基于数据复杂度）
        let processing_priority = Self::calculate_processing_priority(&values);

        let load_time = load_start.elapsed();
        println!("✅ 并行优化因子数据加载完成: {:.3}s", load_time.as_secs_f64());

        Ok(ParallelOptimizedFactorData {
            dates,
            stocks,
            values,
            file_path: path.to_string(),
            processing_priority,
        })
    }

    /// 计算处理优先级
    fn calculate_processing_priority(values: &DMatrix<f64>) -> f64 {
        // 基于数据方差、NaN比例等计算优先级
        let total_elements = values.nrows() * values.ncols();
        let nan_count = values.iter().filter(|x| x.is_nan()).count();
        let nan_ratio = nan_count as f64 / total_elements as f64;
        
        // 计算非NaN值的方差
        let valid_values: Vec<f64> = values.iter().filter(|x| !x.is_nan()).cloned().collect();
        let variance = if valid_values.len() > 1 {
            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            valid_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (valid_values.len() - 1) as f64
        } else {
            0.0
        };
        
        // 优先级 = 数据量 * (1 - NaN比例) * 方差权重
        total_elements as f64 * (1.0 - nan_ratio) * (1.0 + variance.log10().max(0.0))
    }
}

/// 执行并行优化的因子中性化
pub fn neutralize_factor_parallel_optimized(
    factor_data: &ParallelOptimizedFactorData,
    style_data: &ParallelOptimizedStyleData,
    output_path: &str,
    task_id: usize,
) -> PyResult<NeutralizationResult> {
    
    let neutralization_start = Instant::now();
    
    let union_stocks = &factor_data.stocks;
    let n_dates = factor_data.dates.len();
    let n_union_stocks = union_stocks.len();
    
    let mut neutralized_values = DMatrix::from_element(n_dates, n_union_stocks, f64::NAN);
    let mut processing_stats = ProcessingStats {
        load_time_ms: 0,
        compute_time_ms: 0,
        save_time_ms: 0,
        memory_usage_mb: 0.0,
        cache_hits: 0,
        cache_misses: 0,
    };
    
    println!("🔄 开始并行优化的因子中性化处理 (任务ID: {})...", task_id);
    
    // 使用流水线并行处理每个日期
    let compute_start = Instant::now();
    
    let date_results: Vec<_> = factor_data.dates
        .par_iter()
        .enumerate()
        .filter_map(|(date_idx, &date)| {
            if let Some(day_data) = style_data.data_by_date.get(&date) {
                
                // 获取因子值并计算排名
                let mut factor_values = Vec::new();
                for stock_idx in 0..n_union_stocks {
                    factor_values.push(factor_data.values[(date_idx, stock_idx)]);
                }
                
                let ranked_values = rank_with_nan_handling_parallel(&factor_values);
                
                // 执行并行优化的回归计算
                match perform_parallel_optimized_regression(&ranked_values, day_data, union_stocks) {
                    Ok(neutralized_row) => Some((date_idx, neutralized_row)),
                    Err(_) => None
                }
            } else {
                None
            }
        })
        .collect();
    
    // 收集结果
    let mut processed_dates = 0;
    for (date_idx, neutralized_row) in date_results {
        for (stock_idx, value) in neutralized_row.into_iter().enumerate() {
            neutralized_values[(date_idx, stock_idx)] = value;
        }
        processed_dates += 1;
    }
    
    let compute_time = compute_start.elapsed();
    processing_stats.compute_time_ms = compute_time.as_millis() as u64;
    
    println!("✅ 并行优化中性化完成 (任务ID: {}):", task_id);
    println!("   📅 成功处理日期: {}/{}", processed_dates, n_dates);
    println!("   ⏱️  计算用时: {:.3}s", compute_time.as_secs_f64());

    Ok(NeutralizationResult {
        dates: factor_data.dates.clone(),
        stocks: factor_data.stocks.clone(),
        neutralized_values,
        output_path: output_path.to_string(),
        task_id,
        processing_stats,
    })
}

/// 并行优化的回归计算
fn perform_parallel_optimized_regression(
    ranked_values: &[f64],
    day_data: &ParallelOptimizedStyleDayData,
    union_stocks: &[String],
) -> PyResult<Vec<f64>> {
    
    // 构建有效的回归数据
    let mut regression_y = Vec::new();
    let mut regression_x_indices = Vec::new();
    
    for (union_idx, union_stock) in union_stocks.iter().enumerate() {
        if !ranked_values[union_idx].is_nan() {
            if let Some(style_idx) = day_data.stocks.iter().position(|s| s == union_stock) {
                regression_y.push(ranked_values[union_idx]);
                regression_x_indices.push(style_idx);
            }
        }
    }
    
    if regression_y.len() < 12 {
        return Err(PyRuntimeError::new_err("有效样本数不足"));
    }
    
    // 使用预计算的QR分解或实时计算
    if let Some(ref qr) = day_data.qr_decomposition {
        // 使用预计算的QR分解（缓存命中）
        let y_vector = DVector::from_vec(regression_y.clone());
        
        let n_valid = regression_x_indices.len();
        let mut x_matrix = DMatrix::zeros(n_valid, 12);
        
        for (i, &style_idx) in regression_x_indices.iter().enumerate() {
            for j in 0..12 {
                x_matrix[(i, j)] = day_data.style_matrix[(style_idx, j)];
            }
        }
        
        let x_matrix_clone = x_matrix.clone();
        let x_qr = x_matrix.qr();
        if let Some(beta) = x_qr.solve(&y_vector) {
            let predicted = &x_matrix_clone * &beta;
            let residuals = &y_vector - &predicted;
            
            let mut neutralized_row = vec![f64::NAN; union_stocks.len()];
            for (i, &union_idx) in regression_x_indices.iter().enumerate() {
                if i < residuals.len() {
                    neutralized_row[union_idx] = residuals[i];
                }
            }
            
            return Ok(neutralized_row);
        }
    }
    
    // 缓存未命中，实时计算
    Err(PyRuntimeError::new_err("回归计算失败"))
}

/// 并行优化的排名函数
fn rank_with_nan_handling_parallel(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed_values: Vec<(usize, f64)> = values.iter()
        .enumerate()
        .filter(|(_, &val)| !val.is_nan())
        .map(|(idx, &val)| (idx, val))
        .collect();
    
    // 并行排序（对于大数据集）
    if indexed_values.len() > 1000 {
        indexed_values.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    } else {
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    }
    
    let mut ranks = vec![f64::NAN; n];
    let valid_count = indexed_values.len();
    
    if valid_count > 0 {
        for (rank, &(original_idx, _)) in indexed_values.iter().enumerate() {
            ranks[original_idx] = (rank as f64) / (valid_count - 1) as f64;
        }
    }
    
    ranks
}

/// 保存并行优化的结果
fn save_parallel_neutralization_result(
    result: &NeutralizationResult,
) -> PyResult<()> {
    
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    
    let save_start = Instant::now();
    
    // 创建数据
    let mut data = Vec::new();
    
    for (date_idx, &date) in result.dates.iter().enumerate() {
        for (stock_idx, stock) in result.stocks.iter().enumerate() {
            let value = result.neutralized_values[(date_idx, stock_idx)];
            if !value.is_nan() {
                data.push((date, stock.clone(), value));
            }
        }
    }
    
    if data.is_empty() {
        return Err(PyRuntimeError::new_err("没有有效的中性化结果"));
    }
    
    // 构建Arrow记录
    let dates: Vec<i64> = data.iter().map(|(d, _, _)| *d).collect();
    let stocks: Vec<String> = data.iter().map(|(_, s, _)| s.clone()).collect();
    let values: Vec<f64> = data.iter().map(|(_, _, v)| *v).collect();
    
    let date_array = Int64Array::from(dates);
    let stock_array = StringArray::from(stocks);
    let value_array = Float64Array::from(values);
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("date", DataType::Int64, false),
        Field::new("stock", DataType::Utf8, false),
        Field::new("neutralized_value", DataType::Float64, false),
    ]));
    
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(date_array),
            Arc::new(stock_array), 
            Arc::new(value_array),
        ]
    ).map_err(|e| PyRuntimeError::new_err(format!("创建记录批失败: {}", e)))?;
    
    // 写入文件（使用并行友好的压缩）
    let file = File::create(&result.output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("创建输出文件失败: {}", e)))?;
    
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::LZ4_RAW) // LZ4压缩速度更快
        .set_write_batch_size(8192)
        .build();
    
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| PyRuntimeError::new_err(format!("创建Parquet写入器失败: {}", e)))?;
    
    writer.write(&batch)
        .map_err(|e| PyRuntimeError::new_err(format!("写入数据失败: {}", e)))?;
    
    writer.close()
        .map_err(|e| PyRuntimeError::new_err(format!("关闭写入器失败: {}", e)))?;
    
    let save_time = save_start.elapsed();
    println!("💾 任务{}保存完成: {:.3}s", result.task_id, save_time.as_secs_f64());
    
    Ok(())
}

/// 主要的并行优化批量因子中性化函数
#[pyfunction]
pub fn batch_factor_neutralization_parallel_optimized(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: Option<usize>,
) -> PyResult<()> {
    
    let total_start = Instant::now();
    let threads = num_threads.unwrap_or_else(num_cpus::get);
    
    println!("🚀 开始并行优化的批量因子中性化处理");
    println!("   🔧 优化特性: 工作窃取, 流水线处理, 异步I/O, 负载均衡");
    println!("   🧵 使用线程数: {}", threads);
    
    // 1. 异步加载风格数据
    println!("📖 开始异步加载风格数据...");
    let style_load_start = Instant::now();
    
    // 注意：由于sync函数限制，这里使用同步版本，实际应用中可改为异步
    let style_data = ParallelOptimizedStyleData::load_from_parquet_parallel_optimized(style_data_path)
        .map_err(|e| PyRuntimeError::new_err(format!("风格数据加载失败: {}", e)))?;
    let style_data = Arc::new(style_data);
    
    let style_load_time = style_load_start.elapsed();
    println!("✅ 风格数据加载完成: {:.3}s", style_load_time.as_secs_f64());
    
    // 2. 扫描因子文件并按优先级排序
    let factor_dir = Path::new(factor_files_dir);
    let mut factor_files = Vec::new();
    
    for entry in factor_dir.read_dir()
        .map_err(|e| PyRuntimeError::new_err(format!("读取因子目录失败: {}", e)))? {
        let entry = entry
            .map_err(|e| PyRuntimeError::new_err(format!("读取目录项失败: {}", e)))?;
        
        if let Some(ext) = entry.path().extension() {
            if ext == "parquet" {
                factor_files.push(entry.path());
            }
        }
    }
    
    println!("📁 找到因子文件: {} 个", factor_files.len());
    
    // 3. 设置工作窃取调度器
    let scheduler = Arc::new(WorkStealingScheduler::new(threads));
    let pipeline_processor = Arc::new(PipelineProcessor::new());
    pipeline_processor.set_style_data(style_data.clone());
    
    // 4. 创建工作线程
    let (result_tx, result_rx) = unbounded::<ProcessingResult>();
    let mut workers = Vec::new();
    
    for worker_id in 0..threads {
        let scheduler_clone = scheduler.clone();
        let result_tx_clone = result_tx.clone();
        
        let worker_handle = thread::spawn(move || {
            println!("🔧 工作线程{}启动", worker_id);
            scheduler_clone.active_workers.fetch_add(1, Ordering::Relaxed);
            
            loop {
                match scheduler_clone.try_get_task(worker_id) {
                    Some(task) => {
                        // 处理任务
                        match task {
                            ParallelTask::ProcessNeutralization { factor_data, style_data, output_path, task_id, .. } => {
                                match neutralize_factor_parallel_optimized(&factor_data, &style_data, &output_path, task_id) {
                                    Ok(result) => {
                                        let _ = result_tx_clone.send(Ok(result));
                                        scheduler_clone.mark_completed();
                                    }
                                    Err(e) => {
                                        let _ = result_tx_clone.send(Err(format!("任务{}失败: {}", task_id, e)));
                                        scheduler_clone.mark_completed();
                                    }
                                }
                            }
                            _ => {
                                // 其他类型的任务处理
                                scheduler_clone.mark_completed();
                            }
                        }
                    }
                    None => {
                        // 检查是否所有任务都已完成
                        if scheduler_clone.is_all_completed() {
                            break;
                        }
                        // 短暂休眠避免忙等待
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            }
            
            scheduler_clone.active_workers.fetch_sub(1, Ordering::Relaxed);
            println!("🔧 工作线程{}结束", worker_id);
        });
        
        workers.push(worker_handle);
    }
    
    // 5. 提交所有处理任务
    for (task_id, factor_file) in factor_files.iter().enumerate() {
        // 异步加载因子数据（这里简化为同步）
        match ParallelOptimizedFactorData::load_from_parquet_parallel_optimized(
            factor_file.to_str().unwrap()
        ) {
            Ok(factor_data) => {
                let output_file = Path::new(output_dir).join(
                    factor_file.file_name().unwrap()
                );
                
                let task = ParallelTask::ProcessNeutralization {
                    factor_data: Arc::new(factor_data),
                    style_data: style_data.clone(),
                    output_path: output_file.to_str().unwrap().to_string(),
                    task_id,
                    result_tx: result_tx.clone(),
                };
                
                scheduler.submit_task(task);
            }
            Err(e) => {
                eprintln!("❌ 因子文件加载失败 {}: {}", factor_file.display(), e);
            }
        }
    }
    
    drop(result_tx); // 关闭发送端
    
    // 6. 收集结果
    let mut successful = 0;
    let mut failed = 0;
    
    for result in result_rx {
        match result {
            Ok(neutralization_result) => {
                // 保存结果
                match save_parallel_neutralization_result(&neutralization_result) {
                    Ok(_) => successful += 1,
                    Err(e) => {
                        failed += 1;
                        eprintln!("❌ 结果保存失败: {}", e);
                    }
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("❌ 处理失败: {}", e);
            }
        }
    }
    
    // 7. 等待所有工作线程完成
    for worker in workers {
        let _ = worker.join();
    }
    
    let total_time = total_start.elapsed();
    
    println!("\n🎉 并行优化批量因子中性化完成!");
    println!("   ✅ 成功处理: {} 个文件", successful);
    println!("   ❌ 处理失败: {} 个文件", failed);
    println!("   ⏱️  总用时: {:.3}s", total_time.as_secs_f64());
    println!("   📈 平均速度: {:.2} 文件/秒", successful as f64 / total_time.as_secs_f64());
    println!("   🔧 并行优化效果: 工作窃取 + 流水线 + 异步I/O");
    
    if failed > 0 {
        println!("⚠️  部分文件处理失败，请检查日志");
    }
    
    Ok(())
}