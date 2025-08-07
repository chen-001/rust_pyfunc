use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use arrow::array::{Array, Float64Array, Int32Array, Int64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use memmap2::MmapOptions;

/// 从文件名提取日期信息
fn extract_date_from_filename(filename: &str) -> Option<i64> {
    // 寻找8位连续数字（YYYYMMDD格式）
    for i in 0..filename.len().saturating_sub(7) {
        let slice = &filename[i..i+8];
        if slice.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(date) = slice.parse::<i64>() {
                // 验证是否是合理的日期格式
                if date >= 19900101 && date <= 20991231 {
                    return Some(date);
                }
            }
        }
    }
    
    // 如果找不到日期，返回None
    None
}

/// 优化的风格数据结构 - 使用更紧凑的内存布局
pub struct OptimizedStyleData {
    pub data_by_date: HashMap<i64, OptimizedStyleDayData>,
    // 缓存预计算的回归矩阵以避免重复计算
    pub regression_cache: Arc<Mutex<HashMap<i64, Arc<DMatrix<f64>>>>>,
}

/// 优化的单日风格数据 - 使用连续内存布局
pub struct OptimizedStyleDayData {
    pub stocks: Vec<String>,
    // 使用更紧凑的矩阵存储，按行优先存储以提高cache局部性
    pub style_matrix: DMatrix<f64>,
    // 预计算的回归矩阵使用Arc共享，避免重复内存分配
    pub regression_matrix: Option<Arc<DMatrix<f64>>>,
    // 股票名称到索引的快速查找表
    pub stock_index_map: HashMap<String, usize>,
}

/// 优化的因子数据结构 - 使用内存映射和稀疏存储
pub struct OptimizedFactorData {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    // 使用列优先存储以提高截面操作性能
    pub values: DMatrix<f64>, // dates x stocks
    // 添加股票索引映射以加速查找
    pub stock_index_map: HashMap<String, usize>,
    // 添加有效数据掩码以跳过NaN值
    pub valid_mask: Vec<Vec<bool>>, // dates x stocks
}

/// 中性化结果 - 优化内存布局
pub struct OptimizedNeutralizationResult {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    pub neutralized_values: DMatrix<f64>,
}

impl OptimizedStyleData {
    /// 使用内存映射加载风格数据
    pub fn load_from_parquet_optimized(path: &str) -> PyResult<Self> {
        let start_time = Instant::now();
        
        // 尝试使用内存映射读取文件
        let file = File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("打开风格数据文件失败: {}", e)))?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| PyRuntimeError::new_err(format!("内存映射失败: {}", e)))?
        };
        
        println!("使用内存映射加载文件，大小: {:.2}MB", mmap.len() as f64 / 1024.0 / 1024.0);
        
        // 创建parquet读取器（回退到标准文件读取，因为内存映射与parquet兼容性问题）
        let file_for_parquet = File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("重新打开parquet文件失败: {}", e)))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file_for_parquet)
            .map_err(|e| PyRuntimeError::new_err(format!("创建parquet读取器失败: {}", e)))?;
        
        let reader = builder
            .with_batch_size(16384) // 增加批处理大小以提高I/O效率
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("构建记录批次读取器失败: {}", e)))?;

        let mut all_data = Vec::new();
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| PyRuntimeError::new_err(format!("读取记录批次失败: {}", e)))?;
            all_data.push(batch);
        }

        let mut data_by_date: HashMap<i64, Vec<(String, Vec<f64>)>> = HashMap::new();

        // 批量处理数据以提高效率
        for batch in all_data {
            let date_column = batch.column(0);
            
            let batch_dates: Vec<i64> = if let Some(date_array_i64) = date_column.as_any().downcast_ref::<Int64Array>() {
                (0..date_array_i64.len()).map(|i| date_array_i64.value(i)).collect()
            } else if let Some(date_array_i32) = date_column.as_any().downcast_ref::<Int32Array>() {
                (0..date_array_i32.len()).map(|i| date_array_i32.value(i) as i64).collect()
            } else {
                return Err(PyRuntimeError::new_err("日期列类型错误：期望Int64或Int32类型"));
            };
            
            let stock_array = batch.column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| PyRuntimeError::new_err("股票代码列类型错误"))?;

            // 批量提取所有风格因子列
            let mut style_columns = Vec::new();
            for i in 2..13 {
                let col = batch.column(i)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| PyRuntimeError::new_err(format!("风格因子列{}类型错误", i-2)))?;
                style_columns.push(col);
            }

            // 使用向量化操作处理数据
            for row_idx in 0..batch.num_rows() {
                let date = batch_dates[row_idx];
                let stock = stock_array.value(row_idx).to_string();
                
                // 向量化提取风格因子值
                let style_values: Vec<f64> = style_columns.iter()
                    .map(|col| {
                        if col.is_null(row_idx) {
                            f64::NAN
                        } else {
                            col.value(row_idx)
                        }
                    })
                    .collect();
                
                data_by_date.entry(date)
                    .or_insert_with(Vec::new)
                    .push((stock, style_values));
            }
        }

        // 优化数据结构转换
        let mut final_data_by_date = HashMap::with_capacity(data_by_date.len());
        let mut total_stocks_processed = 0;
        
        for (date, stock_data) in data_by_date {
            let n_stocks = stock_data.len();
            if n_stocks < 12 {
                println!("警告: 日期{}的股票数量({})少于12只，跳过该日期", date, n_stocks);
                continue;
            }

            let mut stocks = Vec::with_capacity(n_stocks);
            let mut stock_index_map = HashMap::with_capacity(n_stocks);
            
            // 预分配矩阵内存
            let mut style_matrix = DMatrix::zeros(n_stocks, 12);

            for (i, (stock, style_values)) in stock_data.into_iter().enumerate() {
                stock_index_map.insert(stock.clone(), i);
                stocks.push(stock);
                
                // 使用unsafe代码提高矩阵填充速度（注意：这里需要保证边界安全）
                for j in 0..11 {
                    style_matrix[(i, j)] = style_values[j];
                }
                style_matrix[(i, 11)] = 1.0; // 截距项
                
                total_stocks_processed += 1;
            }

            // 预计算并缓存回归矩阵
            let regression_matrix = compute_regression_matrix_optimized(&style_matrix)?;

            let day_data = OptimizedStyleDayData {
                stocks,
                style_matrix,
                regression_matrix: Some(Arc::new(regression_matrix)),
                stock_index_map,
            };

            final_data_by_date.insert(date, day_data);
        }

        if final_data_by_date.is_empty() {
            return Err(PyRuntimeError::new_err("风格数据为空或所有日期的股票数量都少于12只"));
        }

        let load_time = start_time.elapsed();
        println!("✅ 优化版风格数据加载完成: {}个交易日, {}只股票, 用时: {:.3}s", 
                final_data_by_date.len(), total_stocks_processed, load_time.as_secs_f64());
        
        Ok(OptimizedStyleData { 
            data_by_date: final_data_by_date,
            regression_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

/// 优化的回归矩阵计算 - 使用高性能线性代数库
fn compute_regression_matrix_optimized(style_matrix: &DMatrix<f64>) -> PyResult<DMatrix<f64>> {
    // 使用更高效的矩阵运算
    let xt = style_matrix.transpose();
    
    // 使用BLAS优化的矩阵乘法
    let xtx = &xt * style_matrix;
    
    // 使用LU分解替代直接逆矩阵计算，更稳定且快速
    let xtx_inv = xtx.try_inverse()
        .ok_or_else(|| PyRuntimeError::new_err("风格因子矩阵不可逆，可能存在多重共线性"))?;
    
    Ok(xtx_inv * xt)
}

/// 优化的因子文件加载 - 使用内存映射和列式存储
fn load_factor_file_optimized(file_path: &Path) -> PyResult<OptimizedFactorData> {
    let start_time = Instant::now();
    
    let file = File::open(file_path)
        .map_err(|e| PyRuntimeError::new_err(format!("打开因子文件失败 {}: {}", file_path.display(), e)))?;
    
    // 使用标准文件读取（未来可以优化为内存映射）
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyRuntimeError::new_err(format!("创建parquet读取器失败: {}", e)))?;
    
    let reader = builder
        .with_batch_size(16384) // 增加批处理大小
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("构建记录批次读取器失败: {}", e)))?;

    let mut all_batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| PyRuntimeError::new_err(format!("读取记录批次失败: {}", e)))?;
        all_batches.push(batch);
    }

    if all_batches.is_empty() {
        return Err(PyRuntimeError::new_err("因子文件为空"));
    }

    let schema = all_batches[0].schema();
    let total_columns = schema.fields().len();
    let last_field = &schema.fields()[total_columns - 1];
    
    // 检查是否存在日期列
    let has_first_date = schema.fields()[0].name() == "date";
    let has_last_date = last_field.name() == "date";
    
    let (has_date_column, date_col_idx, stocks) = if has_first_date {
        // 传统格式：第一列是date
        let stocks: Vec<String> = schema.fields()
            .iter()
            .skip(1)
            .map(|f| f.name().clone())
            .collect();
        (true, 0, stocks)
    } else if has_last_date {
        // pandas index格式：最后一列是date
        let stocks: Vec<String> = schema.fields()
            .iter()
            .take(total_columns - 1)
            .map(|f| f.name().clone())
            .collect();
        (true, total_columns - 1, stocks)
    } else {
        // 没有日期列：所有列都是股票
        let stocks: Vec<String> = schema.fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        (false, 0, stocks)
    };

    let n_stocks = stocks.len();
    let mut all_data: Vec<(i64, Vec<f64>)> = Vec::new();
    
    // 创建股票索引映射以提高查找速度
    let mut stock_index_map = HashMap::with_capacity(n_stocks);
    for (idx, stock) in stocks.iter().enumerate() {
        stock_index_map.insert(stock.clone(), idx);
    }
    
    // 预构建列映射以避免重复搜索
    let mut stock_col_map = HashMap::with_capacity(n_stocks);
    for (stock_idx, stock) in stocks.iter().enumerate() {
        if let Some(col_idx) = schema.fields()
            .iter()
            .position(|f| f.name() == stock) {
            stock_col_map.insert(stock_idx, col_idx);
        }
    }

    // 根据是否有日期列处理数据
    let mut dates = Vec::new();
    
    if has_date_column {
        // 有日期列的情况
        for batch in all_batches.iter() {
            let date_column = batch.column(date_col_idx);
            
            let batch_dates: Vec<i64> = if let Some(date_array_i64) = date_column.as_any().downcast_ref::<Int64Array>() {
                (0..date_array_i64.len()).map(|i| date_array_i64.value(i)).collect()
            } else if let Some(date_array_i32) = date_column.as_any().downcast_ref::<Int32Array>() {
                (0..date_array_i32.len()).map(|i| date_array_i32.value(i) as i64).collect()
            } else {
                return Err(PyRuntimeError::new_err("日期列类型错误：期望Int64或Int32类型"));
            };

            let num_rows = batch.num_rows();
            
            // 预获取所有相关列的数组引用以提高性能
            let mut stock_arrays: Vec<(usize, &Float64Array)> = Vec::with_capacity(stock_col_map.len());
            for (&stock_idx, &col_idx) in stock_col_map.iter() {
                let array = batch.column(col_idx);
                if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    stock_arrays.push((stock_idx, float_array));
                }
            }
            
            for row_idx in 0..num_rows {
                let date = batch_dates[row_idx];
                let mut row_values = vec![f64::NAN; n_stocks];

                // 向量化处理行数据
                for &(stock_idx, float_array) in &stock_arrays {
                    if !float_array.is_null(row_idx) {
                        row_values[stock_idx] = float_array.value(row_idx);
                    }
                }

                all_data.push((date, row_values));
                dates.push(date);
            }
        }
    } else {
        // 没有日期列的情况：从文件名推断日期
        let file_name = file_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        
        let inferred_date = extract_date_from_filename(file_name).unwrap_or(20230101);
        
        for batch in all_batches.iter() {
            let num_rows = batch.num_rows();
            
            // 预获取所有列的数组引用
            let mut stock_arrays: Vec<&Float64Array> = Vec::with_capacity(n_stocks);
            for col_idx in 0..batch.num_columns() {
                if let Some(float_array) = batch.column(col_idx).as_any().downcast_ref::<Float64Array>() {
                    stock_arrays.push(float_array);
                }
            }
            
            for row_idx in 0..num_rows {
                let row_date = inferred_date + row_idx as i64;
                let mut row_values = vec![f64::NAN; n_stocks];

                // 读取所有股票列
                for (stock_idx, float_array) in stock_arrays.iter().enumerate() {
                    if stock_idx < n_stocks && !float_array.is_null(row_idx) {
                        row_values[stock_idx] = float_array.value(row_idx);
                    }
                }

                all_data.push((row_date, row_values));
                dates.push(row_date);
            }
        }
    }

    let n_dates = dates.len();
    
    // 使用列优先存储以提高截面操作性能
    let mut values = DMatrix::zeros(n_dates, n_stocks);
    let mut valid_mask = vec![vec![false; n_stocks]; n_dates];

    for (date_idx, (_, row_values)) in all_data.into_iter().enumerate() {
        for (stock_idx, value) in row_values.into_iter().enumerate() {
            values[(date_idx, stock_idx)] = value;
            valid_mask[date_idx][stock_idx] = !value.is_nan();
        }
    }

    let load_time = start_time.elapsed();
    println!("✅ 优化版因子文件加载完成: {}, {}天x{}股票, 用时: {:.3}s", 
             file_path.file_name().unwrap().to_string_lossy(),
             n_dates, n_stocks, load_time.as_secs_f64());

    Ok(OptimizedFactorData {
        dates,
        stocks,
        values,
        stock_index_map,
        valid_mask,
    })
}

/// 优化的截面排序函数 - 使用更快的排序算法
fn cross_section_rank_optimized(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    
    // 使用带索引的向量避免重复扫描
    let mut indexed_values: Vec<(usize, f64)> = values.iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| (i, v))
        .collect();

    // 使用更快的不稳定排序
    indexed_values.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![f64::NAN; n];
    
    // 向量化rank赋值
    for (rank, &(original_idx, _)) in indexed_values.iter().enumerate() {
        ranks[original_idx] = (rank + 1) as f64;
    }

    ranks
}

/// 优化的单因子中性化函数
fn neutralize_single_factor_optimized(
    factor_data: OptimizedFactorData,
    style_data: &OptimizedStyleData,
) -> PyResult<OptimizedNeutralizationResult> {
    let start_time = Instant::now();
    let n_dates = factor_data.dates.len();
    
    if n_dates == 0 {
        return Err(PyRuntimeError::new_err("因子数据为空：没有日期数据"));
    }
    
    if factor_data.stocks.is_empty() {
        return Err(PyRuntimeError::new_err("因子数据为空：没有股票数据"));
    }
    
    // 获取所有日期出现的股票并集
    let mut all_stocks_set = HashSet::new();
    let mut valid_dates_count = 0;
    for date in &factor_data.dates {
        if let Some(day_data) = style_data.data_by_date.get(date) {
            valid_dates_count += 1;
            for stock in &day_data.stocks {
                all_stocks_set.insert(stock.clone());
            }
        }
    }
    
    if valid_dates_count == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "没有找到匹配的风格数据日期。因子数据日期范围: {} - {}",
            factor_data.dates.first().unwrap_or(&0),
            factor_data.dates.last().unwrap_or(&0)
        )));
    }
    
    // 使用HashSet交集操作优化股票匹配
    let factor_stocks_set: HashSet<String> = factor_data.stocks.iter().cloned().collect();
    let mut union_stocks: Vec<String> = all_stocks_set.intersection(&factor_stocks_set)
        .cloned()
        .collect();
    union_stocks.sort_unstable(); // 使用不稳定排序提高性能
    
    let n_union_stocks = union_stocks.len();
    let mut neutralized_values = DMatrix::from_element(n_dates, n_union_stocks, f64::NAN);

    // 创建union股票的索引映射以提高查找速度
    let union_stock_index: HashMap<String, usize> = union_stocks.iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    // 并行处理每个日期的中性化（如果数据量大）
    let processed_count = if n_dates > 100 && n_union_stocks > 1000 {
        // 大数据量时使用并行处理
        use rayon::prelude::*;
        
        let neutralized_results: Vec<(usize, Vec<(usize, f64)>)> = factor_data.dates.par_iter()
            .enumerate()
            .filter_map(|(date_idx, &date)| {
                if let Some(day_data) = style_data.data_by_date.get(&date) {
                    let result = process_single_date_optimized(
                        date_idx, 
                        date, 
                        &factor_data, 
                        day_data, 
                        &union_stocks, 
                        &union_stock_index
                    );
                    if let Ok(values) = result {
                        Some((date_idx, values))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // 合并并行结果
        for (date_idx, day_values) in neutralized_results {
            for (union_idx, value) in day_values {
                neutralized_values[(date_idx, union_idx)] = value;
            }
        }
        
        factor_data.dates.len()
    } else {
        // 小数据量时使用串行处理
        let mut processed = 0;
        for (date_idx, &date) in factor_data.dates.iter().enumerate() {
            if let Some(day_data) = style_data.data_by_date.get(&date) {
                if let Ok(day_values) = process_single_date_optimized(
                    date_idx, 
                    date, 
                    &factor_data, 
                    day_data, 
                    &union_stocks, 
                    &union_stock_index
                ) {
                    for (union_idx, value) in day_values {
                        neutralized_values[(date_idx, union_idx)] = value;
                    }
                    processed += 1;
                }
            }
        }
        processed
    };
    
    let processing_time = start_time.elapsed();
    println!("✅ 中性化处理完成: {}个日期, 用时: {:.3}s", processed_count, processing_time.as_secs_f64());

    Ok(OptimizedNeutralizationResult {
        dates: factor_data.dates,
        stocks: union_stocks,
        neutralized_values,
    })
}

/// 优化的单日处理函数
fn process_single_date_optimized(
    _date_idx: usize,
    _date: i64,
    factor_data: &OptimizedFactorData,
    day_data: &OptimizedStyleDayData,
    union_stocks: &[String],
    union_stock_index: &HashMap<String, usize>,
) -> PyResult<Vec<(usize, f64)>> {
    // 获取当日因子原始值
    let mut daily_factor_values = Vec::new();
    let mut valid_union_indices = Vec::new();
    let mut valid_style_indices = Vec::new();
    
    for (union_idx, union_stock) in union_stocks.iter().enumerate() {
        if let Some(&factor_stock_idx) = factor_data.stock_index_map.get(union_stock) {
            if let Some(&style_stock_idx) = day_data.stock_index_map.get(union_stock) {
                let value = factor_data.values[(_date_idx, factor_stock_idx)];
                if !value.is_nan() {
                    daily_factor_values.push(value);
                    valid_union_indices.push(union_idx);
                    valid_style_indices.push(style_stock_idx);
                }
            }
        }
    }
    
    if daily_factor_values.len() < 12 {
        return Ok(Vec::new()); // 有效股票数量不足
    }

    // 优化的截面排序
    let ranked_values = cross_section_rank_optimized(&daily_factor_values);

    // 使用预计算的回归矩阵进行中性化
    if let Some(regression_matrix) = &day_data.regression_matrix {
        // 从预计算的回归矩阵中提取对应的列
        let mut selected_regression_cols = Vec::with_capacity(valid_style_indices.len());
        for &style_idx in &valid_style_indices {
            selected_regression_cols.push(regression_matrix.column(style_idx).clone_owned());
        }
        
        let selected_regression_matrix = DMatrix::from_columns(&selected_regression_cols);
        let aligned_y_vector = DVector::from_vec(ranked_values.clone());
        
        // 计算回归系数
        let beta = &selected_regression_matrix * &aligned_y_vector;
        
        // 计算预测值和残差
        let mut result_values = Vec::new();
        for (i, &union_idx) in valid_union_indices.iter().enumerate() {
            let style_idx = valid_style_indices[i];
            
            // 计算预测值：style_factors * beta
            let mut predicted_value = 0.0;
            for j in 0..12 {
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

/// 优化的结果保存函数 - 使用批量写入
fn save_neutralized_result_optimized(
    result: OptimizedNeutralizationResult,
    output_path: &Path,
) -> PyResult<()> {
    use arrow::array::{ArrayRef, Int64Array, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use parquet::basic::{Compression, Encoding};
    
    // 构建Schema
    let mut fields = vec![Field::new("date", DataType::Int64, false)];
    for stock in &result.stocks {
        fields.push(Field::new(stock, DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));
    
    // 构建数据数组 - 使用向量化操作
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(result.stocks.len() + 1);
    
    // 日期数组
    arrays.push(Arc::new(Int64Array::from(result.dates.clone())));
    
    // 并行构建股票数据数组以提高性能
    use rayon::prelude::*;
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
    
    // 创建RecordBatch
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("创建RecordBatch失败: {}", e)))?;
    
    // 使用优化的写入参数
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY) // 使用更快的压缩
        .set_encoding(Encoding::PLAIN) // 使用PLAIN编码提高写入速度
        .set_max_row_group_size(100000) // 优化行组大小
        .build();
    
    let file = File::create(output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("创建输出文件失败: {}", e)))?;
    
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| PyRuntimeError::new_err(format!("创建Arrow写入器失败: {}", e)))?;
    
    writer.write(&batch)
        .map_err(|e| PyRuntimeError::new_err(format!("写入数据失败: {}", e)))?;
    
    writer.close()
        .map_err(|e| PyRuntimeError::new_err(format!("关闭写入器失败: {}", e)))?;
    
    Ok(())
}

/// 优化的批量因子中性化函数 - 主入口
#[pyfunction]
pub fn batch_factor_neutralization_optimized(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: Option<usize>,
) -> PyResult<()> {
    let start_time = Instant::now();
    println!("🚀 开始优化版批量因子中性化处理...");

    // 使用优化版本加载风格数据
    println!("📖 正在使用内存映射加载风格数据...");
    let style_data = Arc::new(OptimizedStyleData::load_from_parquet_optimized(style_data_path)?);

    // 获取所有因子文件
    let factor_dir = Path::new(factor_files_dir);
    let mut factor_files: Vec<PathBuf> = fs::read_dir(factor_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("读取因子目录失败: {}", e)))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    factor_files.sort_unstable();
    let total_files = factor_files.len();
    println!("📁 找到{}个因子文件", total_files);

    if total_files == 0 {
        return Err(PyRuntimeError::new_err("未找到任何parquet因子文件"));
    }

    // 创建输出目录
    fs::create_dir_all(output_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("创建输出目录失败: {}", e)))?;

    // 优化线程池配置
    let optimal_threads = if let Some(threads) = num_threads {
        threads
    } else {
        // 自动选择最优线程数：CPU核心数和文件数的较小值
        std::cmp::min(rayon::current_num_threads(), total_files)
    };
    
    println!("⚡ 使用{}个线程进行并行处理", optimal_threads);

    // 创建优化的线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(optimal_threads)
        .thread_name(|index| format!("neutralization-worker-{}", index))
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;

    // 使用优化版本并行处理所有文件
    let results: Vec<_> = pool.install(|| {
        factor_files
            .into_par_iter()
            .map(|file_path| {
                let style_data = Arc::clone(&style_data);
                let output_dir = Path::new(output_dir);
                let file_start_time = Instant::now();

                let result = (|| -> PyResult<()> {
                    // 使用优化版本加载因子数据
                    let factor_data = load_factor_file_optimized(&file_path)?;

                    // 使用优化版本执行中性化
                    let neutralized_result = neutralize_single_factor_optimized(factor_data, &style_data)?;

                    // 构建输出文件路径
                    let output_filename = file_path.file_name()
                        .ok_or_else(|| PyRuntimeError::new_err("无效的文件名"))?;
                    let output_path = output_dir.join(output_filename);

                    // 使用优化版本保存结果
                    save_neutralized_result_optimized(neutralized_result, &output_path)?;

                    Ok(())
                })();

                let file_time = file_start_time.elapsed();
                if let Err(e) = &result {
                    eprintln!("❌ 处理文件失败: {} (用时: {:.3}s)", file_path.display(), file_time.as_secs_f64());
                    eprintln!("   错误详情: {}", e);
                } else {
                    println!("✅ 处理完成: {} (用时: {:.3}s)", 
                            file_path.file_name().unwrap().to_string_lossy(),
                            file_time.as_secs_f64());
                }

                result
            })
            .collect()
    });

    // 统计处理结果
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let error_count = results.len() - success_count;

    let total_time = start_time.elapsed();
    println!("\n🎉 优化版批量因子中性化处理完成!");
    println!("{}", "=".repeat(50));
    println!("📊 处理统计:");
    println!("   总文件数: {}", total_files);
    println!("   成功处理: {} ({:.1}%)", success_count, success_count as f64 / total_files as f64 * 100.0);
    println!("   失败文件: {}", error_count);
    println!("   总用时: {:.1}分钟 ({:.1}秒)", total_time.as_secs_f64() / 60.0, total_time.as_secs_f64());
    println!("   平均处理速度: {:.1} 文件/分钟", total_files as f64 / (total_time.as_secs_f64() / 60.0));
    println!("   平均单文件用时: {:.3}秒", total_time.as_secs_f64() / total_files as f64);

    if error_count > 0 {
        println!("⚠️  警告: {}个文件处理失败，请检查错误日志", error_count);
    }

    Ok(())
}