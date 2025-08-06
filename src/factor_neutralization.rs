use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use arrow::array::{Array, Float64Array, Int32Array, Int64Array, StringArray, ArrayRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

pub struct StyleData {
    pub data_by_date: HashMap<i64, StyleDayData>,
}

pub struct StyleDayData {
    pub stocks: Vec<String>,
    pub style_matrix: DMatrix<f64>,
    pub regression_matrix: Option<DMatrix<f64>>, // (X'X)^(-1)X' 预计算矩阵
}

pub struct FactorData {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    pub values: DMatrix<f64>, // dates x stocks
}

pub struct NeutralizationResult {
    pub dates: Vec<i64>,
    pub stocks: Vec<String>,
    pub neutralized_values: DMatrix<f64>,
}

impl StyleData {
    pub fn load_from_parquet(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("打开风格数据文件失败: {}", e)))?;
        
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| PyRuntimeError::new_err(format!("创建parquet读取器失败: {}", e)))?;
        
        let reader = builder
            .with_batch_size(8192)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("构建记录批次读取器失败: {}", e)))?;

        let mut all_data = Vec::new();
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| PyRuntimeError::new_err(format!("读取记录批次失败: {}", e)))?;
            all_data.push(batch);
        }

        let mut data_by_date: HashMap<i64, Vec<(String, Vec<f64>)>> = HashMap::new();

        // 处理每个批次的数据
        for batch in all_data {
            let date_column = batch.column(0);
            
            // 尝试解析为i64或i32类型
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

            // 提取11个风格因子列
            let mut style_columns = Vec::new();
            for i in 2..13 { // value_0 到 value_10
                let col = batch.column(i)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| PyRuntimeError::new_err(format!("风格因子列{}类型错误", i-2)))?;
                style_columns.push(col);
            }

            // 按日期分组数据
            for row_idx in 0..batch.num_rows() {
                let date = batch_dates[row_idx];
                let stock = stock_array.value(row_idx).to_string();
                
                let mut style_values = Vec::new();
                for col in &style_columns {
                    let value = if col.is_null(row_idx) {
                        f64::NAN
                    } else {
                        col.value(row_idx)
                    };
                    style_values.push(value);
                }
                
                data_by_date.entry(date)
                    .or_insert_with(Vec::new)
                    .push((stock, style_values));
            }
        }

        // 转换为最终的数据结构
        let mut final_data_by_date = HashMap::new();
        for (date, stock_data) in data_by_date {
            let n_stocks = stock_data.len();
            if n_stocks < 12 {
                println!("警告: 日期{}的股票数量({})少于12只，跳过该日期", date, n_stocks);
                continue;
            }

            let mut stocks = Vec::new();
            let mut style_matrix = DMatrix::zeros(n_stocks, 12);

            for (i, (stock, style_values)) in stock_data.into_iter().enumerate() {
                stocks.push(stock);
                for j in 0..11 {
                    style_matrix[(i, j)] = style_values[j];
                }
                style_matrix[(i, 11)] = 1.0; // 截距项
            }

            // 预计算回归矩阵
            let regression_matrix = compute_regression_matrix(&style_matrix)?;

            let day_data = StyleDayData {
                stocks,
                style_matrix,
                regression_matrix: Some(regression_matrix),
            };

            final_data_by_date.insert(date, day_data);
        }

        if final_data_by_date.is_empty() {
            return Err(PyRuntimeError::new_err("风格数据为空或所有日期的股票数量都少于12只"));
        }

        println!("成功加载风格数据: {}个交易日", final_data_by_date.len());
        Ok(StyleData { data_by_date: final_data_by_date })
    }
}

fn compute_regression_matrix(style_matrix: &DMatrix<f64>) -> PyResult<DMatrix<f64>> {
    let xt = style_matrix.transpose();
    let xtx = &xt * style_matrix;
    
    let xtx_inv = xtx.try_inverse()
        .ok_or_else(|| PyRuntimeError::new_err("风格因子矩阵不可逆，可能存在多重共线性"))?;
    
    Ok(xtx_inv * xt)
}

fn load_factor_file(file_path: &Path) -> PyResult<FactorData> {
    let file = File::open(file_path)
        .map_err(|e| PyRuntimeError::new_err(format!("打开因子文件失败 {}: {}", file_path.display(), e)))?;
    
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyRuntimeError::new_err(format!("创建parquet读取器失败: {}", e)))?;
    
    let reader = builder
        .with_batch_size(8192)
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

    // 从第一个批次获取列名和结构
    let schema = all_batches[0].schema();
    let mut dates = Vec::new();
    
    // 检查是否最后一列是date列（pandas索引列）
    let total_columns = schema.fields().len();
    let last_field = &schema.fields()[total_columns - 1];
    
    let (date_col_idx, stocks) = if last_field.name() == "date" {
        // 最后一列是date列，前面的都是股票列
        let stocks: Vec<String> = schema.fields()
            .iter()
            .take(total_columns - 1) // 排除最后的date列
            .map(|f| f.name().clone())
            .collect();
        (total_columns - 1, stocks)
    } else {
        // 传统格式：第一列是date，其余是股票列
        let stocks: Vec<String> = schema.fields()
            .iter()
            .skip(1) // 跳过第一列的date列
            .map(|f| f.name().clone())
            .collect();
        (0, stocks)
    };

    let n_stocks = stocks.len();
    let mut all_data: Vec<(i64, Vec<f64>)> = Vec::new();
    
    // 预先构建股票名称到列索引的映射，避免重复搜索
    let mut stock_col_map = std::collections::HashMap::new();
    for (stock_idx, stock) in stocks.iter().enumerate() {
        if let Some(col_idx) = schema.fields()
            .iter()
            .position(|f| f.name() == stock) {
            stock_col_map.insert(stock_idx, col_idx);
        }
    }

    // 处理所有批次
    for (batch_idx, batch) in all_batches.iter().enumerate() {
        let date_column = batch.column(date_col_idx);
        
        // 尝试解析为i64或i32类型
        let batch_dates: Vec<i64> = if let Some(date_array_i64) = date_column.as_any().downcast_ref::<Int64Array>() {
            (0..date_array_i64.len()).map(|i| date_array_i64.value(i)).collect()
        } else if let Some(date_array_i32) = date_column.as_any().downcast_ref::<Int32Array>() {
            (0..date_array_i32.len()).map(|i| date_array_i32.value(i) as i64).collect()
        } else {
            return Err(PyRuntimeError::new_err("日期列类型错误：期望Int64或Int32类型"));
        };

        let num_rows = batch.num_rows();
        
        // 预先获取所有相关列的数组引用
        let mut stock_arrays = Vec::new();
        for (stock_idx, &col_idx) in stock_col_map.iter() {
            let array = batch.column(col_idx);
            if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                stock_arrays.push((*stock_idx, float_array));
            }
        }
        
        for row_idx in 0..num_rows {
            let date = batch_dates[row_idx];
            let mut row_values = vec![f64::NAN; n_stocks];

            // 使用预先获取的数组引用
            for &(stock_idx, float_array) in &stock_arrays {
                if !float_array.is_null(row_idx) {
                    row_values[stock_idx] = float_array.value(row_idx);
                }
            }

            all_data.push((date, row_values));
            dates.push(date);
        }
    }

    let n_dates = dates.len();
    let mut values = DMatrix::zeros(n_dates, n_stocks);

    for (date_idx, (_, row_values)) in all_data.into_iter().enumerate() {
        for (stock_idx, value) in row_values.into_iter().enumerate() {
            values[(date_idx, stock_idx)] = value;
        }
    }

    Ok(FactorData {
        dates,
        stocks,
        values,
    })
}

fn cross_section_rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed_values: Vec<(usize, f64)> = values.iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| (i, v))
        .collect();

    indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![f64::NAN; n];
    for (rank, (original_idx, _)) in indexed_values.iter().enumerate() {
        ranks[*original_idx] = (rank + 1) as f64;
    }

    ranks
}

fn neutralize_single_factor(
    factor_data: FactorData,
    style_data: &StyleData,
) -> PyResult<NeutralizationResult> {
    let n_dates = factor_data.dates.len();
    let n_stocks = factor_data.stocks.len();
    
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
    
    // 与因子数据的股票取交集，确保风格数据中有对应股票
    let mut union_stocks: Vec<String> = all_stocks_set.intersection(
        &factor_data.stocks.iter().cloned().collect()
    ).cloned().collect();
    union_stocks.sort();
    
    let n_union_stocks = union_stocks.len();
    let mut neutralized_values = DMatrix::from_element(n_dates, n_union_stocks, f64::NAN);

    // 为每个日期处理因子中性化
    let mut processed_dates = 0;
    for (date_idx, &date) in factor_data.dates.iter().enumerate() {
        if let Some(day_data) = style_data.data_by_date.get(&date) {
            // 获取当日因子原始值
            let mut daily_factor_values = Vec::new();
            let mut valid_stock_indices = Vec::new();
            
            for (union_idx, union_stock) in union_stocks.iter().enumerate() {
                if let Some(factor_stock_idx) = factor_data.stocks.iter().position(|s| s == union_stock) {
                    let value = factor_data.values[(date_idx, factor_stock_idx)];
                    if !value.is_nan() {
                        daily_factor_values.push(value);
                        valid_stock_indices.push(union_idx);
                    }
                }
            }
            
            if daily_factor_values.len() < 12 {
                continue; // 有效股票数量不足，跳过该日期
            }

            // 截面排序
            let ranked_values = cross_section_rank(&daily_factor_values);

            // 构建回归用的因子向量和风格矩阵
            let mut regression_y = Vec::new();
            let mut regression_x_rows = Vec::new();
            
            for (i, &union_idx) in valid_stock_indices.iter().enumerate() {
                let union_stock = &union_stocks[union_idx];
                if let Some(style_stock_idx) = day_data.stocks.iter().position(|s| s == union_stock) {
                    regression_y.push(ranked_values[i]);
                    
                    // 获取该股票的风格因子行
                    let mut style_row = Vec::new();
                    for j in 0..12 {
                        style_row.push(day_data.style_matrix[(style_stock_idx, j)]);
                    }
                    regression_x_rows.push(style_row);
                }
            }
            
            if regression_y.len() < 12 {
                continue;
            }

            // 执行回归中性化，使用预计算的回归矩阵
            let y_vector = DVector::from_vec(regression_y.clone());
            
            if let Some(regression_matrix) = &day_data.regression_matrix {
                // 关键：regression_y 和 valid_stock_indices 都已经经过了风格数据的筛选
                // regression_y中的每个元素都对应一个在风格数据中找到的股票
                // 我们需要重新收集style_stock_indices来匹配regression_y的顺序
                
                let mut style_stock_indices = Vec::new();
                
                for (i, &union_idx) in valid_stock_indices.iter().enumerate() {
                    let union_stock = &union_stocks[union_idx];
                    if let Some(style_stock_idx) = day_data.stocks.iter().position(|s| s == union_stock) {
                        style_stock_indices.push(style_stock_idx);
                    }
                }
                
                // 检查数据一致性
                if style_stock_indices.len() != regression_y.len() {
                    println!("警告: 日期{}的股票索引长度({})与回归数据长度({})不匹配，跳过处理", 
                        date, style_stock_indices.len(), regression_y.len());
                    continue;
                }
                
                let aligned_y_values = regression_y.clone();
                
                // 从预计算的回归矩阵中提取对应的列
                // regression_matrix: (12, n_style_stocks), 我们要提取特定的列
                let mut selected_regression_cols = Vec::new();
                for &style_idx in &style_stock_indices {
                    let col = regression_matrix.column(style_idx);
                    selected_regression_cols.push(col.clone_owned());
                }
                
                // 构建适用于当前有效股票的回归矩阵
                let selected_regression_matrix = DMatrix::from_columns(&selected_regression_cols);
                let aligned_y_vector = DVector::from_vec(aligned_y_values);
                
                // 现在矩阵维度应该匹配：(12, n_valid) * (n_valid,) = (12,)
                let beta = &selected_regression_matrix * &aligned_y_vector;
                
                // 计算预测值和残差
                let mut predicted_values = Vec::new();
                for (i, &union_idx) in valid_stock_indices.iter().enumerate() {
                    let union_stock = &union_stocks[union_idx];
                    if let Some(style_stock_idx) = day_data.stocks.iter().position(|s| s == union_stock) {
                        // 计算该股票的预测值：style_factors * beta
                        let mut predicted_value = 0.0;
                        for j in 0..12 {
                            predicted_value += day_data.style_matrix[(style_stock_idx, j)] * beta[j];
                        }
                        predicted_values.push(predicted_value);
                    }
                }
                
                // 计算残差
                for (i, &union_idx) in valid_stock_indices.iter().enumerate() {
                    if i < predicted_values.len() {
                        let residual = ranked_values[i] - predicted_values[i];
                        neutralized_values[(date_idx, union_idx)] = residual;
                    }
                }
            }
            
            processed_dates += 1;
        }
    }

    Ok(NeutralizationResult {
        dates: factor_data.dates,
        stocks: union_stocks,
        neutralized_values,
    })
}

fn save_neutralized_result(
    result: NeutralizationResult,
    output_path: &Path,
) -> PyResult<()> {
    use arrow::array::{ArrayRef, Int64Array, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    
    // 构建Schema
    let mut fields = vec![Field::new("date", DataType::Int64, false)];
    for stock in &result.stocks {
        fields.push(Field::new(stock, DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));
    
    // 构建数据数组
    let mut arrays: Vec<ArrayRef> = Vec::new();
    
    // 日期数组
    let date_array: ArrayRef = Arc::new(Int64Array::from(result.dates.clone()));
    arrays.push(date_array);
    
    // 股票数据数组
    for (stock_idx, _stock) in result.stocks.iter().enumerate() {
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
        let stock_array: ArrayRef = Arc::new(Float64Array::from(column_data));
        arrays.push(stock_array);
    }
    
    // 创建RecordBatch
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("创建RecordBatch失败: {}", e)))?;
    
    // 写入parquet文件
    let file = File::create(output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("创建输出文件失败: {}", e)))?;
    
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| PyRuntimeError::new_err(format!("创建Arrow写入器失败: {}", e)))?;
    
    writer.write(&batch)
        .map_err(|e| PyRuntimeError::new_err(format!("写入数据失败: {}", e)))?;
    
    writer.close()
        .map_err(|e| PyRuntimeError::new_err(format!("关闭写入器失败: {}", e)))?;
    
    Ok(())
}

struct ProgressMonitor {
    total: usize,
    completed: usize,
    start_time: Instant,
}

impl ProgressMonitor {
    fn new(total: usize) -> Self {
        Self {
            total,
            completed: 0,
            start_time: Instant::now(),
        }
    }

    fn update(&mut self) {
        self.completed += 1;
        self.display_progress(self.completed);
    }
    
    fn display_progress(&self, current_completed: usize) {
        let elapsed = self.start_time.elapsed().as_secs().max(1); // 避免除零
        let rate = current_completed as f64 / elapsed as f64;
        let remaining = if current_completed < self.total {
            ((self.total - current_completed) as f64 / rate) as u64
        } else {
            0
        };
        
        println!(
            "进度: {}/{} ({:.1}%) | 已用时: {}s | 预计剩余: {}s | 处理速度: {:.1} files/s",
            current_completed,
            self.total,
            (current_completed as f64 / self.total as f64) * 100.0,
            elapsed,
            remaining,
            rate
        );
    }
}

#[pyfunction]
pub fn batch_factor_neutralization(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: Option<usize>,
) -> PyResult<()> {
    let start_time = Instant::now();
    println!("开始批量因子中性化处理...");

    // 加载风格数据
    println!("正在加载风格数据...");
    let style_data = Arc::new(StyleData::load_from_parquet(style_data_path)?);
    println!("风格数据加载完成");

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

    factor_files.sort();
    let total_files = factor_files.len();
    println!("找到{}个因子文件", total_files);

    if total_files == 0 {
        return Err(PyRuntimeError::new_err("未找到任何parquet因子文件"));
    }

    // 创建输出目录
    fs::create_dir_all(output_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("创建输出目录失败: {}", e)))?;

    // 设置线程数
    let threads = num_threads.unwrap_or(rayon::current_num_threads());
    println!("指定使用{}个线程进行并行处理", threads);
    
    // 显示当前系统信息
    println!("系统默认rayon线程数: {}", rayon::current_num_threads());
    println!("系统CPU核心数: {}", std::thread::available_parallelism().unwrap().get());

    // 创建进度监控
    let progress_counter = Arc::new(Mutex::new(0usize));
    let (tx, rx) = mpsc::channel();

    // 启动进度监控线程
    let progress_counter_clone = Arc::clone(&progress_counter);
    let progress_handle = thread::spawn(move || {
        let monitor = ProgressMonitor::new(total_files);
        while let Ok(()) = rx.recv_timeout(Duration::from_millis(500)) {
            let current_count = *progress_counter_clone.lock().unwrap();
            monitor.display_progress(current_count);
        }
        // 打印最终进度
        let final_count = *progress_counter_clone.lock().unwrap();
        monitor.display_progress(final_count);
    });

    // 创建受控的线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;
    
    println!("已创建{}线程的rayon线程池", threads);

    // 并行处理所有文件
    let results: Vec<_> = pool.install(|| {
        factor_files
            .into_par_iter()
            .map(|file_path| {
            let style_data = Arc::clone(&style_data);
            let output_dir = Path::new(output_dir);
            let tx = tx.clone();
            let file_path_str = file_path.display().to_string(); // 保存路径字符串用于错误输出
            let file_path_str_clone = file_path_str.clone(); // 为闭包创建副本

            let result = (move || -> PyResult<()> {
                // 加载因子数据
                let factor_data = load_factor_file(&file_path)
                    .map_err(|e| PyRuntimeError::new_err(format!("加载因子文件失败: {}", e)))?;

                // 执行中性化
                let neutralized_result = neutralize_single_factor(factor_data, &style_data)
                    .map_err(|e| PyRuntimeError::new_err(format!("中性化处理失败: {}", e)))?;

                // 构建输出文件路径
                let output_filename = file_path.file_name()
                    .ok_or_else(|| PyRuntimeError::new_err("无效的文件名"))?;
                let output_path = output_dir.join(output_filename);

                // 保存结果
                save_neutralized_result(neutralized_result, &output_path)
                    .map_err(|e| PyRuntimeError::new_err(format!("保存结果失败: {}", e)))?;

                Ok(())
            })();

            // 更新进度
            {
                let mut counter = progress_counter.lock().unwrap();
                *counter += 1;
                let _ = tx.send(()); // 通知进度监控线程
            }

            // 处理错误
            if let Err(e) = &result {
                eprintln!("❌ 处理文件失败: {}", file_path_str);
                eprintln!("   错误详情: {}", e);
                eprintln!("   错误类型: {:?}", e);
            }

            result
        })
        .collect()
    });

    // 关闭进度监控
    drop(tx);
    progress_handle.join().unwrap();

    // 统计处理结果
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let error_count = results.len() - success_count;

    let total_time = start_time.elapsed();
    println!("\n=== 批量因子中性化处理完成 ===");
    println!("总文件数: {}", total_files);
    println!("成功处理: {}", success_count);
    println!("失败文件: {}", error_count);
    println!("总用时: {:.1}分钟", total_time.as_secs_f64() / 60.0);
    println!("平均处理速度: {:.1} 文件/分钟", total_files as f64 / (total_time.as_secs_f64() / 60.0));

    if error_count > 0 {
        println!("警告: {}个文件处理失败，请检查错误日志", error_count);
    }

    Ok(())
}