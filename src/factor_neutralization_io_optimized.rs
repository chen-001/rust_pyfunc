use arrow::array::{Array, Float64Array, Int32Array, Int64Array};
use chrono::Local;
use hdf5_metno as hdf5;
use nalgebra::{DMatrix, DVector};
use ndarray::s;
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
    /// 仅含前10列风格因子(value_0~9)的回归矩阵，用于关闭行业中性化场景
    pub regression_matrix_style_only: Option<Arc<DMatrix<f64>>>,
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
    /// SzBa 风格字段 → 风格矩阵前 10 列（value_0~value_9）的固定列序。
    ///
    /// 契约写死，不得调整：`factor_neutralize_std.rs` 里 `barra_list[2]` 被当作 size，
    /// 所以 `value_2` 必须是 `size.h5`。第 11 列起是 industry 的 one-hot `ind_1~ind_31`。
    const SZBA_STYLE_FIELDS: [&'static str; 10] = [
        "residual_volatility", // value_0
        "book_to_price",       // value_1
        "size",                // value_2
        "momentum",            // value_3
        "leverage",            // value_4
        "earnings_yield",      // value_5
        "growth",              // value_6
        "liquidity",           // value_7
        "beta",                // value_8
        "non_linear_size",     // value_9
    ];

    /// 直接从 `{vars_root}/SzBa/` 的 H5 加载风格数据（不再读 barra parquet）。
    ///
    /// `vars_root` 是 vars 根目录（默认 `/ssd_data/data/vars`），其下 `SzBa/` 有：
    /// - `calendar_map.csv`：第 1 行表头，之后每行一个交易日；第 r 行 = h5 第 r 行；
    /// - `symbol_map.csv`：`symbol,pos`；第 c 行 = h5 第 c 列；
    /// - `{字段}.h5`：dataset 名固定 `data`，shape (7000, 8000)，float64，NaN = 无数据。
    ///
    /// 列序按 `SZBA_STYLE_FIELDS` 写死，`value_2` = size；`ind_1~ind_31` 是 industry 的
    /// one-hot（行业号 k → ind_k = 1，行和恒为 1，已含截距，不再加常数项）。
    ///
    /// 行集口径：某天某只股票只要 10 个风格值全部有限就进入该天矩阵。industry 为 NaN 或
    /// 不在 1~31 时按「行业未知」处理 —— 31 个 ind 列全置 0，这只股票仍然进矩阵（旧
    /// parquet 就是这么编码的：当年 join 不到就 fill_null(0.0)），丢掉这些行会改变当天
    /// 截面内的排名，连带改变留下来的股票的残差。
    ///
    /// 与旧 parquet 路径唯一的行为差别：某个风格值是 NaN 时只把这只股票从当天排除，
    /// 而不是像旧路径那样让 `compute_regression_matrix_io_optimized` 的 X'X 求逆失败、
    /// 把整天静默丢掉。
    pub fn load_from_vars_h5(vars_root: &str) -> PyResult<Self> {
        let start_time = Instant::now();
        println!("🔄 开始从 H5 加载风格数据: {}/SzBa", vars_root);

        let szba = Path::new(vars_root).join("SzBa");
        let cal_path = szba.join("calendar_map.csv");
        let cal_text = fs::read_to_string(&cal_path).map_err(|e| {
            PyRuntimeError::new_err(format!("读取 {} 失败: {}", cal_path.display(), e))
        })?;
        let dates: Vec<i64> = cal_text
            .lines()
            .skip(1)
            .filter_map(|l| l.trim().parse::<i64>().ok())
            .collect();
        if dates.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "{} 里没有交易日",
                cal_path.display()
            )));
        }

        let sym_path = szba.join("symbol_map.csv");
        let sym_text = fs::read_to_string(&sym_path).map_err(|e| {
            PyRuntimeError::new_err(format!("读取 {} 失败: {}", sym_path.display(), e))
        })?;
        let codes: Vec<String> = sym_text
            .lines()
            .skip(1)
            .map(|l| l.split(',').next().unwrap_or("").trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        if codes.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "{} 里没有股票",
                sym_path.display()
            )));
        }
        let n_codes = codes.len();

        // 11 个 h5 一次性打开（10 个风格字段 + industry），dataset 名固定 "data"。
        // 逐块顺序读取，HDF5 C 库只在主线程被调用，不用考虑线程安全。
        let mut files: Vec<(&'static str, hdf5::File, hdf5::Dataset)> = Vec::with_capacity(11);
        for field in Self::SZBA_STYLE_FIELDS
            .iter()
            .copied()
            .chain(std::iter::once("industry"))
        {
            let path = szba.join(format!("{}.h5", field));
            let file = hdf5::File::open(&path).map_err(|e| {
                PyRuntimeError::new_err(format!("打开 {} 失败: {}", path.display(), e))
            })?;
            let ds = file.dataset("data").map_err(|e| {
                PyRuntimeError::new_err(format!("{} 里没有 dataset data: {}", path.display(), e))
            })?;
            files.push((field, file, ds));
        }

        let n_dates = dates.len();
        let mut data_by_date: HashMap<i64, IOOptimizedStyleDayData> = HashMap::with_capacity(n_dates);
        let mut total_rows = 0usize;
        let mut dropped_days = 0usize;
        let mut empty_days = 0usize;
        let mut unknown_industry = 0usize;

        // 逐块读盘：一块 BLOCK_ROWS 行 × 11 个字段常驻内存（64×5789×8×11 ≈ 33MB），
        // 块内按日并行构造 41 列矩阵，避免把上千万行中间结果全留在内存里。
        const BLOCK_ROWS: usize = 64;
        let mut r0 = 0usize;
        while r0 < n_dates {
            let r1 = (r0 + BLOCK_ROWS).min(n_dates);
            let mut flats: Vec<Vec<f64>> = Vec::with_capacity(11);
            for (field, _file, ds) in &files {
                let arr: ndarray::Array2<f64> =
                    ds.read_slice_2d(s![r0..r1, 0..n_codes]).map_err(|e| {
                        PyRuntimeError::new_err(format!("读取 {} 的 {} 失败: {}", field, r0, e))
                    })?;
                flats.push(arr.iter().copied().collect());
            }

            let day_results: Vec<(i64, usize, usize, Option<PyResult<IOOptimizedStyleDayData>>)> =
                (r0..r1)
                    .into_par_iter()
                    .map(|r| {
                        let base = (r - r0) * n_codes;
                        let mut stock_data: Vec<(String, Vec<f64>)> = Vec::new();
                        let mut unknown_ind = 0usize;
                        for c in 0..n_codes {
                            let mut style = [0.0f64; 10];
                            let mut complete = true;
                            for k in 0..10 {
                                let v = flats[k][base + c];
                                if !v.is_finite() {
                                    complete = false;
                                    break;
                                }
                                style[k] = v;
                            }
                            if !complete {
                                continue;
                            }
                            // industry 为 NaN 或不在 1~31 时按「行业未知」处理：31 个 ind 列
                            // 全置 0，这只股票仍然进矩阵 —— 旧 parquet 就是这么编码的
                            // （当年 join 不到就 fill_null(0.0)），丢掉这些行会改变当天截面排名。
                            let mut row = vec![0.0f64; 41];
                            row[..10].copy_from_slice(&style);
                            let ind = flats[10][base + c];
                            let k = ind as usize;
                            if ind.is_finite() && (1..=31).contains(&k) {
                                row[10 + k - 1] = 1.0;
                            } else {
                                unknown_ind += 1;
                            }
                            stock_data.push((codes[c].clone(), row));
                        }
                        let n = stock_data.len();
                        if n < 12 {
                            return (dates[r], n, unknown_ind, None);
                        }
                        (
                            dates[r],
                            n,
                            unknown_ind,
                            Some(Self::convert_date_data_optimized(dates[r], stock_data)),
                        )
                    })
                    .collect();

            for (date, n, unknown, result) in day_results {
                unknown_industry += unknown;
                match result {
                    Some(Ok(day)) => {
                        total_rows += n;
                        data_by_date.insert(date, day);
                    }
                    Some(Err(e)) => {
                        println!("⚠️ 日期 {} 风格矩阵构造失败，跳过: {}", date, e);
                        dropped_days += 1;
                    }
                    None => {
                        if n == 0 {
                            empty_days += 1;
                        } else {
                            dropped_days += 1;
                        }
                    }
                }
            }
            r0 = r1;
        }

        if data_by_date.is_empty() {
            return Err(PyRuntimeError::new_err(
                "风格数据为空或所有日期的完整行都不足12只",
            ));
        }

        println!("✅ H5 风格数据加载完成!");
        println!(
            "   📊 统计: {}个交易日, {}只股票",
            data_by_date.len(),
            total_rows
        );
        println!(
            "   📊 风格矩阵: 41列 (value_0~9 = {}), 股票池 {} 只",
            Self::SZBA_STYLE_FIELDS.join(","),
            n_codes
        );
        if dropped_days > 0 {
            println!(
                "   ⚠️ 有完整行但不足12只、被跳过的日期: {} 天（风格矩阵会退化，不参与中性化）",
                dropped_days
            );
        }
        if empty_days > 0 {
            println!(
                "   ℹ️ 全天无任何完整行的日期: {} 天（H5 日历含未来行与早期无数据行）",
                empty_days
            );
        }
        if unknown_industry > 0 {
            println!(
                "   ℹ️ industry 未知（NaN 或不在 1~31）、31 个 ind 列全置 0 的行: {}",
                unknown_industry
            );
        }
        println!("   ⏱️  总耗时: {:.3}s", start_time.elapsed().as_secs_f64());

        Ok(IOOptimizedStyleData {
            data_by_date,
            file_cache: Arc::new(Mutex::new(Vec::new())),
        })
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

        // 预计算仅风格因子(前10列 value_0~9)的回归矩阵，用于关闭行业中性化场景
        let style_only_matrix = style_matrix.columns(0, 10).into_owned();
        let regression_matrix_style_only =
            compute_regression_matrix_io_optimized(&style_only_matrix)?;

        Ok(IOOptimizedStyleDayData {
            stocks,
            style_matrix,
            regression_matrix: Some(Arc::new(regression_matrix)),
            regression_matrix_style_only: Some(Arc::new(regression_matrix_style_only)),
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
    style_vars_dir: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: Option<usize>,
    log_detailed: Option<bool>,
) -> PyResult<()> {
    let start_time = Instant::now();
    println!("🚀 开始I/O优化版批量因子中性化处理...");

    // 使用I/O优化版本加载风格数据
    println!("📖 正在使用I/O优化加载风格数据...");
    let style_data = Arc::new(IOOptimizedStyleData::load_from_vars_h5(
        style_vars_dir,
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
