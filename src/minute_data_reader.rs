//! 分钟数据读取器：从 HDF5 文件按日读取全市场分钟数据。
//!
//! 数据布局：`/ssd_data/data/1min_factor_text/{field}.h5`
//!   - dataset "data": shape (total_minutes, n_stocks), dtype=float64, C-order
//!   - total_minutes = n_trading_days × 240（每天 240 根 1min K 线）
//!   - n_stocks ≤ 7000（按 symbol_map.csv 的 pos 索引列）
//!
//! 元数据文件（同目录）：
//!   - calendar_map.csv: 单列 date_min，行号 = day_idx
//!   - symbol_map.csv: 两列 symbol,pos，pos = 列号
//!
//! 读取方式：hyperslab `s![day_idx*240 .. (day_idx+1)*240, ..]`
//!   - 行切片连续，C-order 下 IO 高效（~20-30ms/字段/天）
//!   - 每天最后一个 day_idx 的行数可能 < 240（数据未对齐），按实际 shape 截断

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::LazyLock;

#[cfg(feature = "hdf5")]
use hdf5_metno as hdf5;

const MIN_PER_DAY: usize = 240;
const DATA_DIR: &str = "/ssd_data/data/1min_factor_text";

// ============================================================
// 元数据（全局懒加载，线程安全）
// ============================================================

struct MinuteMeta {
    /// date_int → day_idx（day_idx * 240 = HDF5 行偏移）
    date_to_dayidx: HashMap<i64, usize>,
    /// pos → stock_code（列号 → 股票代码）
    col_to_code: Vec<String>,
    /// stock_code → pos（股票代码 → 列号）
    code_to_col: HashMap<String, usize>,
    /// 实际总行数（用于安全截断最后一天）
    total_rows: usize,
}

static META: LazyLock<io::Result<MinuteMeta>> = LazyLock::new(load_meta);

fn get_meta() -> io::Result<&'static MinuteMeta> {
    META.as_ref()
        .map_err(|e| io::Error::new(e.kind(), format!("加载分钟数据元数据失败: {e}")))
}

fn load_meta() -> io::Result<MinuteMeta> {
    let dir = Path::new(DATA_DIR);

    // calendar_map.csv：单列 date_min，行号 = day_idx
    let cal_path = dir.join("calendar_map.csv");
    let cal_content = fs::read_to_string(&cal_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("读取 calendar_map.csv 失败 {}: {e}", cal_path.display()),
        )
    })?;
    let mut date_to_dayidx = HashMap::new();
    let mut day_idx = 0usize;
    for line in cal_content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // 跳过表头（非数字行），成功 parse 的才递增 day_idx
        if let Ok(date) = line.parse::<i64>() {
            date_to_dayidx.insert(date, day_idx);
            day_idx += 1;
        }
    }

    // symbol_map.csv：两列 symbol,pos
    let sym_path = dir.join("symbol_map.csv");
    let sym_content = fs::read_to_string(&sym_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("读取 symbol_map.csv 失败 {}: {e}", sym_path.display()),
        )
    })?;
    let mut col_to_code: Vec<(usize, String)> = Vec::new();
    let mut max_pos = 0usize;
    for line in sym_content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        // 跳过表头（symbol/pos 等非数字）
        let pos = match parts[1].trim().parse::<usize>() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let symbol = parts[0].trim().to_string();
        if pos > max_pos {
            max_pos = pos;
        }
        col_to_code.push((pos, symbol));
    }
    let n_cols = max_pos + 1;
    let mut col_arr = vec![String::new(); n_cols];
    let mut code_to_col = HashMap::with_capacity(col_to_code.len());
    for (pos, symbol) in &col_to_code {
        col_arr[*pos] = symbol.clone();
        code_to_col.insert(symbol.clone(), *pos);
    }

    // 读总行数：打开任意一个 .h5 文件拿 shape
    let total_rows = get_total_rows().unwrap_or(0);

    Ok(MinuteMeta {
        date_to_dayidx,
        col_to_code: col_arr,
        code_to_col,
        total_rows,
    })
}

#[cfg(feature = "hdf5")]
fn get_total_rows() -> Option<usize> {
    let path = format!("{DATA_DIR}/close.h5");
    let file = hdf5::File::open(&path).ok()?;
    let ds = file.dataset("data").ok()?;
    let shape = ds.shape();
    shape.into_iter().next()
}

#[cfg(not(feature = "hdf5"))]
fn get_total_rows() -> Option<usize> {
    None
}

// ============================================================
// HDF5 文件句柄缓存（线程局部）
// ============================================================

/// 多进程模式下每个 worker 是独立进程，各线程打开各自的 HDF5 文件句柄。
/// 用 thread_local 避免 HDF5 C 库的线程安全问题。
#[cfg(feature = "hdf5")]
thread_local! {
    static H5_CACHE: std::cell::RefCell<HashMap<String, hdf5::File>> =
        std::cell::RefCell::new(HashMap::new());
}

// ============================================================
// 公开 API
// ============================================================

/// 读单字段单日的分钟数据 → (codes, data)
///
/// - `field`: 字段名，如 "close"、"volume"、"act_buy_amount_sum"
/// - `date`: 8位整数日期，如 20220819
///
/// 返回：
/// - `codes`: 当天有效的股票代码列表（NaN 全排除后的列）
/// - `data`: shape (n_minutes, n_valid_stocks) 的二维数组
///
/// 注意：返回的是**全部列**（全部 symbol_map 中的股票），不做 NaN 列过滤。
/// 调用方按需自行过滤 NaN。这样可以保持列索引与 symbol_map 一致。
#[cfg(feature = "hdf5")]
pub fn read_minute_field(
    field: &str,
    date: i64,
) -> io::Result<(Vec<String>, ndarray::Array2<f64>)> {
    use ndarray::s;

    let meta = get_meta()?;

    let day_idx = *meta.date_to_dayidx.get(&date).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("日期 {date} 不在 calendar_map 中"),
        )
    })?;

    let row_start = day_idx * MIN_PER_DAY;
    // 最后一天可能不足 240 行，按 total_rows 截断
    let row_end = if meta.total_rows > 0 {
        std::cmp::min(row_start + MIN_PER_DAY, meta.total_rows)
    } else {
        row_start + MIN_PER_DAY
    };

    let h5_path = format!("{DATA_DIR}/{field}.h5");

    let data = H5_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let file = cache.entry(field.to_string()).or_insert_with(|| {
            hdf5::File::open(&h5_path).unwrap_or_else(|e| panic!("打开 {h5_path} 失败: {e}"))
        });
        let dataset = file.dataset("data").map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("读取 dataset 失败: {e}"),
            )
        })?;

        // hyperslab: 读取 day_idx 的 240 行 × 全部列
        let arr: ndarray::Array2<f64> =
            dataset
                .read_slice_2d(s![row_start..row_end, ..])
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("hyperslab 读取失败: {e}"),
                    )
                })?;
        io::Result::Ok(arr)
    })?;

    // 最后4根K线置为NaN（14:57-15:00 集合竞价阶段，与 Python read_minute_data 一致）
    let n_rows = data.nrows();
    if n_rows >= 4 {
        let mut data = data;
        for i in (n_rows - 4)..n_rows {
            for j in 0..data.ncols() {
                data[(i, j)] = f64::NAN;
            }
        }
        return Ok((meta.col_to_code.clone(), data));
    }

    Ok((meta.col_to_code.clone(), data))
}

#[cfg(not(feature = "hdf5"))]
pub fn read_minute_field(
    _field: &str,
    _date: i64,
) -> io::Result<(Vec<String>, ndarray::Array2<f64>)> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "HDF5 支持未启用（编译时未包含 hdf5-metno）",
    ))
}

/// 读多字段单日 → Vec<(codes, data)>，每字段一个 (codes, Array2)
///
/// 比逐字段调用更高效：共享同一天的 day_idx 计算。
#[cfg(feature = "hdf5")]
pub fn read_minute_fields(
    fields: &[&str],
    date: i64,
) -> io::Result<Vec<(Vec<String>, ndarray::Array2<f64>)>> {
    let mut results = Vec::with_capacity(fields.len());
    for &field in fields {
        results.push(read_minute_field(field, date)?);
    }
    Ok(results)
}

/// 读单字段多日 → Array3 (n_days, n_minutes, n_stocks)
///
/// 用于需要跨日数据的因子计算（如 rolling）。
#[cfg(feature = "hdf5")]
pub fn read_minute_field_multi_day(
    field: &str,
    dates: &[i64],
) -> io::Result<(Vec<String>, ndarray::Array3<f64>)> {
    use ndarray::s;

    let meta = get_meta()?;

    if dates.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dates 不能为空",
        ));
    }

    // 计算连续行范围
    let mut day_indices = Vec::with_capacity(dates.len());
    for &date in dates {
        let day_idx = *meta.date_to_dayidx.get(&date).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("日期 {date} 不在 calendar_map 中"),
            )
        })?;
        day_indices.push(day_idx);
    }

    let h5_path = format!("{DATA_DIR}/{field}.h5");
    let n_stocks = meta.col_to_code.len();

    let mut out = ndarray::Array3::<f64>::from_elem((dates.len(), MIN_PER_DAY, n_stocks), f64::NAN);

    H5_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let file = cache.entry(field.to_string()).or_insert_with(|| {
            hdf5::File::open(&h5_path).unwrap_or_else(|e| panic!("打开 {h5_path} 失败: {e}"))
        });
        let dataset = file.dataset("data").map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("读取 dataset 失败: {e}"),
            )
        })?;

        for (day_offset, &day_idx) in day_indices.iter().enumerate() {
            let row_start = day_idx * MIN_PER_DAY;
            let row_end = if meta.total_rows > 0 {
                std::cmp::min(row_start + MIN_PER_DAY, meta.total_rows)
            } else {
                row_start + MIN_PER_DAY
            };

            let arr: ndarray::Array2<f64> = dataset
                .read_slice_2d(s![row_start..row_end, ..])
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("hyperslab 读取失败: {e}"),
                    )
                })?;

            let n_rows = arr.nrows().min(MIN_PER_DAY);
            // 最后4根K线置为NaN
            for i in 0..n_rows {
                for j in 0..n_stocks.min(arr.ncols()) {
                    out[(day_offset, i, j)] = if i >= n_rows.saturating_sub(4) {
                        f64::NAN
                    } else {
                        arr[(i, j)]
                    };
                }
            }
        }
        io::Result::Ok(())
    })?;

    Ok((meta.col_to_code.clone(), out))
}

#[cfg(not(feature = "hdf5"))]
pub fn read_minute_field_multi_day(
    _field: &str,
    _dates: &[i64],
) -> io::Result<(Vec<String>, ndarray::Array3<f64>)> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "HDF5 支持未启用",
    ))
}

/// 获取当天有效的股票代码列表（去掉 NaN 列）。
/// 适用于需要知道有哪些股票活跃的场景。
pub fn get_active_codes(date: i64) -> io::Result<Vec<String>> {
    #[cfg(feature = "hdf5")]
    {
        // 读 volume 字段判断哪些股票有数据
        let (codes, data) = read_minute_field("volume", date)?;
        let active: Vec<String> = (0..data.ncols())
            .filter(|&j| {
                // 该列只要有任一非 NaN 值就算活跃
                (0..data.nrows()).any(|i| data[(i, j)].is_finite())
            })
            .map(|j| codes[j].clone())
            .collect();
        Ok(active)
    }
    #[cfg(not(feature = "hdf5"))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "HDF5 支持未启用",
        ))
    }
}

// ============================================================
// Python 包装
// ============================================================

/// Python 入口：读单字段单日 → (codes, flat_data, n_rows, n_cols)
#[pyfunction]
#[cfg(feature = "hdf5")]
pub fn py_read_minute_data(
    field: &str,
    date: i64,
) -> PyResult<(Vec<String>, Vec<f64>, usize, usize)> {
    let (codes, data) = read_minute_field(field, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let n_rows = data.nrows();
    let n_cols = data.ncols();
    let flat: Vec<f64> = data.iter().copied().collect();
    Ok((codes, flat, n_rows, n_cols))
}

#[pyfunction]
#[cfg(not(feature = "hdf5"))]
pub fn py_read_minute_data(
    _field: &str,
    _date: i64,
) -> PyResult<(Vec<String>, Vec<f64>, usize, usize)> {
    Err(pyo3::exceptions::PyIOError::new_err("HDF5 支持未启用"))
}
