//! 行业分类（申万一级, 直接读 `SzBa/industry.h5`, 纯 Rust）。
//!
//! 唯一源头：`{vars_root}/SzBa/industry.h5`。vars_root 由 data_paths 解析：
//! pipeline 参数 `vars_root` > 环境变量 `RUST_PYFUNC_VARS_DIR` > 默认 `/ssd_data/data/vars`。
//!   - dataset `data`，行 r = `{vars_root}/SzBa/calendar_map.csv` 的第 r 个日期；
//!   - 列 c = `{basic_info_dir}/symbol_map.csv` 里 pos=c 的股票；
//!   - 行业号 1..31，NaN = 未知 → 本模块统一转成 -1。
//!
//! 进程内只读一次：首次取数时把整表分块读进内存并压成 i16（峰值内存 ≈ 块行数 × 列数 × 8B，
//! 常驻 ≈ 行数 × 列数 × 2B），之后所有日期共用同一份表；vars_root 变化时自动重读。
//! 读不到文件直接返回 Err —— 不静默降级成整列 NaN（那正是 ind_* 因子在
//! 20260720~20260908 整列为空的成因）。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use ndarray::s;

/// 行业 h5（相对 vars_root）。
pub const INDUSTRY_H5: &str = "SzBa/industry.h5";
/// 行业 h5 的行轴日历（相对 vars_root）。
pub const INDUSTRY_CALENDAR: &str = "SzBa/calendar_map.csv";
/// 分块读取的行数。
const BLOCK_ROWS: usize = 256;

/// 全市场行业表：日期轴 + 列序 + i16 行业号。
pub struct IndustryTable {
    /// 数据来源（用于判断 vars_root 变化后是否要重读）。
    pub source: PathBuf,
    dates: Vec<i64>,
    ncols: usize,
    data: Vec<i16>,
    pos: HashMap<String, usize>,
}

impl IndustryTable {
    /// 最后一个 <= date 的行号（早于首日返回 None）。
    fn row_of(&self, date: i64) -> Option<usize> {
        match self.dates.partition_point(|&d| d <= date) {
            0 => None,
            n => Some(n - 1),
        }
    }

    /// 指定日期、单只股票的行业号（-1 = 未知或不在列内）。
    pub fn ind_of(&self, date: i64, code: &str) -> i16 {
        let Some(r) = self.row_of(date) else {
            return -1;
        };
        match self.pos.get(code) {
            Some(&c) if c < self.ncols => self.data[r * self.ncols + c],
            _ => -1,
        }
    }
}

static TABLE: RwLock<Option<Arc<IndustryTable>>> = RwLock::new(None);

fn io_err<E: std::fmt::Display>(what: &str, e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{what}: {e}"))
}

/// 保留原错误种类（NotFound 等），并在消息里带上文件路径。
fn path_err(what: &str, path: &std::path::Path, e: std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), format!("{what} {}: {e}", path.display()))
}

fn load_table() -> std::io::Result<IndustryTable> {
    let source = crate::data_paths::vars_path(INDUSTRY_H5);
    let cal_path = crate::data_paths::vars_path(INDUSTRY_CALENDAR);
    let cal = std::fs::read_to_string(&cal_path)
        .map_err(|e| path_err("读取行业日历失败", &cal_path, e))?;
    let dates: Vec<i64> = cal
        .lines()
        .skip(1)
        .filter_map(|l| l.trim().parse::<i64>().ok())
        .collect();
    let sym_path = crate::data_paths::basic_info_path("symbol_map.csv");
    let sym =
        std::fs::read_to_string(&sym_path).map_err(|e| path_err("读取 symbol_map 失败", &sym_path, e))?;
    let codes: Vec<String> = sym
        .lines()
        .skip(1)
        .map(|l| l.split(',').next().unwrap_or("").trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    let ncols = codes.len();
    let pos: HashMap<String, usize> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), i))
        .collect();

    let f = hdf5_metno::File::open(&source)
        .map_err(|e| path_err("打开行业 h5 失败", &source, std::io::Error::other(e.to_string())))?;
    let ds = f
        .dataset("data")
        .map_err(|e| io_err("行业 h5 缺少 dataset data", e))?;
    let nrows = ds.shape()[0].min(dates.len());

    let mut data: Vec<i16> = Vec::with_capacity(nrows * ncols);
    let mut r0 = 0;
    while r0 < nrows {
        let r1 = (r0 + BLOCK_ROWS).min(nrows);
        let arr: ndarray::Array2<f64> = ds
            .read_slice_2d(s![r0..r1, 0..ncols])
            .map_err(|e| io_err("行业 h5 分块读取失败", e))?;
        data.extend(arr.iter().map(|&v| if v.is_nan() { -1i16 } else { v as i16 }));
        r0 = r1;
    }
    Ok(IndustryTable {
        source,
        dates,
        ncols,
        data,
        pos,
    })
}

/// 进程内缓存的行业表：首次调用读盘，之后直接复用；vars_root 变化时重读。
pub fn table() -> std::io::Result<Arc<IndustryTable>> {
    let want = crate::data_paths::vars_path(INDUSTRY_H5);
    if let Some(t) = TABLE.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        if t.source == want {
            return Ok(Arc::clone(t));
        }
    }
    let fresh = Arc::new(load_table()?);
    *TABLE.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::clone(&fresh));
    Ok(fresh)
}

/// 指定日期、指定股票列表的行业号（-1 = 未知）。
pub fn industry_of(date: i64, codes: &[String]) -> std::io::Result<Vec<i16>> {
    let t = table()?;
    Ok(codes.iter().map(|c| t.ind_of(date, c)).collect())
}
