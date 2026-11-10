//! 基因动态规划系列（genedp_*，四个横截面因子）共用的小工具。
//!
//! 四个系列各自带一份按 f64 精度解析的逐笔/盘口读取器（从沙箱原样搬来，因子值与沙箱逐字节一致），
//! 这里统一它们的「数据在哪」：原始数据目录、前一交易日、日频变量目录都走 crate::data_paths，
//! 因此 pipeline 的 data_root 参数、RUST_PYFUNC_LEVEL2_ROOT、RUST_PYFUNC_VARS_DIR 对它们同样生效。

use std::io::{Error, ErrorKind};

/// 原始 Level2 文件路径。与 crate::fast_csv_reader 的同名函数口径相同：
/// pipeline 参数 data_root > RUST_PYFUNC_LEVEL2_ROOT > 旧变量 RUST_PYFUNC_LEVEL2_PATH > 默认两级目录。
pub fn resolve_stock_path(date: i64, subdir: &str, filename: &str) -> std::io::Result<String> {
    let roots = crate::data_paths::level2_roots();
    for root in &roots {
        let p = root.join(date.to_string()).join(subdir).join(filename);
        if p.exists() {
            return Ok(p.to_string_lossy().into_owned());
        }
    }
    Err(Error::new(
        ErrorKind::NotFound,
        format!("股票数据文件未找到: date={date}, subdir={subdir}, file={filename}, 根目录={roots:?}"),
    ))
}

/// 生效的 Level2 根目录（字符串形式，供各系列拼 `{root}/{date}/transaction`）。
pub fn level2_root_string() -> String {
    crate::data_paths::level2_roots()[0].to_string_lossy().into_owned()
}

/// 前一交易日：Level2 根目录下早于 date、且含 transaction 子目录的最大日期（只列一层目录）。
/// 四个系列都只和「昨天」比，不读任何落盘缓存；昨天的指标一律从昨天的原始数据重新算。
pub fn prev_level2_date(date: i64) -> Option<i64> {
    let root = crate::data_paths::level2_roots()[0].clone();
    std::fs::read_dir(&root)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse::<i64>().ok())
        .filter(|&d| d < date && d > 19_000_000 && root.join(d.to_string()).join("transaction").is_dir())
        .max()
}

/// 日频变量目录下的文件（默认 /ssd_data/data/vars/{rel}）。
pub fn vars_file(rel: &str) -> String {
    crate::data_paths::vars_path(rel).to_string_lossy().into_owned()
}

/// Python 调试入口用：在独立的 50 线程池里跑（项目规定横截面函数最多 50 线程）。
/// pipeline 的 worker 进程已按 n_jobs 配好线程数，不经过这里。
pub fn run_in_50_threads<T: Send>(f: impl FnOnce() -> T + Send) -> std::io::Result<T> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(50)
        .build()
        .map_err(|e| Error::new(ErrorKind::Other, format!("建线程池失败: {e}")))?;
    Ok(pool.install(f))
}

pub fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}
