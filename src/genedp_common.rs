//! 基因动态规划系列（genedp_*，四个横截面因子）共用的小工具。
//!
//! 四个系列的 Level2 文件打开、缓冲复用与页缓存释放统一委托 `crate::fast_csv_reader`；
//! 这里统一它们的「目录列举与日频变量路径」：原始数据目录、前一交易日、日频变量目录都走 `crate::data_paths`，
//! 因此 pipeline 的 data_root 参数、RUST_PYFUNC_LEVEL2_ROOT、RUST_PYFUNC_VARS_DIR 对它们同样生效。

use std::collections::HashSet;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

/// 生效的 Level2 根目录（字符串形式，供各系列拼 `{root}/{date}/transaction`）。
pub fn level2_root_string() -> String {
    crate::data_paths::level2_roots()[0].to_string_lossy().into_owned()
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 按根目录缓存含 `transaction` 子目录的有效交易日升序表（仅加速目录枚举，不缓存因子值）。
fn cached_level2_dates(root: &Path) -> Option<Arc<Vec<i64>>> {
    static CACHE: OnceLock<Mutex<Option<(PathBuf, Arc<Vec<i64>>)>>> = OnceLock::new();
    let lock = CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = lock.lock() {
        if let Some((ref cached_root, ref dates)) = *guard {
            if cached_root == root {
                return Some(Arc::clone(dates));
            }
        }
    }
    let mut dates: Vec<i64> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse::<i64>().ok())
        .filter(|&d| d > 19_000_000 && root.join(d.to_string()).join("transaction").is_dir())
        .collect();
    dates.sort_unstable();
    dates.dedup();
    let arc = Arc::new(dates);
    if let Ok(mut guard) = lock.lock() {
        *guard = Some((root.to_path_buf(), Arc::clone(&arc)));
    }
    Some(arc)
}

/// 前一交易日：Level2 根目录下早于 date、且含 transaction 子目录的最大日期（只列一层目录并缓存日期表）。
/// 四个系列都只和「昨天」比，不读任何落盘缓存。
pub fn prev_level2_date(date: i64) -> Option<i64> {
    let root = crate::data_paths::level2_roots()[0].clone();
    let dates = cached_level2_dates(&root)?;
    let idx = dates.partition_point(|&d| d < date);
    if idx > 0 {
        Some(dates[idx - 1])
    } else {
        None
    }
}

struct WhitelistCacheEntry {
    path: PathBuf,
    mtime: Option<SystemTime>,
    len: u64,
    hash: u64,
    codes: Arc<Vec<String>>,
}

/// 读取并校验 `symbol_map.csv` 股票白名单（成功时按 `(path, mtime, len)` 缓存；缺失/空表/非法代码一律立即返回错误，绝不缓存错误）。
pub fn load_symbol_whitelist(symbol_map: &Path) -> Result<(Arc<Vec<String>>, u64), String> {
    static WL_CACHE: OnceLock<Mutex<Option<WhitelistCacheEntry>>> = OnceLock::new();
    let meta = std::fs::metadata(symbol_map)
        .map_err(|e| format!("股票表读取失败 {}: {e}", symbol_map.display()))?;
    let mtime = meta.modified().ok();
    let len = meta.len();
    let lock = WL_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = lock.lock() {
        if let Some(ref entry) = *guard {
            if entry.path == symbol_map && entry.mtime == mtime && entry.len == len {
                return Ok((Arc::clone(&entry.codes), entry.hash));
            }
        }
    }
    let bytes = std::fs::read(symbol_map)
        .map_err(|e| format!("股票表读取失败 {}: {e}", symbol_map.display()))?;
    let mut reader = csv::Reader::from_reader(&bytes[..]);
    let column = reader
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .position(|h| h == "symbol")
        .ok_or_else(|| "股票表缺少 symbol 列".to_string())?;
    let mut set = std::collections::BTreeSet::new();
    for rec in reader.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        let code = rec.get(column).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!("股票表含无效代码: {code}"));
        }
        set.insert(format!("{code:0>6}"));
    }
    if set.is_empty() {
        return Err("股票白名单为空".into());
    }
    let hash = fnv64(&bytes);
    let codes: Arc<Vec<String>> = Arc::new(set.into_iter().collect());
    if let Ok(mut guard) = lock.lock() {
        *guard = Some(WhitelistCacheEntry {
            path: symbol_map.to_path_buf(),
            mtime,
            len,
            hash,
            codes: Arc::clone(&codes),
        });
    }
    Ok((codes, hash))
}

/// 单次 `read_dir` 获取某日某子目录下在白名单中的股票代码（升序），省去每日 5,500 次 `stat()` 系统调用，语义与逐股 `is_file()` 完全一致。
pub fn stock_codes_whitelisted(
    root: &str,
    symbol_map: &Path,
    date: i64,
    subdir: &str,
) -> Result<(Vec<String>, String, u64), String> {
    let (wl, hash) = load_symbol_whitelist(symbol_map)?;
    let dir = Path::new(root).join(date.to_string()).join(subdir);
    if !dir.is_dir() {
        return Err(format!("原始数据目录不存在: {}", dir.display()));
    }
    let suffix = format!("_{date}_{subdir}.csv");
    let rd = std::fs::read_dir(&dir)
        .map_err(|e| format!("读取原始数据目录失败 {}: {e}", dir.display()))?;
    let mut present: HashSet<String> = HashSet::with_capacity(6000);
    for entry in rd.flatten() {
        let fname = entry.file_name();
        let Some(s) = fname.to_str() else { continue };
        let Some(code) = s.strip_suffix(&suffix) else { continue };
        if code.len() == 6 && code.bytes().all(|b| b.is_ascii_digit()) {
            let is_valid_file = match entry.file_type() {
                Ok(ft) if ft.is_file() => true,
                Ok(ft) if ft.is_symlink() => entry.path().is_file(),
                _ => entry.path().is_file(),
            };
            if is_valid_file {
                present.insert(code.to_string());
            }
        }
    }
    let codes: Vec<String> = wl.iter().filter(|c| present.contains(c.as_str())).cloned().collect();
    Ok((codes, symbol_map.display().to_string(), hash))
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
