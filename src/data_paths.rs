//! 原始数据根目录解析：pipeline 参数 `data_root` 的透传与强制覆盖。
//!
//! 约定：`data_root` 是"默认布局的根目录"（不是某个具体文件），默认 `/ssd_data`，
//! 其下结构与当前生产数据完全一致：
//!
//! ```text
//! {data_root}/
//!   stock/{date}/{subdir}/{code}_{date}_*.csv     # Level2 逐笔(transaction)/盘口(order)
//!   data/1min_factor_text/{field}.h5              # 分钟数据 + calendar_map.csv / symbol_map.csv
//!   data/vars/SzBa/industry.h5 等                 # 行业/日频变量
//!   data/basic_info/symbol_map.csv                # 代码-列号映射
//! ```
//!
//! 优先级（高 → 低）：
//!   1. pipeline 参数 `data_root`：进程内全局覆盖，`set_data_root` 返回的 guard 在
//!      调用结束时自动还原（Python 进程后续调用不受影响）；
//!   2. 环境变量 `RUST_PYFUNC_DATA_ROOT`：主进程 spawn worker 时透传给子进程；
//!   3. 兼容旧环境变量 `RUST_PYFUNC_LEVEL2_PATH`：语义 = 直接指向 Level2 的 stock 根
//!      （即 `{data_root}/stock`），仅影响 Level2，分钟数据仍用默认；
//!   4. 默认 `/ssd_data`。
//!
//! 一旦出现 1 或 2（即显式指定了数据根），Level2 只在该根下查找，**不再回退**到
//! `/nas197/binary/stock/sz_alpha/stock`——保证"指定的路径就是唯一数据源"，路径写错会
//! 直接报文件未找到，而不是静默读到另一份数据。

use std::path::PathBuf;
use std::sync::RwLock;

/// 默认数据根（其下含 stock/ 与 data/）。
pub const DEFAULT_DATA_ROOT: &str = "/ssd_data";
/// 环境变量名：主进程 → worker 的数据根透传通道。
pub const ENV_DATA_ROOT: &str = "RUST_PYFUNC_DATA_ROOT";
/// 旧环境变量名（只覆盖 Level2，语义 = stock 根）。
pub const LEGACY_ENV_LEVEL2_ROOT: &str = "RUST_PYFUNC_LEVEL2_PATH";
/// 无显式覆盖时的 Level2 二级回退根（历史行为）。
const FALLBACK_LEVEL2_ROOT: &str = "/nas197/binary/stock/sz_alpha/stock";

static OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

fn read_override() -> Option<PathBuf> {
    OVERRIDE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// 设置进程内数据根覆盖；返回的 guard 在 drop 时还原为设置前的值。
///
/// pipeline 入口调用一次即可，函数内所有读取器（含 rayon 工作线程）都会看到覆盖值。
pub fn set_data_root(root: Option<&str>) -> DataRootGuard {
    let mut w = OVERRIDE.write().unwrap_or_else(|e| e.into_inner());
    let prev = w.take();
    *w = root.map(PathBuf::from);
    DataRootGuard { prev }
}

/// RAII guard：drop 时还原覆盖前状态。
pub struct DataRootGuard {
    prev: Option<PathBuf>,
}

impl Drop for DataRootGuard {
    fn drop(&mut self) {
        let mut w = OVERRIDE.write().unwrap_or_else(|e| e.into_inner());
        *w = self.prev.take();
    }
}

/// 当前进程是否被 pipeline 参数覆盖（spawn worker 时用它透传环境变量）。
pub fn override_root() -> Option<PathBuf> {
    read_override()
}

/// 生效的数据根：pipeline 参数 > RUST_PYFUNC_DATA_ROOT > 默认。
pub fn data_root() -> PathBuf {
    if let Some(p) = read_override() {
        return p;
    }
    if let Ok(v) = std::env::var(ENV_DATA_ROOT) {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    PathBuf::from(DEFAULT_DATA_ROOT)
}

/// Level2 候选根目录列表（每个根下是 `{date}/{subdir}/`）。
///
/// 有显式覆盖（参数或环境变量）时只返回一个根（强制覆盖、不回退）；
/// 无覆盖时保持历史行为：`/ssd_data/stock` → `/nas197/...` 两级回退。
pub fn level2_roots() -> Vec<PathBuf> {
    if let Some(p) = read_override() {
        return vec![p.join("stock")];
    }
    if let Ok(v) = std::env::var(ENV_DATA_ROOT) {
        if !v.is_empty() {
            return vec![PathBuf::from(v).join("stock")];
        }
    }
    if let Ok(v) = std::env::var(LEGACY_ENV_LEVEL2_ROOT) {
        if !v.is_empty() {
            return vec![PathBuf::from(v)];
        }
    }
    vec![
        PathBuf::from(DEFAULT_DATA_ROOT).join("stock"),
        PathBuf::from(FALLBACK_LEVEL2_ROOT),
    ]
}

/// `{level2_root}/{date}/{subdir}`（首个候选根，用于列目录/构造路径）。
pub fn level2_dir(date: i64, subdir: &str) -> PathBuf {
    level2_roots()[0].join(date.to_string()).join(subdir)
}

/// `{level2_root}/{date}/{subdir}/{filename}`。
pub fn level2_file(date: i64, subdir: &str, filename: &str) -> PathBuf {
    level2_dir(date, subdir).join(filename)
}

/// 分钟数据目录 `{root}/data/1min_factor_text`。
pub fn minute_dir() -> PathBuf {
    data_root().join("data").join("1min_factor_text")
}

/// 分钟数据目录下的文件 `{root}/data/1min_factor_text/{name}`。
pub fn minute_file(name: &str) -> PathBuf {
    minute_dir().join(name)
}

/// `{root}/data/vars/{rel}`。
pub fn vars_path(rel: &str) -> PathBuf {
    data_root().join("data").join("vars").join(rel)
}

/// `{root}/data/basic_info/{rel}`。
pub fn basic_info_path(rel: &str) -> PathBuf {
    data_root().join("data").join("basic_info").join(rel)
}

/// 可读的当前生效数据根（pipeline 入口打印用）。
pub fn describe() -> String {
    if let Some(p) = read_override() {
        return format!("{}（pipeline 参数 data_root 覆盖）", p.display());
    }
    if let Ok(v) = std::env::var(ENV_DATA_ROOT) {
        if !v.is_empty() {
            return format!("{v}（环境变量 {ENV_DATA_ROOT}）");
        }
    }
    format!("{DEFAULT_DATA_ROOT}（默认）")
}
