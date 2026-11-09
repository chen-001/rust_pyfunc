//! 原始数据目录解析：pipeline 参数 `data_root` 的透传与强制覆盖。
//!
//! **约定：`data_root` 就是"该 pipeline 原始数据所在的那个目录"本身，引擎不做任何拼接。**
//!
//! - `run_factor_pipeline` / `run_factor_pipeline_cross_section`（Level2）：
//!   传"直接包含各日期文件夹的目录"，默认 `/ssd_data/stock`：
//!   ```text
//!   {data_root}/{date}/transaction/{code}_{date}_transaction.csv   # 逐笔
//!   {data_root}/{date}/market_data/{code}_{date}_market_data.csv   # 盘口
//!   ```
//! - `run_factor_pipeline_minute`（分钟）：
//!   传"直接包含各 .h5 数据文件的目录"，默认 `/ssd_data/data/1min_factor_text`：
//!   ```text
//!   {data_root}/{field}.h5
//!   {data_root}/calendar_map.csv、{data_root}/symbol_map.csv
//!   ```
//!
//! 少量横截面因子额外读取的行业 / 基础信息**不在 data_root 下**，由各自的根目录参数指定
//! （默认值即生产路径，与 data_root 互不影响）：
//!   - 行业 / 日频变量根 `vars_root`（pipeline 参数）→ `{vars_root}/SzBa/industry.h5`
//!   - `RUST_PYFUNC_VARS_DIR`（默认 `/ssd_data/data/vars`）→ 同上，worker 透传通道
//!   - `RUST_PYFUNC_BASIC_INFO_DIR`（默认 `/ssd_data/data/basic_info`）→ `symbol_map.csv`
//!     同时也是**股票名单**的唯一来源（`list_stock_codes` 用它剔除 ETF/基金/债券/B 股）
//!
//! 优先级（高 → 低）：
//!   1. pipeline 参数 `data_root`：进程内覆盖，`set_*_root` 返回的 guard 在调用结束时还原；
//!   2. 环境变量 `RUST_PYFUNC_LEVEL2_ROOT` / `RUST_PYFUNC_MINUTE_ROOT`：
//!      主进程 spawn worker 时透传给子进程；
//!   3. 兼容旧环境变量 `RUST_PYFUNC_LEVEL2_PATH`（语义与 `RUST_PYFUNC_LEVEL2_ROOT` 相同，
//!      都指向 Level2 数据目录）；
//!   4. 默认目录。
//!
//! 一旦出现 1 或 2（即显式指定了目录），Level2 只在该目录下查找，**不再回退**到
//! `/nas197/binary/stock/sz_alpha/stock`——保证"指定的目录就是唯一数据源"，路径写错会
//! 直接报文件未找到，而不是静默读到另一份数据。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

/// Level2 数据目录默认值（直接包含 `{date}/` 各日期文件夹）。
pub const DEFAULT_LEVEL2_ROOT: &str = "/ssd_data/stock";
/// 分钟数据目录默认值（直接包含 `{field}.h5`）。
pub const DEFAULT_MINUTE_ROOT: &str = "/ssd_data/data/1min_factor_text";
/// 行业 / 日频变量目录默认值。
pub const DEFAULT_VARS_DIR: &str = "/ssd_data/data/vars";
/// 基础信息（symbol_map.csv 等）目录默认值。
pub const DEFAULT_BASIC_INFO_DIR: &str = "/ssd_data/data/basic_info";

/// 环境变量：Level2 数据目录（主进程 → worker 透传通道）。
pub const ENV_LEVEL2_ROOT: &str = "RUST_PYFUNC_LEVEL2_ROOT";
/// 环境变量：分钟数据目录（主进程 → worker 透传通道）。
pub const ENV_MINUTE_ROOT: &str = "RUST_PYFUNC_MINUTE_ROOT";
/// 旧环境变量：Level2 数据目录（语义同上，保留兼容）。
pub const LEGACY_ENV_LEVEL2_ROOT: &str = "RUST_PYFUNC_LEVEL2_PATH";
/// 环境变量：行业 / 日频变量目录。
pub const ENV_VARS_DIR: &str = "RUST_PYFUNC_VARS_DIR";
/// 环境变量：基础信息目录。
pub const ENV_BASIC_INFO_DIR: &str = "RUST_PYFUNC_BASIC_INFO_DIR";

/// 无显式覆盖时的 Level2 二级回退目录（历史行为）。
const FALLBACK_LEVEL2_ROOT: &str = "/nas197/binary/stock/sz_alpha/stock";

/// 覆盖的种类：Level2 / 分钟 / 行业变量各自独立，互不干扰。
#[derive(Clone, Copy)]
pub enum DataKind {
    Level2,
    Minute,
    Vars,
}

static LEVEL2_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);
static MINUTE_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);
static VARS_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

fn lock_of(kind: DataKind) -> &'static RwLock<Option<PathBuf>> {
    match kind {
        DataKind::Level2 => &LEVEL2_OVERRIDE,
        DataKind::Minute => &MINUTE_OVERRIDE,
        DataKind::Vars => &VARS_OVERRIDE,
    }
}

fn read_override(kind: DataKind) -> Option<PathBuf> {
    lock_of(kind)
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => None,
    }
}

/// 设置进程内 Level2 数据目录覆盖；返回的 guard 在 drop 时还原。
pub fn set_level2_root(root: Option<&str>) -> DataRootGuard {
    set_root(DataKind::Level2, root)
}

/// 设置进程内分钟数据目录覆盖；返回的 guard 在 drop 时还原。
pub fn set_minute_root(root: Option<&str>) -> DataRootGuard {
    set_root(DataKind::Minute, root)
}

/// 设置进程内行业 / 日频变量目录覆盖；返回的 guard 在 drop 时还原。
pub fn set_vars_root(root: Option<&str>) -> DataRootGuard {
    set_root(DataKind::Vars, root)
}

fn set_root(kind: DataKind, root: Option<&str>) -> DataRootGuard {
    let mut w = lock_of(kind).write().unwrap_or_else(|e| e.into_inner());
    let prev = w.take();
    *w = root.map(PathBuf::from);
    DataRootGuard { kind, prev }
}

/// RAII guard：drop 时还原覆盖前的值。
pub struct DataRootGuard {
    kind: DataKind,
    prev: Option<PathBuf>,
}

impl Drop for DataRootGuard {
    fn drop(&mut self) {
        let mut w = lock_of(self.kind)
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *w = self.prev.take();
    }
}

/// 当前进程的 Level2 覆盖值（spawn worker 时用它透传环境变量）。
pub fn override_level2() -> Option<PathBuf> {
    read_override(DataKind::Level2)
}

/// 当前进程的分钟覆盖值（spawn worker 时用它透传环境变量）。
pub fn override_minute() -> Option<PathBuf> {
    read_override(DataKind::Minute)
}

/// 当前进程的行业变量目录覆盖值（spawn worker 时用它透传环境变量）。
pub fn override_vars() -> Option<PathBuf> {
    read_override(DataKind::Vars)
}

/// 生效的 Level2 数据目录：参数 > RUST_PYFUNC_LEVEL2_ROOT > 旧变量 > 默认。
pub fn level2_root() -> PathBuf {
    override_level2()
        .or_else(|| env_path(ENV_LEVEL2_ROOT))
        .or_else(|| env_path(LEGACY_ENV_LEVEL2_ROOT))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_LEVEL2_ROOT))
}

/// Level2 候选目录列表（每个目录下是 `{date}/{subdir}/`）。
///
/// 有显式覆盖（参数或环境变量）时只返回一个目录（强制覆盖、不回退）；
/// 无覆盖时保持历史行为：`/ssd_data/stock` → `/nas197/...` 两级回退。
pub fn level2_roots() -> Vec<PathBuf> {
    if let Some(p) = override_level2() {
        return vec![p];
    }
    if let Some(p) = env_path(ENV_LEVEL2_ROOT) {
        return vec![p];
    }
    if let Some(p) = env_path(LEGACY_ENV_LEVEL2_ROOT) {
        return vec![p];
    }
    vec![
        PathBuf::from(DEFAULT_LEVEL2_ROOT),
        PathBuf::from(FALLBACK_LEVEL2_ROOT),
    ]
}

/// `{level2_root}/{date}/{subdir}`（首个候选目录，用于列目录/构造路径）。
pub fn level2_dir(date: i64, subdir: &str) -> PathBuf {
    level2_roots()[0].join(date.to_string()).join(subdir)
}

/// A 股股票名单的来源文件（`symbol_map.csv`，首行表头 `symbol,pos`，之后每行 `代码,序号`）。
///
/// 与 `switch_moment_level2` / `yhyb_indtop_metrics` 读的是同一个文件：
/// `{RUST_PYFUNC_BASIC_INFO_DIR}/symbol_map.csv`，默认 `/ssd_data/data/basic_info/symbol_map.csv`。
/// 用交易所/数据平台维护的正式名单，而不是自己按代码段猜——将来代码段规则变化时，
/// 只要平台更新了这张表，这里自动跟上，不需要改代码。
///
/// 已知的小缺口：`000022` / `000043` / `601313` 这三只因更名换码而从名单消失的历史股票
/// （分别并入 001872 / 001914 / 601360），按名单过滤时会被一起丢掉，影响可忽略。
fn stock_universe_path() -> PathBuf {
    basic_info_path("symbol_map.csv")
}

/// 股票名单缓存（key = 名单文件路径）：worker 一进程一 pipeline，读一次即够。
/// `None` 表示名单读不到或内容为空。
static STOCK_UNIVERSE_CACHE: LazyLock<Mutex<HashMap<PathBuf, Option<Arc<HashSet<String>>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// 全量股票代码集合；读不到名单时返回 `None`（调用方退化为"所有 6 位数字代码"）。
fn stock_universe() -> Option<Arc<HashSet<String>>> {
    let path = stock_universe_path();
    let mut cache = STOCK_UNIVERSE_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = cache.get(&path) {
        return hit.clone();
    }
    let loaded = std::fs::read_to_string(&path)
        .ok()
        .map(|text| {
            let set: HashSet<String> = text
                .lines()
                .filter_map(|l| l.split(',').next())
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() == 6 && s.bytes().all(|b| b.is_ascii_digit()))
                .collect();
            Arc::new(set)
        })
        .filter(|s| !s.is_empty());
    if loaded.is_none() {
        // 宁可退化也不能返回空名单：空名单会让整批横截面任务变成"空 universe"而失败。
        eprintln!(
            "⚠️ 读不到股票名单 {}（文件缺失或内容为空）：本次横截面枚举退化为「所有 6 位数字代码」，\
             可能混入 ETF/基金/债券/B 股等非股票标的。请检查 RUST_PYFUNC_BASIC_INFO_DIR。",
            path.display()
        );
    }
    cache.insert(path, loaded.clone());
    loaded
}

/// 6 位代码是否为 **A 股股票**（剔除 ETF/LOF/REITs/可转债/国债逆回购/B 股等非股票标的）。
///
/// 判定依据是 `symbol_map.csv` 这份正式名单，不再按代码段猜（见 `stock_universe_path`）。
/// 名单读不到时退化为"是 6 位数字就算"，即修复前的行为。
pub fn is_stock_code(code: &str) -> bool {
    if code.len() != 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    match stock_universe() {
        Some(universe) => universe.contains(code),
        None => true,
    }
}

/// 列出某天某子目录下**只含 A 股股票**的代码（文件名形如 `{code}_{date}_{type}.csv`）。
///
/// 这是所有横截面 pipeline 枚举全市场股票的唯一入口：非股票标的（ETF/基金/债券/B 股）
/// 在这里就被剔除，既不读它们的文件，也不会被算进横截面统计。
/// 返回按代码升序排列，顺序确定（多个进程/多次运行结果一致）。
pub fn list_stock_codes(date: i64, subdir: &str) -> Vec<String> {
    let universe = stock_universe(); // 循环外取一次，避免每个文件都加锁查缓存
    let dir = level2_dir(date, subdir);
    let mut set = std::collections::BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                let is_six_digits =
                    code.len() == 6 && code.bytes().all(|b| b.is_ascii_digit());
                // universe 为 None（名单读不到）时退化成修复前的行为：所有 6 位数字代码。
                let in_universe = universe
                    .as_ref()
                    .map_or(true, |u| u.contains(code));
                if is_six_digits && in_universe {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// 生效的分钟数据目录：参数 > RUST_PYFUNC_MINUTE_ROOT > 默认。
pub fn minute_root() -> PathBuf {
    override_minute()
        .or_else(|| env_path(ENV_MINUTE_ROOT))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MINUTE_ROOT))
}

/// 分钟数据目录（= minute_root，语义化别名）。
pub fn minute_dir() -> PathBuf {
    minute_root()
}

/// 分钟数据目录下的文件 `{minute_root}/{name}`。
pub fn minute_file(name: &str) -> PathBuf {
    minute_root().join(name)
}

/// 生效的行业 / 日频变量目录：参数 > RUST_PYFUNC_VARS_DIR > 默认。
pub fn vars_root() -> PathBuf {
    override_vars()
        .or_else(|| env_path(ENV_VARS_DIR))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_VARS_DIR))
}

/// `{vars_root}/{rel}`，默认 `{DEFAULT_VARS_DIR}/{rel}`。
pub fn vars_path(rel: &str) -> PathBuf {
    vars_root().join(rel)
}

/// `{RUST_PYFUNC_BASIC_INFO_DIR}/{rel}`，默认 `{DEFAULT_BASIC_INFO_DIR}/{rel}`。
pub fn basic_info_path(rel: &str) -> PathBuf {
    env_path(ENV_BASIC_INFO_DIR)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BASIC_INFO_DIR))
        .join(rel)
}

/// 可读的当前生效 Level2 目录（pipeline 入口打印用）。
pub fn describe_level2() -> String {
    if let Some(p) = override_level2() {
        return format!("{}（pipeline 参数 data_root 覆盖）", p.display());
    }
    for name in [ENV_LEVEL2_ROOT, LEGACY_ENV_LEVEL2_ROOT] {
        if let Some(p) = env_path(name) {
            return format!("{}（环境变量 {name}）", p.display());
        }
    }
    format!("{DEFAULT_LEVEL2_ROOT}（默认）")
}

/// 可读的当前生效分钟目录（pipeline 入口打印用）。
pub fn describe_minute() -> String {
    if let Some(p) = override_minute() {
        return format!("{}（pipeline 参数 data_root 覆盖）", p.display());
    }
    if let Some(p) = env_path(ENV_MINUTE_ROOT) {
        return format!("{}（环境变量 {ENV_MINUTE_ROOT}）", p.display());
    }
    format!("{DEFAULT_MINUTE_ROOT}（默认）")
}

/// 可读的当前生效行业 / 日频变量目录（pipeline 入口打印用）。
pub fn describe_vars() -> String {
    if let Some(p) = override_vars() {
        return format!("{}（pipeline 参数 vars_root 覆盖）", p.display());
    }
    if let Some(p) = env_path(ENV_VARS_DIR) {
        return format!("{}（环境变量 {ENV_VARS_DIR}）", p.display());
    }
    format!("{DEFAULT_VARS_DIR}（默认）")
}
