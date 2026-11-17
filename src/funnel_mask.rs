//! 两阶段漏斗（中间变量模块裁剪 + 存活列投影）公共纯 `std` 底座。
//!
//! 设计目标：
//! 1. **零第三方依赖（纯 `std`）**：不仅主库 `src/` 可直接调 `crate::funnel_mask`，
//!    任何独立 `sandbox_xxx` 也可通过 `#[path = "../../src/funnel_mask.rs"] mod funnel_mask;`
//!    直接挂载，与 `fast_csv_reader.rs`、`features_core.rs` 组成统一探索底座。
//! 2. **两级解耦**：
//!    - **Level A（中间变量模块开关）**：因子核函数顶部调一次 `let mask = funnel_mask::snapshot();`，
//!      每个耗时中间变量计算块前判断 `if mask.is_active("模块名") { ... } else { 填定长 NaN }`。
//!      核函数内部无需改动 `OUT_LEN` 或动态计算列偏移，天然遵守「输出列数恒定、无定义写空值」。
//!    - **Level B（存活列投影）**：由流水线调度层（`factor_pipeline` / `worker_pipeline`）在
//!      写入共享内存与 `factor_store` 之前，按 `keep_col_indices` 一次性抽取存活列（如 ≤3,500 列），
//!      直接把磁盘体积压进 `/nvme/chenzongwei` 固态盘 300 GB 预算内。

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

/// 主进程 → worker 子进程透传：启用的中间模块列表（逗号分隔，或 `@文件路径`）。
pub const ENV_ACTIVE_MODULES: &str = "RUST_PYFUNC_ACTIVE_MODULES";
/// 主进程 → worker 子进程透传：保留列下标文件路径（每行或逗号分隔的 `usize`）。
pub const ENV_KEEP_COLS_FILE: &str = "RUST_PYFUNC_KEEP_COLS_FILE";
/// 主进程 → worker 子进程透传：核函数原始完整列数（投影前长度）。
pub const ENV_RAW_FULL_LEN: &str = "RUST_PYFUNC_RAW_FULL_LEN";

static INPROC_MODULES: RwLock<Option<Arc<BTreeSet<String>>>> = RwLock::new(None);
static INPROC_MODULES_ENV: RwLock<Option<String>> = RwLock::new(None);
static INPROC_KEEP_COLS: RwLock<Option<Arc<Vec<usize>>>> = RwLock::new(None);
static INPROC_KEEP_COLS_FILE: RwLock<Option<PathBuf>> = RwLock::new(None);
static INPROC_RAW_FULL_LEN: RwLock<Option<usize>> = RwLock::new(None);
static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// 单次函数调用内的只读模块掩码快照（顶部取一次，循环内零锁开销）。
#[derive(Clone, Debug, Default)]
pub struct ActiveModuleMask {
    allowed: Option<Arc<BTreeSet<String>>>,
}

impl ActiveModuleMask {
    /// 未配置模块过滤时恒返回 `true`；配置后仅集合内模块返回 `true`。
    #[inline]
    pub fn is_active(&self, module_id: &str) -> bool {
        match &self.allowed {
            None => true,
            Some(set) => set.contains(module_id),
        }
    }

    /// 只要任一列出的模块处于启用状态即返回 `true`（供被多个下游差值模块共享的底表模块使用）。
    #[inline]
    pub fn any_active(&self, module_ids: &[&str]) -> bool {
        match &self.allowed {
            None => true,
            Some(set) => module_ids.iter().any(|m| set.contains(*m)),
        }
    }

    /// 是否处于全开模式（未裁剪任何模块）。
    #[inline]
    pub fn is_all_active(&self) -> bool {
        self.allowed.is_none()
    }
}

fn parse_modules_spec(raw: &str) -> Option<Arc<BTreeSet<String>>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let content = if let Some(path) = trimmed.strip_prefix('@') {
        std::fs::read_to_string(path).ok()?
    } else {
        trimmed.to_string()
    };
    let set: BTreeSet<String> = content
        .split(|c: char| c == ',' || c == '\n' || c == '\r' || c == ';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if set.is_empty() {
        None
    } else {
        Some(Arc::new(set))
    }
}

fn env_modules_cached() -> &'static Option<Arc<BTreeSet<String>>> {
    static ENV_MODS: OnceLock<Option<Arc<BTreeSet<String>>>> = OnceLock::new();
    ENV_MODS.get_or_init(|| {
        std::env::var(ENV_ACTIVE_MODULES)
            .ok()
            .and_then(|v| parse_modules_spec(&v))
    })
}

/// 获取当前生效的中间变量模块开关快照（优先进程内覆盖，其次 worker 环境变量，默认全开）。
#[inline]
pub fn snapshot() -> ActiveModuleMask {
    if let Some(arc) = INPROC_MODULES
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
    {
        return ActiveModuleMask { allowed: Some(arc) };
    }
    ActiveModuleMask {
        allowed: env_modules_cached().clone(),
    }
}

/// 便捷单次判断某个中间变量模块是否启用。
#[inline]
pub fn is_module_active(module_id: &str) -> bool {
    snapshot().is_active(module_id)
}

fn parse_keep_cols_file(path: &str) -> Option<Arc<Vec<usize>>> {
    let content = std::fs::read_to_string(path).ok()?;
    let cols: Vec<usize> = content
        .split(|c: char| c == ',' || c.is_ascii_whitespace())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();
    if cols.is_empty() {
        None
    } else {
        Some(Arc::new(cols))
    }
}

fn env_keep_cols_cached() -> &'static Option<Arc<Vec<usize>>> {
    static ENV_COLS: OnceLock<Option<Arc<Vec<usize>>>> = OnceLock::new();
    ENV_COLS.get_or_init(|| {
        std::env::var(ENV_KEEP_COLS_FILE)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .and_then(|p| parse_keep_cols_file(p.trim()))
    })
}

fn env_raw_full_len_cached() -> &'static Option<usize> {
    static ENV_RAW_LEN: OnceLock<Option<usize>> = OnceLock::new();
    ENV_RAW_LEN.get_or_init(|| {
        std::env::var(ENV_RAW_FULL_LEN)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

/// 获取当前生效的保留列下标列表（None 表示保留全部列）。
#[inline]
pub fn current_keep_cols() -> Option<Arc<Vec<usize>>> {
    if let Some(arc) = INPROC_KEEP_COLS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
    {
        return Some(arc);
    }
    env_keep_cols_cached().clone()
}

/// 获取投影前的核函数原始完整列数（None 表示等于外部 expected_len）。
#[inline]
pub fn current_raw_full_len() -> Option<usize> {
    if let Some(n) = *INPROC_RAW_FULL_LEN
        .read()
        .unwrap_or_else(|e| e.into_inner())
    {
        return Some(n);
    }
    *env_raw_full_len_cached()
}

/// 供主进程 spawn worker 子进程时读取透传环境变量。
pub fn override_active_modules_env() -> Option<String> {
    INPROC_MODULES_ENV
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// 供主进程 spawn worker 子进程时读取保留列文件路径。
pub fn override_keep_cols_file() -> Option<PathBuf> {
    INPROC_KEEP_COLS_FILE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// 供主进程 spawn worker 子进程时读取投影前完整列数。
pub fn override_raw_full_len() -> Option<usize> {
    *INPROC_RAW_FULL_LEN
        .read()
        .unwrap_or_else(|e| e.into_inner())
}

/// 按 `keep_cols` 投影单只股票结果向量；越界下标安全填 `NaN`。
#[inline]
pub fn project_row_f32(full: &[f32], keep_cols: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(keep_cols.len());
    let n = full.len();
    for &idx in keep_cols {
        out.push(if idx < n { full[idx] } else { f32::NAN });
    }
    out
}

/// 按 `keep_cols` 投影横截面平坦矩阵 `(n_stocks × full_cols)` → `(n_stocks × keep_cols.len())`。
#[inline]
pub fn project_flat_f32(flat: &[f32], full_cols: usize, keep_cols: &[usize]) -> Vec<f32> {
    if full_cols == 0 || flat.is_empty() {
        return Vec::new();
    }
    let n_rows = flat.len() / full_cols;
    let k = keep_cols.len();
    let mut out = Vec::with_capacity(n_rows * k);
    for r in 0..n_rows {
        let row = &flat[r * full_cols..(r + 1) * full_cols];
        for &idx in keep_cols {
            out.push(if idx < full_cols { row[idx] } else { f32::NAN });
        }
    }
    out
}

/// 进程内 + 子进程透传的漏斗过滤守卫（Drop 时自动还原进程内状态并清理临时文件）。
pub struct FunnelFilterGuard {
    prev_mods: Option<Arc<BTreeSet<String>>>,
    prev_mods_env: Option<String>,
    prev_cols: Option<Arc<Vec<usize>>>,
    prev_cols_file: Option<PathBuf>,
    prev_raw_len: Option<usize>,
    cols_file_to_clean: Option<PathBuf>,
    mods_file_to_clean: Option<PathBuf>,
}

impl Drop for FunnelFilterGuard {
    fn drop(&mut self) {
        *INPROC_MODULES.write().unwrap_or_else(|e| e.into_inner()) = self.prev_mods.take();
        *INPROC_MODULES_ENV.write().unwrap_or_else(|e| e.into_inner()) = self.prev_mods_env.take();
        *INPROC_KEEP_COLS.write().unwrap_or_else(|e| e.into_inner()) = self.prev_cols.take();
        *INPROC_KEEP_COLS_FILE
            .write()
            .unwrap_or_else(|e| e.into_inner()) = self.prev_cols_file.take();
        *INPROC_RAW_FULL_LEN
            .write()
            .unwrap_or_else(|e| e.into_inner()) = self.prev_raw_len.take();
        if let Some(p) = self.cols_file_to_clean.take() {
            let _ = std::fs::remove_file(p);
        }
        if let Some(p) = self.mods_file_to_clean.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// 设置本次流水线运行的模块开关与保留列索引。
pub fn set_funnel_filter(
    active_modules: Option<&[String]>,
    keep_col_indices: Option<&[usize]>,
    full_result_length: Option<usize>,
) -> std::io::Result<FunnelFilterGuard> {
    let pid = std::process::id();
    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = if std::path::Path::new("/dev/shm").is_dir() {
        PathBuf::from("/dev/shm")
    } else {
        std::env::temp_dir()
    };

    let (new_mods, mods_env_val, mods_file_path) = match active_modules {
        Some(list) => {
            let set: BTreeSet<String> = list
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if set.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "active_modules 不能为空列表（不裁剪请传 None）",
                ));
            }
            let joined: String = set.iter().cloned().collect::<Vec<_>>().join(",");
            if joined.len() <= 4096 {
                (Some(Arc::new(set)), Some(joined), None)
            } else {
                let p = tmp_dir.join(format!("rp_active_mods_{pid}_{seq}.txt"));
                std::fs::write(&p, &joined)?;
                let spec = format!("@{}", p.display());
                (Some(Arc::new(set)), Some(spec), Some(p))
            }
        }
        None => (None, None, None),
    };

    let (new_cols, cols_file_path, inferred_raw_len) = match keep_col_indices {
        Some(cols) => {
            if cols.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "keep_col_indices 不能为空列表（不裁剪请传 None）",
                ));
            }
            let max_idx = cols.iter().copied().max().unwrap_or(0);
            let raw_len = match full_result_length {
                Some(n) => {
                    if max_idx >= n {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "keep_col_indices 最大下标 {max_idx} 超出 full_result_length={n}"
                            ),
                        ));
                    }
                    n
                }
                None => max_idx + 1,
            };
            let text = cols
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            let p = tmp_dir.join(format!("rp_keep_cols_{pid}_{seq}.txt"));
            std::fs::write(&p, text)?;
            (Some(Arc::new(cols.to_vec())), Some(p), Some(raw_len))
        }
        None => (None, None, full_result_length),
    };

    let prev_mods = std::mem::replace(
        &mut *INPROC_MODULES.write().unwrap_or_else(|e| e.into_inner()),
        new_mods,
    );
    let prev_mods_env = std::mem::replace(
        &mut *INPROC_MODULES_ENV.write().unwrap_or_else(|e| e.into_inner()),
        mods_env_val,
    );
    let prev_cols = std::mem::replace(
        &mut *INPROC_KEEP_COLS.write().unwrap_or_else(|e| e.into_inner()),
        new_cols,
    );
    let prev_cols_file = std::mem::replace(
        &mut *INPROC_KEEP_COLS_FILE
            .write()
            .unwrap_or_else(|e| e.into_inner()),
        cols_file_path.clone(),
    );
    let prev_raw_len = std::mem::replace(
        &mut *INPROC_RAW_FULL_LEN
            .write()
            .unwrap_or_else(|e| e.into_inner()),
        inferred_raw_len,
    );

    Ok(FunnelFilterGuard {
        prev_mods,
        prev_mods_env,
        prev_cols,
        prev_cols_file,
        prev_raw_len,
        cols_file_to_clean: cols_file_path,
        mods_file_to_clean: mods_file_path,
    })
}
