//! Storage decisions follow the mounted filesystem, including symlink targets.
//! No naming heuristic: /nas197/user_home_unsafe is a symlink to local HDD.
use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct StorageProfile {
    pub network: bool,
    pub btrfs: bool,
}
impl StorageProfile {
    pub fn detect(path: &Path) -> Result<Self, String> {
        let mut existing = path.to_path_buf();
        loop {
            match std::fs::metadata(&existing) {
                Ok(_) => break,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    if !existing.pop() {
                        existing = PathBuf::from(".");
                        break;
                    }
                }
                Err(e) => return Err(format!("检查存储路径 {existing:?}: {e}")),
            }
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let f = File::open(&existing).map_err(|e| format!("打开存储路径 {existing:?}: {e}"))?;
            let mut st = std::mem::MaybeUninit::<libc::statfs>::uninit();
            if unsafe { libc::fstatfs(f.as_raw_fd(), st.as_mut_ptr()) } != 0 {
                return Err(format!(
                    "检查文件系统 {existing:?}: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let ty = unsafe { st.assume_init() }.f_type as u64;
            return Ok(Self {
                network: matches!(ty, 0x6969 | 0xff534d42 | 0xfe534d42),
                btrfs: ty == libc::BTRFS_SUPER_MAGIC as u64,
            });
        }
        #[cfg(not(target_os = "linux"))]
        Ok(Self {
            network: false,
            btrfs: false,
        })
    }
    pub fn io_concurrency(self, jobs: usize) -> usize {
        if self.network {
            jobs.min(8).max(1)
        } else {
            jobs.max(1)
        }
    }
    pub fn io_batch_size(self) -> usize {
        if self.network {
            4
        } else {
            1
        }
    }
    pub fn write_queue_batches(self, batch_bytes: usize) -> usize {
        if self.network {
            (512 * 1024 * 1024 / batch_bytes.max(1)).clamp(1, 50)
        } else {
            50
        }
    }
}

/// A real sync is required before publishing a commit/projection; flush is not fsync.
///
/// 这里刻意不重试。2026-10-16 独立核验实测：往 NAS 写 64 GiB，第一次确认在 192.6 秒报 EIO，
/// 同一个句柄上紧接着再确认一次 3.4 秒就返回成功，但读回时 65536 块里有 28582 块内容不对，
/// 服务端真实长度比客户端报的少约 64 MB（两轮都复现）。也就是说软挂载超时之后，客户端已经
/// 把那批写丢掉了，再问一次得到的成功是假的。重试会把「确定失败」变成「静默成功且产物损坏」，
/// 方向与「读取失败不得被解释成缺失值」正相反。要扛住对面停顿靠的是把确认拆小
/// （见 factor_store_v5.rs 的 write_proj_values），不是失败之后重问。
pub fn sync_file(file: &File, context: &str) -> Result<(), String> {
    file.sync_all()
        .map_err(|e| format!("保存 {context} 失败: {e}"))
}
pub fn sync_dir(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        File::open(path)
            .and_then(|f| f.sync_all())
            .map_err(|e| format!("保存目录 {path:?} 失败: {e}"))?;
    }
    Ok(())
}
pub fn publish(tmp: &Path, path: &Path) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .open(tmp)
        .map_err(|e| format!("打开待发布文件 {tmp:?}: {e}"))?;
    sync_file(&file, &tmp.display().to_string())?;
    std::fs::rename(tmp, path).map_err(|e| format!("发布 {tmp:?} → {path:?}: {e}"))?;
    sync_dir(path.parent().unwrap_or(Path::new(".")))
}
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("atomic.tmp");
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)
        .map_err(|e| format!("创建 {tmp:?}: {e}"))?;
    f.write_all(bytes)
        .map_err(|e| format!("写入 {tmp:?}: {e}"))?;
    drop(f);
    publish(&tmp, path)
}
/// Missing is distinct from an I/O error. Network errors must never erase progress.
pub fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match File::open(path) {
        Ok(mut f) => {
            let mut bytes = Vec::new();
            f.read_to_end(&mut bytes)
                .map_err(|e| format!("读取 {path:?}: {e}"))?;
            Ok(Some(bytes))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("打开 {path:?}: {e}")),
    }
}
#[pyfunction]
pub fn factor_store_storage_profile(path: String) -> PyResult<PyObject> {
    let p =
        StorageProfile::detect(Path::new(&path)).map_err(pyo3::exceptions::PyIOError::new_err)?;
    Python::with_gil(|py| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item(
            "kind",
            if p.network {
                "nas"
            } else if p.btrfs {
                "local_btrfs"
            } else {
                "local"
            },
        )?;
        d.set_item("network", p.network)?;
        d.set_item("btrfs", p.btrfs)?;
        d.set_item("preallocate", !p.network)?;
        d.set_item("io_concurrency", p.io_concurrency(192))?;
        d.set_item("io_batch_size", p.io_batch_size())?;
        d.set_item("io_batch_bytes", 64 * 1024 * 1024usize)?;
        d.set_item("delta_max_segments", 16usize)?;
        Ok(d.into())
    })
}

/// Restrict existing readers, not computation threads. No additional threads are spawned.
pub struct IoGate {
    limit: usize,
    active: std::sync::Mutex<usize>,
    wake: std::sync::Condvar,
}
pub struct IoPermit<'a> {
    gate: &'a IoGate,
    counted: bool,
}
impl IoGate {
    pub fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            active: std::sync::Mutex::new(0),
            wake: std::sync::Condvar::new(),
        }
    }
    pub fn acquire(&self) -> Result<IoPermit<'_>, String> {
        if self.limit == usize::MAX {
            return Ok(IoPermit {
                gate: self,
                counted: false,
            });
        }
        let mut n = self.active.lock().map_err(|e| e.to_string())?;
        while *n >= self.limit {
            n = self.wake.wait(n).map_err(|e| e.to_string())?;
        }
        *n += 1;
        Ok(IoPermit {
            gate: self,
            counted: true,
        })
    }
}
impl Drop for IoPermit<'_> {
    fn drop(&mut self) {
        if self.counted {
            let mut n = self.gate.active.lock().unwrap_or_else(|e| e.into_inner());
            *n -= 1;
            self.gate.wake.notify_one();
        }
    }
}
