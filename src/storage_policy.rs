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
    /// 同时进行的 IO 批次上限。
    ///
    /// 本地盘以前直接放行 n_jobs。2026-11-13 实测（任务 590，n_jobs=300、store 落在
    /// /hdd 的 8 块机械盘上）：Engine 阶段 42924 个因子 3h59m41s 才读完 1.8 TB，
    /// 等效 125 MB/s，远低于 8 盘合计带宽；同一时刻 iostat 里 8 块盘只有 1~3 块在动、
    /// %util 18~20%、aqu-sz 0.62，说明瓶颈是并发过高压出来的寻道排队，不是带宽。
    /// 机械盘要的是一条长顺序流，不是 300 条随机流；配合 io_batch_bytes 调大、
    /// 让一个批次覆盖多列连续因子，限到 16 条并发流才跑得起来。
    ///
    /// 网络盘维持原来的 8（远程存储上高并发反而更容易超时）。
    pub fn io_concurrency(self, jobs: usize) -> usize {
        if self.network {
            jobs.min(8).max(1)
        } else {
            jobs.min(16).max(1)
        }
    }
    /// 单次顺序连读的因子列数上限（实际列数还会受 io_batch_bytes 字节预算约束）。
    ///
    /// 本地盘此前写死为 1，导致即便把 io_batch_bytes 调到 512 MiB，storage_read_cache 计算
    /// `max_cols = min(io_batch_size, io_batch_bytes / matrix_bytes)` 后仍恒等于 1，
    /// 批量连读分支一次都不会进入。本地机械盘放开到 8 列（配合 512 MiB 字节预算，单段每次
    /// 顺序连读约 6~8 MiB 连续块，既将寻道与系统调用降约 8 倍、避开 >32 MiB 临时分配的
    /// mmap/TLB 抖动，又保证 n_jobs=128~192 时 16~24 个读取组能持续喂满 16 个并发 IO 槽位）。
    pub fn io_batch_size(self) -> usize {
        if self.network {
            4
        } else {
            8
        }
    }
    /// 单次顺序连读的字节预算上限。
    ///
    /// 本地盘默认 512 MiB（常规全历史 ~45~55 MiB/列下匹配 8 列上限），NAS 默认 256 MiB（匹配 4 列上限）。
    pub fn io_batch_bytes(self) -> usize {
        if self.network {
            256 * 1024 * 1024
        } else {
            512 * 1024 * 1024
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
        d.set_item("io_batch_bytes", p.io_batch_bytes())?;
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
