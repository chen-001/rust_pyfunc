//! 进程间 `/dev/shm` 共享内存结果通道 + 双路处理器 NUMA 就近绑定。
//!
//! 设计要点：
//! 1. **零残留 `/dev/shm` 共享槽**：父进程用 `O_CREAT | O_EXCL | O_CLOEXEC` 创建唯一文件（避免并发 spawn 时
//!    无关子进程继承 fd），子进程打开后**双方立即 unlink 目录项**，仅保留各自进程内的打开句柄与 mmap。
//!    即便父子进程被 `SIGKILL (-9)` 强杀，内核也会在句柄关闭瞬间自动回收全部共享内存页，`/dev/shm` 绝不残留孤儿文件。
//! 2. **防 `SIGBUS` 预分配**：扩容时先调 `posix_fallocate` 实占 tmpfs 页，若 `/dev/shm` 空间不足直接返回普通 IO 错误并自动回退管道通信。
//! 3. **双路处理器 NUMA 分片就近绑定**：严格与父进程当前 `sched_getaffinity` 可用核集求交集（完全兼容任务管理器 / 容器 cpuset），
//!    将每个横截面多线程 worker 或无单核绑定的单股 worker 固定在单颗处理器芯片内部，消除跨片线程漂移与远端内存访问。

use crate::backup_reader::TaskResult;
use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const ENV_SHM_PATH: &str = "RUST_PYFUNC_SHM_PATH";
pub const ENV_SHM_IPC: &str = "RUST_PYFUNC_SHM_IPC";
pub const ENV_NUMA_CPULIST: &str = "RUST_PYFUNC_NUMA_CPULIST";
pub const ENV_NUMA_BIND: &str = "RUST_PYFUNC_NUMA_BIND";

const DAY_BATCH_MAGIC: u64 = 0x5250_5348_4D42_3031; // "RPSHMB01"
const SMALL_ALIGN: usize = 64 * 1024; // 64 KiB
const LARGE_ALIGN: usize = 2 * 1024 * 1024; // 2 MiB（对齐透明大页）
const LARGE_THRESHOLD: usize = 4 * 1024 * 1024;

static SHM_SLOT_SEQ: AtomicU64 = AtomicU64::new(1);

fn env_enabled(var: &str, default_on: bool) -> bool {
    match std::env::var(var) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            if s.is_empty() {
                default_on
            } else {
                !matches!(s.as_str(), "0" | "false" | "off" | "no")
            }
        }
        Err(_) => default_on,
    }
}

#[inline]
fn align_up(val: usize, align: usize) -> usize {
    let rem = val % align;
    if rem == 0 {
        val
    } else {
        val.saturating_add(align - rem)
    }
}

#[inline]
fn round_slot_capacity(needed: usize) -> usize {
    let min_needed = needed.max(SMALL_ALIGN);
    if min_needed <= LARGE_THRESHOLD {
        align_up(min_needed, SMALL_ALIGN)
    } else {
        align_up(min_needed, LARGE_ALIGN)
    }
}

/// 父进程持有的单 worker 专属 `/dev/shm` 共享槽（一对一无锁交替读写）。
pub struct ShmParentSlot {
    file: File,
    path: PathBuf,
    unlinked: bool,
    mmap: Option<Mmap>,
    cap: usize,
}

impl ShmParentSlot {
    #[cfg(target_os = "linux")]
    pub fn create(worker_idx: usize) -> Option<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        if !env_enabled(ENV_SHM_IPC, true) {
            return None;
        }
        let shm_dir = Path::new("/dev/shm");
        if !shm_dir.is_dir() {
            return None;
        }
        let pid = std::process::id();
        let seq = SHM_SLOT_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = shm_dir.join(format!("rp_ipc_{pid}_{seq}_{worker_idx}"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .ok()?;
        Some(Self {
            file,
            path,
            unlinked: false,
            mmap: None,
            cap: 0,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn create(_worker_idx: usize) -> Option<Self> {
        None
    }

    pub fn path_str(&self) -> &str {
        self.path.to_str().unwrap_or("")
    }

    /// 握手完成后立即删除 `/dev/shm` 目录项；父子进程仍通过已打开 fd 共享同一内核 shmem 对象。
    pub fn unlink_now(&mut self) {
        if !self.unlinked {
            let _ = std::fs::remove_file(&self.path);
            self.unlinked = true;
        }
    }

    fn ensure_mapped(&mut self, slot_cap: u64, used_bytes: u64) -> Result<&[u8], String> {
        if used_bytes > slot_cap {
            return Err(format!(
                "共享内存长度越界: used_bytes={used_bytes} > slot_cap={slot_cap}"
            ));
        }
        let cap_usize = usize::try_from(slot_cap).map_err(|_| "slot_cap 溢出".to_string())?;
        let used_usize = usize::try_from(used_bytes).map_err(|_| "used_bytes 溢出".to_string())?;
        if self.cap < cap_usize || self.mmap.is_none() {
            self.mmap = None;
            self.cap = 0;
            let file_len = self
                .file
                .metadata()
                .map_err(|e| format!("读取共享内存元数据失败: {e}"))?
                .len();
            if file_len < slot_cap {
                return Err(format!(
                    "共享内存文件实际长度不足: file_len={file_len} < slot_cap={slot_cap}"
                ));
            }
            let mmap = unsafe {
                Mmap::map(&self.file).map_err(|e| format!("映射共享内存失败: {e}"))?
            };
            self.cap = mmap.len();
            self.mmap = Some(mmap);
        }
        let buf = self.mmap.as_ref().unwrap();
        if used_usize > buf.len() {
            return Err(format!(
                "共享内存映射长度不足: used={used_usize} > mapped={}",
                buf.len()
            ));
        }
        Ok(&buf[..used_usize])
    }

    /// 读取单股结果 `Vec<f32>`。
    pub fn read_stock_f32(&mut self, slot_cap: u64, n_floats: u64) -> Result<Vec<f32>, String> {
        let n = usize::try_from(n_floats).map_err(|_| "n_floats 溢出".to_string())?;
        let byte_len = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "单股字节数溢出".to_string())?;
        let raw = self.ensure_mapped(slot_cap, byte_len as u64)?;
        let mut out = Vec::<f32>::with_capacity(n);
        if n > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr() as *const f32,
                    out.as_mut_ptr(),
                    n,
                );
                out.set_len(n);
            }
        }
        Ok(out)
    }

    /// 零拷贝借用整日横截面/分钟结果：返回 `(股票代码切片列表, 行优先连续 f32 切片, 列数)`。
    pub fn borrow_day_batch(
        &mut self,
        slot_cap: u64,
        used_bytes: u64,
    ) -> Result<(Vec<&str>, &[f32], usize), String> {
        if used_bytes < 32 {
            return Err(format!("整日共享内存头过短: {used_bytes}"));
        }
        let raw = self.ensure_mapped(slot_cap, used_bytes)?;
        let u64_at = |off: usize| -> u64 {
            u64::from_le_bytes(raw[off..off + 8].try_into().unwrap())
        };
        let magic = u64_at(0);
        if magic != DAY_BATCH_MAGIC {
            return Err(format!("整日共享内存魔数不匹配: {magic:#x}"));
        }
        let n_rows = usize::try_from(u64_at(8)).map_err(|_| "n_rows 溢出".to_string())?;
        let n_cols = usize::try_from(u64_at(16)).map_err(|_| "n_cols 溢出".to_string())?;
        let codes_len = usize::try_from(u64_at(24)).map_err(|_| "codes_len 溢出".to_string())?;
        let codes_end = 32usize
            .checked_add(codes_len)
            .ok_or_else(|| "codes_end 溢出".to_string())?;
        let vals_offset = align_up(codes_end, 64);
        let n_floats = n_rows
            .checked_mul(n_cols)
            .ok_or_else(|| "n_rows*n_cols 溢出".to_string())?;
        let vals_bytes = n_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "vals_bytes 溢出".to_string())?;
        let expected_used = vals_offset
            .checked_add(vals_bytes)
            .ok_or_else(|| "expected_used 溢出".to_string())?;
        if expected_used != raw.len() {
            return Err(format!(
                "整日共享内存长度校验失败: expected={expected_used}, got={}",
                raw.len()
            ));
        }
        let codes_str = std::str::from_utf8(&raw[32..codes_end])
            .map_err(|e| format!("整日共享内存股票代码 UTF-8 非法: {e}"))?;
        let codes: Vec<&str> = if n_rows == 0 {
            Vec::new()
        } else {
            codes_str.split('\n').collect()
        };
        if codes.len() != n_rows {
            return Err(format!(
                "整日共享内存股票数不一致: codes={} != n_rows={n_rows}",
                codes.len()
            ));
        }
        let vals_ptr = unsafe { raw.as_ptr().add(vals_offset) };
        if (vals_ptr as usize) % std::mem::align_of::<f32>() != 0 {
            return Err("整日共享内存浮点区未对齐".to_string());
        }
        let vals = unsafe { std::slice::from_raw_parts(vals_ptr as *const f32, n_floats) };
        Ok((codes, vals, n_cols))
    }
}

impl Drop for ShmParentSlot {
    fn drop(&mut self) {
        self.unlink_now();
    }
}

/// Worker 子进程持有的 `/dev/shm` 写入槽。
pub struct ShmWorkerSlot {
    file: File,
    mmap: Option<MmapMut>,
    cap: usize,
}

impl ShmWorkerSlot {
    #[cfg(target_os = "linux")]
    pub fn open_from_env() -> Option<Self> {
        let path = std::env::var(ENV_SHM_PATH).ok()?;
        if path.is_empty() {
            return None;
        }
        let file = OpenOptions::new().read(true).write(true).open(&path).ok()?;
        // 打开即 unlink，确保崩溃或 SIGKILL 时不留任何 /dev/shm 文件。
        let _ = std::fs::remove_file(&path);
        Some(Self {
            file,
            mmap: None,
            cap: 0,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn open_from_env() -> Option<Self> {
        None
    }

    #[cfg(target_os = "linux")]
    fn ensure_capacity(&mut self, needed: usize) -> std::io::Result<&mut [u8]> {
        use std::os::unix::io::AsRawFd;
        let target_cap = round_slot_capacity(needed);
        if self.cap < target_cap || self.mmap.is_none() {
            self.mmap = None;
            self.cap = 0;
            let off_len = libc::off_t::try_from(target_cap).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "target_cap 超出 off_t")
            })?;
            let rc = unsafe { libc::posix_fallocate(self.file.as_raw_fd(), 0, off_len) };
            if rc != 0 {
                return Err(std::io::Error::from_raw_os_error(rc));
            }
            let mmap = unsafe { MmapMut::map_mut(&self.file)? };
            self.cap = mmap.len();
            self.mmap = Some(mmap);
        }
        Ok(&mut self.mmap.as_mut().unwrap()[..needed])
    }

    #[cfg(not(target_os = "linux"))]
    fn ensure_capacity(&mut self, _needed: usize) -> std::io::Result<&mut [u8]> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "仅 Linux 支持 /dev/shm 槽",
        ))
    }

    /// 将单股 `f32` 结果直接写入共享内存，返回 `(slot_cap, n_floats)`。
    pub fn write_stock_f32(&mut self, vals: &[f32]) -> std::io::Result<(u64, u64)> {
        let byte_len = vals
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "单股长度溢出"))?;
        let alloc_len = byte_len.max(64);
        let cap_after = round_slot_capacity(alloc_len).max(self.cap);
        let buf = self.ensure_capacity(alloc_len)?;
        if !vals.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    vals.as_ptr() as *const u8,
                    buf.as_mut_ptr(),
                    byte_len,
                );
            }
        }
        let actual_cap = self.cap.max(cap_after);
        Ok((actual_cap as u64, vals.len() as u64))
    }

    /// 将整日 `TaskResult` 批量直接铺平写入共享内存，返回 `Some((slot_cap, used_bytes))`。
    /// 若为空批或存在非常规列长，返回 `Ok(None)` 由调用方走原管道分支。
    pub fn write_day_batch(
        &mut self,
        batch: &[TaskResult],
        expected_cols: usize,
    ) -> std::io::Result<Option<(u64, u64)>> {
        if batch.is_empty() || expected_cols == 0 {
            return Ok(None);
        }
        let n_rows = batch.len();
        let mut codes_len = n_rows.saturating_sub(1); // '\n' 分隔符
        for r in batch {
            if r.facs.len() != expected_cols || r.code.is_empty() || r.code.contains('\n') {
                return Ok(None);
            }
            codes_len = codes_len.saturating_add(r.code.len());
        }
        let codes_end = 32usize.checked_add(codes_len).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "codes_end 溢出")
        })?;
        let vals_offset = align_up(codes_end, 64);
        let n_floats = n_rows.checked_mul(expected_cols).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "矩阵元素数溢出")
        })?;
        let vals_bytes = n_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "矩阵字节数溢出")
            })?;
        let used_bytes = vals_offset.checked_add(vals_bytes).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "used_bytes 溢出")
        })?;

        let buf = self.ensure_capacity(used_bytes)?;
        buf[0..8].copy_from_slice(&DAY_BATCH_MAGIC.to_le_bytes());
        buf[8..16].copy_from_slice(&(n_rows as u64).to_le_bytes());
        buf[16..24].copy_from_slice(&(expected_cols as u64).to_le_bytes());
        buf[24..32].copy_from_slice(&(codes_len as u64).to_le_bytes());

        let mut pos = 32usize;
        for (idx, r) in batch.iter().enumerate() {
            if idx > 0 {
                buf[pos] = b'\n';
                pos += 1;
            }
            let cb = r.code.as_bytes();
            buf[pos..pos + cb.len()].copy_from_slice(cb);
            pos += cb.len();
        }
        for b in &mut buf[codes_end..vals_offset] {
            *b = 0;
        }

        let vals_dst = unsafe {
            std::slice::from_raw_parts_mut(
                buf.as_mut_ptr().add(vals_offset) as *mut f32,
                n_floats,
            )
        };
        if vals_bytes >= 16 * 1024 * 1024 && rayon::current_num_threads() > 1 {
            vals_dst
                .par_chunks_mut(expected_cols)
                .zip(batch.par_iter())
                .for_each(|(dst_row, r)| {
                    dst_row.copy_from_slice(&r.facs);
                });
        } else {
            for (dst_row, r) in vals_dst.chunks_mut(expected_cols).zip(batch.iter()) {
                dst_row.copy_from_slice(&r.facs);
            }
        }
        Ok(Some((self.cap as u64, used_bytes as u64)))
    }
}

// ============================================================================
// 双路处理器 NUMA 就近核集规划与绑定
// ============================================================================

#[derive(Clone, Debug)]
pub struct NumaNodeInfo {
    pub node_id: usize,
    pub allowed_logical: Vec<usize>,
    pub physical_cores: usize,
    pub cpulist: String,
}

/// 解析 Linux cpulist 格式（如 `"0-107,256-363"`）。
pub fn parse_cpulist(text: &str) -> Vec<usize> {
    let mut out = BTreeSet::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                if start <= end && end < 4096 {
                    for c in start..=end {
                        out.insert(c);
                    }
                }
            }
        } else if let Ok(c) = part.parse::<usize>() {
            if c < 4096 {
                out.insert(c);
            }
        }
    }
    out.into_iter().collect()
}

/// 压缩有序 CPU 列表为紧凑 cpulist 字符串（如 `[0,1,2,256,257]` → `"0-2,256-257"`）。
pub fn format_cpulist(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];
    for &c in &cpus[1..] {
        if c == prev + 1 {
            prev = c;
        } else {
            if start == prev {
                parts.push(start.to_string());
            } else {
                parts.push(format!("{start}-{prev}"));
            }
            start = c;
            prev = c;
        }
    }
    if start == prev {
        parts.push(start.to_string());
    } else {
        parts.push(format!("{start}-{prev}"));
    }
    parts.join(",")
}

#[cfg(target_os = "linux")]
fn current_allowed_cpus() -> BTreeSet<usize> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set)
    };
    let mut out = BTreeSet::new();
    if rc == 0 {
        let max_cpus = std::mem::size_of::<libc::cpu_set_t>() * 8;
        for cpu in 0..max_cpus {
            if unsafe { libc::CPU_ISSET(cpu, &set) } {
                out.insert(cpu);
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn count_physical_cores(logical_cpus: &[usize]) -> usize {
    let mut reps = BTreeSet::new();
    for &cpu in logical_cpus {
        let sib_path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        if let Ok(s) = std::fs::read_to_string(&sib_path) {
            let sibs = parse_cpulist(&s);
            if let Some(&first) = sibs.first() {
                reps.insert(first);
                continue;
            }
        }
        reps.insert(cpu);
    }
    reps.len().max(1)
}

#[cfg(target_os = "linux")]
pub fn detect_allowed_numa_nodes() -> Vec<NumaNodeInfo> {
    let allowed = current_allowed_cpus();
    if allowed.is_empty() {
        return Vec::new();
    }
    let mut nodes = Vec::new();
    for node_id in 0..16usize {
        let p = format!("/sys/devices/system/node/node{node_id}/cpulist");
        let Ok(raw) = std::fs::read_to_string(&p) else {
            continue;
        };
        let node_cpus: Vec<usize> = parse_cpulist(&raw)
            .into_iter()
            .filter(|c| allowed.contains(c))
            .collect();
        if node_cpus.is_empty() {
            continue;
        }
        let physical_cores = count_physical_cores(&node_cpus);
        let cpulist = format_cpulist(&node_cpus);
        nodes.push(NumaNodeInfo {
            node_id,
            allowed_logical: node_cpus,
            physical_cores,
            cpulist,
        });
    }
    nodes
}

#[cfg(not(target_os = "linux"))]
pub fn detect_allowed_numa_nodes() -> Vec<NumaNodeInfo> {
    Vec::new()
}

/// 为 `n_workers` 个子进程（每进程 `threads_per_worker` 线程）规划 NUMA 芯片内 CPU 掩码。
///
/// 规则：
/// - 若 `RUST_PYFUNC_NUMA_BIND=0`、或可用 NUMA 节点 < 2（单路机器 / 容器已被限制在单芯片内），返回 `None`。
/// - 单进程（`n_workers == 1`）仅当某单颗芯片的可用**物理核** ≥ `threads_per_worker` 时才绑定到该芯片，
///   避免把 200 线程硬塞进 128 物理核的单颗芯片。
/// - 多进程（`n_workers >= 2`）轮询分配到可用逻辑核充足的各个芯片；多线程 worker 额外要求每个芯片分到的总线程数
///   不超过该芯片的可用逻辑核数，绝不制造芯片级超订。
pub fn plan_worker_numa_cpulists(
    n_workers: usize,
    threads_per_worker: usize,
    mode_label: &str,
) -> Option<Vec<String>> {
    if !env_enabled(ENV_NUMA_BIND, true) || n_workers == 0 || threads_per_worker == 0 {
        return None;
    }
    let nodes = detect_allowed_numa_nodes();
    if nodes.len() < 2 {
        return None;
    }
    if n_workers == 1 {
        let best = nodes
            .iter()
            .filter(|n| n.physical_cores >= threads_per_worker)
            .max_by_key(|n| n.physical_cores)?;
        println!(
            "📌 NUMA 就近绑定 ({mode_label}): 单进程 {threads_per_worker} 线程固定于 node{}（可用 {} 物理核 / {} 逻辑核）",
            best.node_id,
            best.physical_cores,
            best.allowed_logical.len()
        );
        return Some(vec![best.cpulist.clone()]);
    }

    let eligible: Vec<&NumaNodeInfo> = nodes
        .iter()
        .filter(|n| n.allowed_logical.len() >= threads_per_worker)
        .collect();
    if eligible.len() < 2 {
        return None;
    }

    let mut counts = vec![0usize; eligible.len()];
    let mut plan = Vec::with_capacity(n_workers);
    for w in 0..n_workers {
        let idx = w % eligible.len();
        counts[idx] += 1;
        plan.push(eligible[idx].cpulist.clone());
    }

    if threads_per_worker > 1 {
        for (idx, &cnt) in counts.iter().enumerate() {
            if cnt.saturating_mul(threads_per_worker) > eligible[idx].allowed_logical.len() {
                return None;
            }
        }
    }

    let summary: Vec<String> = eligible
        .iter()
        .zip(counts.iter())
        .map(|(n, &cnt)| {
            format!(
                "node{}({}物理核/{}逻辑核)×{}进程",
                n.node_id,
                n.physical_cores,
                n.allowed_logical.len(),
                cnt
            )
        })
        .collect();
    println!(
        "📌 NUMA 就近绑定 ({mode_label}): {} 进程按芯片分片 → {}",
        n_workers,
        summary.join(", ")
    );
    Some(plan)
}

/// 在 worker 子进程入口（Rayon 全局池初始化之前）应用单核或 NUMA 芯片级亲和性。
pub fn apply_worker_affinity_from_env() {
    if let Ok(idx_str) = std::env::var("RUST_PYFUNC_CORE_AFFINITY_IDX") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            let core_ids = core_affinity::get_core_ids().unwrap_or_default();
            if idx < core_ids.len() {
                let _ = core_affinity::set_for_current(core_ids[idx]);
                return;
            }
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(cpulist) = std::env::var(ENV_NUMA_CPULIST) {
        let cpus = parse_cpulist(&cpulist);
        if !cpus.is_empty() {
            let max_cpus = std::mem::size_of::<libc::cpu_set_t>() * 8;
            let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            let mut any = false;
            for cpu in cpus {
                if cpu < max_cpus {
                    unsafe { libc::CPU_SET(cpu, &mut set) };
                    any = true;
                }
            }
            if any {
                unsafe {
                    let _ = libc::sched_setaffinity(
                        0,
                        std::mem::size_of::<libc::cpu_set_t>(),
                        &set,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpulist_roundtrip() {
        let parsed = parse_cpulist("0-3, 8, 256-258");
        assert_eq!(parsed, vec![0, 1, 2, 3, 8, 256, 257, 258]);
        assert_eq!(format_cpulist(&parsed), "0-3,8,256-258");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_shm_stock_and_day_roundtrip_and_unlink() {
        let mut parent = ShmParentSlot::create(999).expect("create shm slot");
        let path = PathBuf::from(parent.path_str());
        assert!(path.exists());
        std::env::set_var(ENV_SHM_PATH, parent.path_str());
        let mut worker = ShmWorkerSlot::open_from_env().expect("open worker slot");
        // worker open 即 unlink
        assert!(!path.exists());
        parent.unlink_now();

        let stock_vals = vec![1.25f32, f32::NAN, -3.5, 0.0, 42.0];
        let (cap, n) = worker.write_stock_f32(&stock_vals).unwrap();
        let got = parent.read_stock_f32(cap, n).unwrap();
        assert_eq!(got.len(), stock_vals.len());
        assert_eq!(got[0], 1.25);
        assert!(got[1].is_nan());
        assert_eq!(got[2], -3.5);
        assert_eq!(got[4], 42.0);

        let batch = vec![
            TaskResult {
                date: 20241231,
                code: "000001".to_string(),
                timestamp: 0,
                facs: vec![1.0, 2.0, 3.0],
            },
            TaskResult {
                date: 20241231,
                code: "600519".to_string(),
                timestamp: 0,
                facs: vec![4.0, f32::NAN, 6.0],
            },
        ];
        let (cap2, used2) = worker.write_day_batch(&batch, 3).unwrap().unwrap();
        let (codes, flat, ncols) = parent.borrow_day_batch(cap2, used2).unwrap();
        assert_eq!(codes, vec!["000001", "600519"]);
        assert_eq!(ncols, 3);
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[2], 3.0);
        assert_eq!(flat[3], 4.0);
        assert!(flat[4].is_nan());
        assert_eq!(flat[5], 6.0);
    }
}
