//! Bounded, final column segments. Values reach disk once; the manifest is the
//! commit point. No result WAL, no automatic compaction, no final projection.
//! The value buffer is bounded separately from workers, queues and row indexes.
use crate::backup_reader::TaskResult;
use crate::storage_policy::{atomic_write, publish, read_optional, StorageProfile};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

pub const MANIFEST_NAME: &str = "segments.manifest.json";
#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentPart {
    pub file: String,
    pub rows: u64,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentManifest {
    pub version: u32,
    pub factor_names: Vec<String>,
    pub dates: Vec<i64>,
    pub codes: Vec<String>,
    pub parts: Vec<SegmentPart>,
    pub completed_dates: Vec<i64>,
}

pub fn load_manifest(dir: &Path) -> Result<Option<SegmentManifest>, String> {
    let Some(raw) = read_optional(&dir.join(MANIFEST_NAME))? else {
        return Ok(None);
    };
    let m: SegmentManifest =
        serde_json::from_slice(&raw).map_err(|e| format!("直接段清单损坏: {e}"))?;
    if m.version != 1 || m.factor_names.is_empty() {
        return Err("直接段清单版本或因子数无效".into());
    }
    if m.factor_names.iter().collect::<HashSet<_>>().len() != m.factor_names.len()
        || m.dates.iter().collect::<HashSet<_>>().len() != m.dates.len()
        || m.codes.iter().collect::<HashSet<_>>().len() != m.codes.len()
        || m.completed_dates.iter().collect::<HashSet<_>>().len() != m.completed_dates.len()
        || m.completed_dates.iter().any(|d| !m.dates.contains(d))
    {
        return Err("直接段清单字典重复或完成日期无效".into());
    }
    for (i, p) in m.parts.iter().enumerate() {
        if p.file != format!("segment_{i:08}.bin") || p.rows == 0 {
            return Err("直接段清单文件名/行数无效".into());
        }
    }
    Ok(Some(m))
}

/// Read only headers and compressed row IDs for recovery, never factor values.
fn read_keys(dir: &Path, m: &SegmentManifest) -> Result<HashSet<(u32, u32)>, String> {
    let mut keys = HashSet::new();
    for p in &m.parts {
        let path = dir.join(&p.file);
        let mut f = File::open(&path).map_err(|e| format!("打开已提交段 {path:?}: {e}"))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len();
        let mut h = [0u8; 32];
        f.read_exact(&mut h)
            .map_err(|e| format!("读取段头 {path:?}: {e}"))?;
        let u32_at = |n| u32::from_le_bytes(h[n..n + 4].try_into().unwrap());
        let u64_at = |n| u64::from_le_bytes(h[n..n + 8].try_into().unwrap());
        let fc = m.factor_names.len() as u64;
        let ro = u64_at(16);
        let csz = u64_at(24);
        let values = p.rows.checked_mul(4).ok_or("段行数溢出")?;
        let expected = ro
            .checked_add(csz)
            .and_then(|n| values.checked_mul(fc).and_then(|v| n.checked_add(v)));
        if u32_at(0) != 3
            || u32_at(4) as u64 != fc
            || u64_at(8) != p.rows
            || ro != 32 + 16 * fc
            || expected != Some(len)
            || csz == 0
        {
            return Err(format!("已提交段头/长度损坏: {path:?}"));
        }
        let mut expected_off = ro + csz;
        for _ in 0..fc {
            let mut ix = [0u8; 16];
            f.read_exact(&mut ix).map_err(|e| e.to_string())?;
            if u64::from_le_bytes(ix[..8].try_into().unwrap()) != expected_off
                || u64::from_le_bytes(ix[8..].try_into().unwrap()) != values
            {
                return Err(format!("已提交段列索引损坏: {path:?}"));
            }
            expected_off += values;
        }
        let row_bytes = p.rows.checked_mul(8).ok_or("行索引溢出")?;
        // Bound decompression, including malformed frames; no unbounded decode_all.
        let compressed = (&mut f).take(csz);
        let decoder = zstd::stream::read::Decoder::new(compressed).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        decoder
            .take(row_bytes + 1)
            .read_to_end(&mut raw)
            .map_err(|e| e.to_string())?;
        if raw.len() as u64 != row_bytes {
            return Err(format!("段行序损坏: {path:?}"));
        }
        for r in raw.chunks_exact(8) {
            let d = u32::from_le_bytes(r[..4].try_into().unwrap());
            let c = u32::from_le_bytes(r[4..].try_into().unwrap());
            if d as usize >= m.dates.len() || c as usize >= m.codes.len() || !keys.insert((d, c)) {
                return Err(format!("段行索引越界或重复: {path:?}"));
            }
        }
    }
    for date in &m.completed_dates {
        let id = m.dates.iter().position(|d| d == date).unwrap() as u32;
        if !keys.iter().any(|(d, _)| *d == id) {
            return Err(format!("完成日期 {date} 没有数据"));
        }
    }
    Ok(keys)
}

pub struct DirectSegmentWriter {
    dir: PathBuf,
    // The file handle owns an advisory exclusive lock for the full writer lifetime.
    _lock: File,
    manifest: SegmentManifest,
    date_ids: HashMap<i64, u32>,
    code_ids: HashMap<String, u32>,
    committed: HashSet<(u32, u32)>,
    pending: HashSet<(u32, u32)>,
    rows: Vec<(u32, u32)>,
    columns: Vec<Vec<f32>>,
    pending_dates: HashSet<i64>,
    max_rows: usize,
    poisoned: bool,
}

impl DirectSegmentWriter {
    pub fn open(dir: &str, names: &[String], buffer_bytes: usize) -> Result<Self, String> {
        if names.is_empty()
            || names.len() > u32::MAX as usize
            || names.iter().collect::<HashSet<_>>().len() != names.len()
        {
            return Err("直接写入要求非空、唯一的因子名".into());
        }
        let row_bytes = names.len().checked_mul(4).ok_or("因子数溢出")?;
        let max_rows = buffer_bytes / row_bytes;
        if max_rows == 0 {
            return Err(format!(
                "write_buffer_gb 太小：至少容纳一行 {row_bytes} 字节"
            ));
        }
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(dir.join(".segments.writer.lock"))
            .map_err(|e| e.to_string())?;
        #[cfg(unix)]
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(format!("直接段库已有写入者: {dir:?}"));
        }
        #[cfg(not(unix))]
        return Err("direct_write 目前要求支持 flock 的 Unix 系统".into());
        // Refuse mixed/legacy layouts; never silently create a second visible dataset.
        for e in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
            let name = e.map_err(|e| e.to_string())?.file_name();
            let s = name.to_string_lossy();
            let segment_name = s
                .strip_prefix("segment_")
                .and_then(|s| s.strip_suffix(".bin").or_else(|| s.strip_suffix(".tmp")));
            let allowed_segment = segment_name
                .map(|s| s.len() >= 8 && s.bytes().all(|c| c.is_ascii_digit()))
                .unwrap_or(false);
            if s != ".segments.writer.lock"
                && s != MANIFEST_NAME
                && s != "segments.manifest.atomic.tmp"
                && !allowed_segment
            {
                return Err(format!(
                    "direct_write 需要新的直接段目录，发现不兼容文件: {s}"
                ));
            }
        }
        let manifest = match load_manifest(&dir)? {
            Some(m) => {
                if m.factor_names != names {
                    return Err("直接段因子名或顺序不一致".into());
                }
                m
            }
            None => {
                // A missing manifest with final data must not be mistaken for an empty store.
                if std::fs::read_dir(&dir)
                    .map_err(|e| e.to_string())?
                    .any(|e| {
                        e.map(|e| e.path().extension().is_some_and(|x| x == "bin"))
                            .unwrap_or(true)
                    })
                {
                    return Err("直接段清单缺失但存在段文件，拒绝覆盖".into());
                }
                let m = SegmentManifest {
                    version: 1,
                    factor_names: names.to_vec(),
                    dates: vec![],
                    codes: vec![],
                    parts: vec![],
                    completed_dates: vec![],
                };
                atomic_write(
                    &dir.join(MANIFEST_NAME),
                    &serde_json::to_vec(&m).map_err(|e| e.to_string())?,
                )?;
                m
            }
        };
        let committed = read_keys(&dir, &manifest)?;
        Ok(Self {
            date_ids: manifest
                .dates
                .iter()
                .enumerate()
                .map(|(i, d)| (*d, i as u32))
                .collect(),
            code_ids: manifest
                .codes
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i as u32))
                .collect(),
            columns: (0..names.len()).map(|_| Vec::new()).collect(),
            dir,
            _lock: lock,
            manifest,
            committed,
            pending: HashSet::new(),
            rows: vec![],
            pending_dates: HashSet::new(),
            max_rows,
            poisoned: false,
        })
    }
    fn validate(&self, batch: &[TaskResult]) -> Result<(), String> {
        if self.poisoned {
            return Err("直接段写入此前失败；请关闭并重新打开后续算".into());
        }
        let mut seen = HashSet::new();
        let completed_dates: HashSet<_> = self.manifest.completed_dates.iter().copied().collect();
        for r in batch {
            if r.facs.len() != self.columns.len() || r.code.is_empty() {
                return Err(format!(
                    "直接段结果格式错误: date={}, code={}, columns={}，期望 {}",
                    r.date,
                    r.code,
                    r.facs.len(),
                    self.columns.len()
                ));
            }
            if !seen.insert((r.date, r.code.as_str())) {
                return Err("批次中 (date, code) 重复".into());
            }
            if completed_dates.contains(&r.date) {
                let key = self.date_ids.get(&r.date).zip(self.code_ids.get(&r.code));
                if !key.is_some_and(|(d, c)| self.committed.contains(&(*d, *c))) {
                    return Err(format!(
                        "日期 {} 已整日提交，不能追加新股票 {}",
                        r.date, r.code
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn append_batch(&mut self, batch: &[TaskResult]) -> Result<(), String> {
        self.validate(batch)?;
        let result = self.append_validated(batch);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn append_validated(&mut self, batch: &[TaskResult]) -> Result<(), String> {
        // Reserve exact column capacities once, then reuse. RSS may remain at a
        // plateau even when valid payload drops after publishing a segment.
        if !batch.is_empty() && self.columns[0].capacity() == 0 {
            for c in &mut self.columns {
                c.try_reserve_exact(self.max_rows)
                    .map_err(|e| format!("分配直接段缓冲失败: {e}"))?;
            }
        }
        let mut offset = 0;
        while offset < batch.len() {
            let take = (self.max_rows - self.rows.len()).min(batch.len() - offset);
            let mut accepted = Vec::with_capacity(take);
            for r in &batch[offset..offset + take] {
                let d = match self.date_ids.get(&r.date) {
                    Some(d) => *d,
                    None => {
                        let d =
                            u32::try_from(self.manifest.dates.len()).map_err(|_| "日期字典溢出")?;
                        self.manifest.dates.push(r.date);
                        self.date_ids.insert(r.date, d);
                        d
                    }
                };
                let c = match self.code_ids.get(&r.code) {
                    Some(c) => *c,
                    None => {
                        let c =
                            u32::try_from(self.manifest.codes.len()).map_err(|_| "股票字典溢出")?;
                        self.manifest.codes.push(r.code.clone());
                        self.code_ids.insert(r.code.clone(), c);
                        c
                    }
                };
                if !self.committed.contains(&(d, c)) && self.pending.insert((d, c)) {
                    self.rows.push((d, c));
                    accepted.push(r);
                }
            }
            for (i, c) in self.columns.iter_mut().enumerate() {
                c.extend(accepted.iter().map(|r| r.facs[i]));
            }
            offset += take;
            if self.rows.len() == self.max_rows {
                self.flush()?;
            }
        }
        Ok(())
    }
    pub fn append_date(&mut self, date: i64, batch: &[TaskResult]) -> Result<(), String> {
        self.validate(batch)?;
        if batch.is_empty() || batch.iter().any(|r| r.date != date) {
            return Err("整日直接写入要求非空且日期一致".into());
        }
        if let Some(d) = self.date_ids.get(&date) {
            let codes: HashSet<_> = batch.iter().map(|r| r.code.as_str()).collect();
            if self
                .committed
                .iter()
                .chain(self.pending.iter())
                .any(|(di, ci)| {
                    di == d && !codes.contains(self.manifest.codes[*ci as usize].as_str())
                })
            {
                return Err(format!("日期 {date} 重算结果缺少已有股票，拒绝标记完成"));
            }
        }
        self.append_batch(batch)?;
        self.pending_dates.insert(date);
        // Exact-full segments may already have committed all rows above.
        if self.rows.is_empty() {
            self.finish()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        let mut next = self.manifest.clone();
        if !self.rows.is_empty() {
            let file = format!("segment_{:08}.bin", next.parts.len());
            let target = self.dir.join(&file);
            let tmp = target.with_extension("tmp");
            write_segment(&tmp, &self.rows, &self.columns)?;
            publish(&tmp, &target)?;
            next.parts.push(SegmentPart {
                file,
                rows: self.rows.len() as u64,
            });
        }
        next.completed_dates
            .extend(self.pending_dates.iter().copied());
        next.completed_dates.sort_unstable();
        next.completed_dates.dedup();
        // Values and directory entry are durable before the sole commit point.
        atomic_write(
            &self.dir.join(MANIFEST_NAME),
            &serde_json::to_vec(&next).map_err(|e| e.to_string())?,
        )?;
        self.manifest = next;
        self.committed.extend(self.pending.drain());
        self.pending_dates.clear();
        self.rows.clear();
        for c in &mut self.columns {
            c.clear();
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<(), String> {
        if self.poisoned {
            return Err("直接段写入此前失败；拒绝继续提交".into());
        }
        if self.rows.is_empty() && self.pending_dates.is_empty() {
            return Ok(());
        }
        let result = self.flush();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub fn check_completed(&self) -> Result<HashSet<(i64, String)>, String> {
        Ok(self
            .committed
            .iter()
            .map(|(d, c)| {
                (
                    self.manifest.dates[*d as usize],
                    self.manifest.codes[*c as usize].clone(),
                )
            })
            .collect())
    }
    pub fn completed_dates(&self) -> HashSet<i64> {
        self.manifest.completed_dates.iter().copied().collect()
    }
    pub fn record_count(&self) -> u64 {
        self.committed.len() as u64
    }
    pub fn store_dir(&self) -> &Path {
        &self.dir
    }
}

fn write_segment(path: &Path, rows: &[(u32, u32)], columns: &[Vec<f32>]) -> Result<(), String> {
    let mut raw = Vec::with_capacity(rows.len() * 8);
    for (d, c) in rows {
        raw.extend_from_slice(&d.to_le_bytes());
        raw.extend_from_slice(&c.to_le_bytes());
    }
    let ro = zstd::encode_all(&raw[..], 1).map_err(|e| e.to_string())?;
    drop(raw);
    let header_len = 32 + columns.len() as u64 * 16;
    let column_len = rows.len() as u64 * 4;
    let mut off = header_len + ro.len() as u64;
    let mut header = Vec::with_capacity(header_len as usize);
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(columns.len() as u32).to_le_bytes());
    header.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    header.extend_from_slice(&header_len.to_le_bytes());
    header.extend_from_slice(&(ro.len() as u64).to_le_bytes());
    for c in columns {
        if c.len() != rows.len() {
            return Err("直接段列长度不一致".into());
        }
        header.extend_from_slice(&off.to_le_bytes());
        header.extend_from_slice(&column_len.to_le_bytes());
        off = off.checked_add(column_len).ok_or("直接段文件长度溢出")?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let profile = StorageProfile::detect(path)?;
    #[cfg(target_os = "linux")]
    {
        if profile.btrfs {
            let output = std::process::Command::new("chattr")
                .arg("+C")
                .arg(path)
                .output()
                .map_err(|e| e.to_string())?;
            if !output.status.success() {
                return Err(format!(
                    "直接段 chattr +C 失败: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }
        if !profile.network && off <= i64::MAX as u64 {
            let rc = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, off as libc::off_t) };
            if rc != 0 && rc != libc::EOPNOTSUPP && rc != libc::ENOSYS {
                return Err(format!(
                    "直接段预留磁盘空间失败: {}",
                    std::io::Error::from_raw_os_error(rc)
                ));
            }
        }
    }
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.write_all(&header)
        .and_then(|_| file.write_all(&ro))
        .map_err(|e| e.to_string())?;
    // No full-size transposition copy. BufWriter coalesces short columns into
    // 8 MiB writes while borrowing the existing column memory on little endian.
    let mut out = std::io::BufWriter::with_capacity(8 * 1024 * 1024, file);
    let mut unsynced = 0usize;
    for c in columns {
        #[cfg(target_endian = "little")]
        {
            // f32 has no padding; bytes live for this synchronous write call.
            let bytes = unsafe { std::slice::from_raw_parts(c.as_ptr().cast::<u8>(), c.len() * 4) };
            for chunk in bytes.chunks(64 * 1024 * 1024) {
                out.write_all(chunk).map_err(|e| e.to_string())?;
                unsynced += chunk.len();
                if unsynced >= 1024 * 1024 * 1024 {
                    out.flush().map_err(|e| e.to_string())?;
                    crate::storage_policy::sync_file(out.get_ref(), "直接段")?;
                    unsynced = 0;
                }
            }
        }
        #[cfg(target_endian = "big")]
        for value in c {
            out.write_all(&value.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    out.get_ref().set_len(off).map_err(|e| e.to_string())?;
    crate::storage_policy::sync_file(out.get_ref(), "直接段")
}
