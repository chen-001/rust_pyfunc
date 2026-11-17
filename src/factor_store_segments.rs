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

/// 4 通道字节分离（Byte-Shuffle）+ Zstd(level=1) 自适应无损列压缩（首字节 0x01 为帧标记）。
/// 仅当压缩后字节数严格小于原始字节数的 90% 时返回 `Some(compressed)`，否则返回 `None` 存裸 f32。
pub(crate) fn encode_f32_column_adaptive(col: &[f32]) -> Option<Vec<u8>> {
    let n = col.len();
    let raw_len = n.checked_mul(4)?;
    if n < 64 {
        return None;
    }
    let mut shuffled = vec![0u8; raw_len];
    let (b0, rest) = shuffled.split_at_mut(n);
    let (b1, rest) = rest.split_at_mut(n);
    let (b2, b3) = rest.split_at_mut(n);
    for (i, &val) in col.iter().enumerate() {
        let le = val.to_le_bytes();
        b0[i] = le[0];
        b1[i] = le[1];
        b2[i] = le[2];
        b3[i] = le[3];
    }
    let max_dst = zstd::zstd_safe::compress_bound(raw_len);
    let mut out = vec![0u8; 1 + max_dst];
    out[0] = 1u8;
    let csize = zstd::bulk::compress_to_buffer(&shuffled, &mut out[1..], 1).ok()?;
    let total = 1 + csize;
    if total < (raw_len * 9) / 10 {
        out.truncate(total);
        Some(out)
    } else {
        None
    }
}

/// 解码投影/直接段单列字节：兼容 v3 裸 f32 列与 v4 自适应 Byte-Shuffle + Zstd 列。
pub(crate) fn decode_f32_column_bytes<'a>(
    seg_ver: u32,
    col_bytes: &'a [u8],
    expected_rows: usize,
    scratch: &'a mut Vec<u8>,
) -> Result<&'a [u8], String> {
    let raw_len = expected_rows
        .checked_mul(4)
        .ok_or_else(|| "投影列字节长度溢出".to_string())?;
    if seg_ver <= 3 || col_bytes.len() == raw_len {
        if col_bytes.len() != raw_len {
            return Err(format!(
                "投影列字节长度不匹配: got={}, expected={raw_len}",
                col_bytes.len()
            ));
        }
        return Ok(col_bytes);
    }
    if seg_ver != 4 || col_bytes.is_empty() || col_bytes.len() > raw_len {
        return Err(format!(
            "无效的压缩投影列头或长度: ver={seg_ver}, len={}, raw={raw_len}",
            col_bytes.len()
        ));
    }
    if col_bytes[0] != 1u8 {
        return Err(format!("未知的列压缩编码标记: {}", col_bytes[0]));
    }
    let mut shuffled = vec![0u8; raw_len];
    let dec_len = zstd::bulk::decompress_to_buffer(&col_bytes[1..], &mut shuffled)
        .map_err(|e| format!("解压投影列失败: {e}"))?;
    if dec_len != raw_len {
        return Err(format!(
            "解压后列字节数不匹配: got={dec_len}, expected={raw_len}"
        ));
    }
    scratch.resize(raw_len, 0u8);
    let n = expected_rows;
    let (b0, rest) = shuffled.split_at(n);
    let (b1, rest) = rest.split_at(n);
    let (b2, b3) = rest.split_at(n);
    for (i, dst) in scratch.chunks_exact_mut(4).enumerate() {
        dst[0] = b0[i];
        dst[1] = b1[i];
        dst[2] = b2[i];
        dst[3] = b3[i];
    }
    Ok(&scratch[..])
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
        let ver = u32_at(0);
        let fc = m.factor_names.len() as u64;
        let ro = u64_at(16);
        let csz = u64_at(24);
        let values = p.rows.checked_mul(4).ok_or("段行数溢出")?;
        if (ver != 3 && ver != 4)
            || u32_at(4) as u64 != fc
            || u64_at(8) != p.rows
            || ro != 32 + 16 * fc
            || csz == 0
        {
            return Err(format!("已提交段头损坏: {path:?}"));
        }
        let mut ix_bytes = vec![0u8; (fc as usize).saturating_mul(16)];
        f.read_exact(&mut ix_bytes).map_err(|e| e.to_string())?;
        let mut expected_off = ro + csz;
        for ix in ix_bytes.chunks_exact(16) {
            let col_off = u64::from_le_bytes(ix[..8].try_into().unwrap());
            let col_len = u64::from_le_bytes(ix[8..].try_into().unwrap());
            let len_ok = if ver == 3 {
                col_len == values
            } else {
                col_len > 0 && col_len <= values
            };
            if col_off != expected_off || !len_ok {
                return Err(format!("已提交段列索引损坏: {path:?}"));
            }
            expected_off = expected_off.checked_add(col_len).ok_or("段长度溢出")?;
        }
        if expected_off != len {
            return Err(format!("已提交段长度不匹配: {path:?}"));
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
    // 攒满的一段交给后台线程写，计算侧不等它；下一次封段或 finish 才收尾。
    // 有它，封段期间 append 不再停在写盘上。
    inflight: Option<InflightFlush>,
    // 上一段写完后交回来的空缓冲，下一次封段直接换给计算侧用，
    // 免得每个循环重新分配一整块内存。
    spare: Vec<Vec<f32>>,
}

fn fresh_columns(n: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| Vec::new()).collect()
}

/// 已经交给后台线程写、还没等它回话的一段。
/// 段文件和清单都由这个线程按顺序写（段号在封段时就定好），所以写完就是持久的；
/// 崩在写盘中途只会留下一个清单里没有的孤儿段，下次重算覆盖它。
struct InflightFlush {
    part_index: usize,
    rows: u64,
    keys: HashSet<(u32, u32)>,
    dates: HashSet<i64>,
    failed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: std::thread::JoinHandle<(Result<(), String>, Option<SegmentManifest>, Vec<Vec<f32>>)>,
}

const TRANSPOSE_TILE: usize = 64;

fn transpose_task_col_chunk(col_chunk: &mut [Vec<f32>], col_base: usize, accepted: &[&TaskResult]) {
    let n_rows = accepted.len();
    let cw = col_chunk.len();
    let mut base_lens = [0usize; TRANSPOSE_TILE];
    for (c, col) in col_chunk.iter_mut().enumerate() {
        let old_len = col.len();
        base_lens[c] = old_len;
        if col.capacity() < old_len + n_rows {
            col.reserve(n_rows);
        }
        unsafe {
            col.set_len(old_len + n_rows);
        }
    }
    let mut tile = [0.0f32; TRANSPOSE_TILE * TRANSPOSE_TILE];
    let mut row_base = 0usize;
    while row_base < n_rows {
        let rh = (n_rows - row_base).min(TRANSPOSE_TILE);
        for r in 0..rh {
            let src = &accepted[row_base + r].facs[col_base..col_base + cw];
            tile[r * TRANSPOSE_TILE..r * TRANSPOSE_TILE + cw].copy_from_slice(src);
        }
        for (c, col) in col_chunk.iter_mut().enumerate() {
            let dst_start = base_lens[c] + row_base;
            let dst = &mut col[dst_start..dst_start + rh];
            for r in 0..rh {
                dst[r] = tile[r * TRANSPOSE_TILE + c];
            }
        }
        row_base += rh;
    }
}

/// 64×64 L1 分块行转列：仅当当前已在显式受控 Rayon 池中时才并行分块，绝不触发 512 核全局池初始化。
fn transpose_task_rows_into_columns(columns: &mut [Vec<f32>], accepted: &[&TaskResult]) {
    let n_rows = accepted.len();
    let n_cols = columns.len();
    if n_rows == 0 || n_cols == 0 {
        return;
    }
    let use_par = rayon::current_thread_index().is_some()
        && rayon::current_num_threads() > 1
        && n_rows.saturating_mul(n_cols) >= 65_536;
    if use_par {
        use rayon::prelude::*;
        columns
            .par_chunks_mut(TRANSPOSE_TILE)
            .enumerate()
            .for_each(|(chunk_idx, col_chunk)| {
                transpose_task_col_chunk(col_chunk, chunk_idx * TRANSPOSE_TILE, accepted);
            });
    } else {
        for (chunk_idx, col_chunk) in columns.chunks_mut(TRANSPOSE_TILE).enumerate() {
            transpose_task_col_chunk(col_chunk, chunk_idx * TRANSPOSE_TILE, accepted);
        }
    }
}

fn transpose_flat_col_chunk(
    col_chunk: &mut [Vec<f32>],
    col_base: usize,
    row_major_vals: &[f32],
    n_cols: usize,
    accepted_indices: &[usize],
) {
    let n_rows = accepted_indices.len();
    let cw = col_chunk.len();
    let mut base_lens = [0usize; TRANSPOSE_TILE];
    for (c, col) in col_chunk.iter_mut().enumerate() {
        let old_len = col.len();
        base_lens[c] = old_len;
        if col.capacity() < old_len + n_rows {
            col.reserve(n_rows);
        }
        unsafe {
            col.set_len(old_len + n_rows);
        }
    }
    let mut tile = [0.0f32; TRANSPOSE_TILE * TRANSPOSE_TILE];
    let mut row_base = 0usize;
    while row_base < n_rows {
        let rh = (n_rows - row_base).min(TRANSPOSE_TILE);
        for r in 0..rh {
            let row_off = accepted_indices[row_base + r] * n_cols + col_base;
            let src = &row_major_vals[row_off..row_off + cw];
            tile[r * TRANSPOSE_TILE..r * TRANSPOSE_TILE + cw].copy_from_slice(src);
        }
        for (c, col) in col_chunk.iter_mut().enumerate() {
            let dst_start = base_lens[c] + row_base;
            let dst = &mut col[dst_start..dst_start + rh];
            for r in 0..rh {
                dst[r] = tile[r * TRANSPOSE_TILE + c];
            }
        }
        row_base += rh;
    }
}

/// 直接从 `/dev/shm` 行优先平坦切片 64×64 分块转置进列缓冲。
fn transpose_flat_rows_into_columns(
    columns: &mut [Vec<f32>],
    row_major_vals: &[f32],
    n_cols: usize,
    accepted_indices: &[usize],
) {
    let n_rows = accepted_indices.len();
    if n_rows == 0 || columns.is_empty() {
        return;
    }
    let use_par = rayon::current_thread_index().is_some()
        && rayon::current_num_threads() > 1
        && n_rows.saturating_mul(n_cols) >= 65_536;
    if use_par {
        use rayon::prelude::*;
        columns
            .par_chunks_mut(TRANSPOSE_TILE)
            .enumerate()
            .for_each(|(chunk_idx, col_chunk)| {
                transpose_flat_col_chunk(
                    col_chunk,
                    chunk_idx * TRANSPOSE_TILE,
                    row_major_vals,
                    n_cols,
                    accepted_indices,
                );
            });
    } else {
        for (chunk_idx, col_chunk) in columns.chunks_mut(TRANSPOSE_TILE).enumerate() {
            transpose_flat_col_chunk(
                col_chunk,
                chunk_idx * TRANSPOSE_TILE,
                row_major_vals,
                n_cols,
                accepted_indices,
            );
        }
    }
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
            inflight: None,
            spare: fresh_columns(names.len()),
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
        // 后台写盘已经报错：先把错误原文抛出来，别让计算再白攒一段。
        if self
            .inflight
            .as_ref()
            .is_some_and(|f| f.failed.load(std::sync::atomic::Ordering::Relaxed))
        {
            self.join_inflight()?;
            self.poisoned = true;
            return Err("直接段后台写盘线程报告失败".into());
        }
        self.validate(batch)?;
        let result = self.append_validated(batch);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn append_validated(&mut self, batch: &[TaskResult]) -> Result<(), String> {
        let mut offset = 0;
        while offset < batch.len() {
            // Reserve exact column capacities once per buffer (including freshly swapped spare).
            if self.columns[0].capacity() == 0 {
                for c in &mut self.columns {
                    c.try_reserve_exact(self.max_rows)
                        .map_err(|e| format!("分配直接段缓冲失败: {e}"))?;
                }
            }
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
                // 已经交给后台的那一段也算"收过"，避免同一行被写进两个段。
                let in_flight = self
                    .inflight
                    .as_ref()
                    .is_some_and(|f| f.keys.contains(&(d, c)));
                if !self.committed.contains(&(d, c)) && !in_flight && self.pending.insert((d, c)) {
                    self.rows.push((d, c));
                    accepted.push(r);
                }
            }
            transpose_task_rows_into_columns(&mut self.columns, &accepted);
            offset += take;
            if self.rows.len() == self.max_rows {
                self.seal_in_background()?;
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
                .chain(self.inflight.iter().flat_map(|f| f.keys.iter()))
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
    /// 直接从 `/dev/shm` 行优先连续 `f32` 切片转置写入列段缓冲，零中间 `TaskResult` 拷贝。
    pub fn append_date_flat(
        &mut self,
        date: i64,
        codes: &[&str],
        row_major_vals: &[f32],
        n_cols: usize,
    ) -> Result<(), String> {
        if self
            .inflight
            .as_ref()
            .is_some_and(|f| f.failed.load(std::sync::atomic::Ordering::Relaxed))
        {
            self.join_inflight()?;
            self.poisoned = true;
            return Err("直接段后台写盘线程报告失败".into());
        }
        if self.poisoned {
            return Err("直接段写入此前失败；请关闭并重新打开后续算".into());
        }
        if codes.is_empty() {
            return Err("整日直接写入要求非空且日期一致".into());
        }
        if n_cols != self.columns.len() {
            return Err(format!(
                "直接段结果列数错误: date={}, columns={}，期望 {}",
                date,
                n_cols,
                self.columns.len()
            ));
        }
        let expected_vals = codes
            .len()
            .checked_mul(n_cols)
            .ok_or_else(|| "整日直接写入元素数溢出".to_string())?;
        if row_major_vals.len() != expected_vals {
            return Err(format!(
                "直接段平坦矩阵长度错误: got={}, expected={}",
                row_major_vals.len(),
                expected_vals
            ));
        }
        let mut seen = HashSet::with_capacity(codes.len());
        let is_completed_date = self.manifest.completed_dates.contains(&date);
        let existing_date_id = self.date_ids.get(&date).copied();
        for &code in codes {
            if code.is_empty() {
                return Err(format!("直接段结果含空股票代码: date={date}"));
            }
            if !seen.insert(code) {
                return Err("批次中 (date, code) 重复".into());
            }
            if is_completed_date {
                let key = existing_date_id.zip(self.code_ids.get(code).copied());
                if !key.is_some_and(|(d, c)| self.committed.contains(&(d, c))) {
                    return Err(format!("日期 {date} 已整日提交，不能追加新股票 {code}"));
                }
            }
        }
        if let Some(d) = existing_date_id {
            if self
                .committed
                .iter()
                .chain(self.pending.iter())
                .chain(self.inflight.iter().flat_map(|f| f.keys.iter()))
                .any(|(di, ci)| {
                    *di == d && !seen.contains(self.manifest.codes[*ci as usize].as_str())
                })
            {
                return Err(format!("日期 {date} 重算结果缺少已有股票，拒绝标记完成"));
            }
        }
        let append_res = self.append_validated_flat(date, codes, row_major_vals, n_cols);
        if append_res.is_err() {
            self.poisoned = true;
            return append_res;
        }
        self.pending_dates.insert(date);
        if self.rows.is_empty() {
            if let Err(e) = self.finish() {
                self.poisoned = true;
                return Err(e);
            }
        }
        Ok(())
    }
    fn append_validated_flat(
        &mut self,
        date: i64,
        codes: &[&str],
        row_major_vals: &[f32],
        n_cols: usize,
    ) -> Result<(), String> {
        let d = match self.date_ids.get(&date) {
            Some(&d) => d,
            None => {
                let d = u32::try_from(self.manifest.dates.len()).map_err(|_| "日期字典溢出")?;
                self.manifest.dates.push(date);
                self.date_ids.insert(date, d);
                d
            }
        };
        let mut offset = 0usize;
        while offset < codes.len() {
            if self.columns[0].capacity() == 0 {
                for c in &mut self.columns {
                    c.try_reserve_exact(self.max_rows)
                        .map_err(|e| format!("分配直接段缓冲失败: {e}"))?;
                }
            }
            let take = (self.max_rows - self.rows.len()).min(codes.len() - offset);
            let mut accepted_indices = Vec::with_capacity(take);
            for (rel, &code) in codes[offset..offset + take].iter().enumerate() {
                let c = match self.code_ids.get(code) {
                    Some(&c) => c,
                    None => {
                        let c =
                            u32::try_from(self.manifest.codes.len()).map_err(|_| "股票字典溢出")?;
                        let owned = code.to_string();
                        self.manifest.codes.push(owned.clone());
                        self.code_ids.insert(owned, c);
                        c
                    }
                };
                let in_flight = self
                    .inflight
                    .as_ref()
                    .is_some_and(|f| f.keys.contains(&(d, c)));
                if !self.committed.contains(&(d, c)) && !in_flight && self.pending.insert((d, c)) {
                    self.rows.push((d, c));
                    accepted_indices.push(offset + rel);
                }
            }
            transpose_flat_rows_into_columns(
                &mut self.columns,
                row_major_vals,
                n_cols,
                &accepted_indices,
            );
            offset += take;
            if self.rows.len() == self.max_rows {
                self.seal_in_background()?;
            }
        }
        Ok(())
    }
    /// 把攒满的一段交给后台线程写盘，自己立刻返回；下一次封段或 finish 才等它。
    /// 这样封段的那几分钟里，计算侧继续往新缓冲里写，不再停在写盘上。
    fn seal_in_background(&mut self) -> Result<(), String> {
        // 上一段还没落地就再封，会把段文件顺序和清单顺序打乱，所以先收尾。
        self.join_inflight()?;
        let part_index = self.manifest.parts.len();
        let file = format!("segment_{part_index:08}.bin");
        let target = self.dir.join(&file);
        let manifest_path = self.dir.join(MANIFEST_NAME);
        let rows = std::mem::take(&mut self.rows);
        let n_columns = self.manifest.factor_names.len();
        let next_active = if self.spare.len() == n_columns {
            std::mem::take(&mut self.spare)
        } else {
            fresh_columns(n_columns)
        };
        let columns = std::mem::replace(&mut self.columns, next_active);
        let dates = std::mem::take(&mut self.pending_dates);
        let keys = std::mem::take(&mut self.pending);
        let n_rows = rows.len() as u64;
        // 先把"这一段写完之后清单长什么样"算好，交给后台线程自己落账：
        // 段文件和清单都由同一个线程按顺序写，所以封完就是持久的（和同步写时一致），
        // 崩在中途只会留下一个清单里没有的孤儿段，下次重算覆盖它。
        let mut next = self.manifest.clone();
        next.parts.push(SegmentPart {
            file: file.clone(),
            rows: n_rows,
        });
        next.completed_dates.extend(dates.iter().copied());
        next.completed_dates.sort_unstable();
        next.completed_dates.dedup();
        let failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let failed_in_thread = failed.clone();
        let handle = std::thread::spawn(move || {
            let tmp = target.with_extension("tmp");
            let mut result =
                write_segment(&tmp, &rows, &columns).and_then(|()| publish(&tmp, &target));
            let mut committed_manifest = None;
            if result.is_ok() {
                match serde_json::to_vec(&next) {
                    Ok(bytes) => {
                        if let Err(e) = atomic_write(&manifest_path, &bytes) {
                            result = Err(e);
                        } else {
                            committed_manifest = Some(next);
                        }
                    }
                    Err(e) => result = Err(e.to_string()),
                }
            }
            if result.is_err() {
                failed_in_thread.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // 缓冲交回去复用：只清长度，容量留着，下一段不必重新分配。
            let mut columns = columns;
            for c in &mut columns {
                c.clear();
            }
            (result, committed_manifest, columns)
        });
        self.inflight = Some(InflightFlush {
            part_index,
            rows: n_rows,
            keys,
            dates,
            failed,
            handle,
        });
        Ok(())
    }

    fn join_inflight(&mut self) -> Result<(), String> {
        let Some(inflight) = self.inflight.take() else {
            return Ok(());
        };
        let (result, committed_manifest, columns) = match inflight.handle.join() {
            Ok(done) => done,
            Err(_) => (
                Err("直接段后台写盘线程异常退出".to_string()),
                None,
                Vec::new(),
            ),
        };
        // 缓冲交回来，下一段接着用；长度不对（线程崩过）就等下次重新分配。
        if columns.len() == self.manifest.factor_names.len() {
            self.spare = columns;
        }
        match (result, committed_manifest) {
            (Ok(()), Some(next)) => {
                if next.parts.len() != inflight.part_index + 1 {
                    self.poisoned = true;
                    return Err("直接段后台写盘的段序号与清单不一致".into());
                }
                // 只接段列表和完成日期。日期表/股票表在封段之后还会继续增长
                // （封段时算的那份快照只是当下的子集），这里拿快照整体覆盖会把新条目丢掉。
                self.manifest.parts = next.parts;
                self.manifest.completed_dates = next.completed_dates;
                self.committed.extend(inflight.keys);
                Ok(())
            }
            (Ok(()), None) => {
                self.poisoned = true;
                Err("直接段后台写盘没有落账".into())
            }
            (Err(e), _) => {
                self.poisoned = true;
                Err(format!("直接段后台写盘失败: {e}"))
            }
        }
    }

    /// 同步封段：先收掉后台那一段，再把剩下的写掉并更新清单。
    fn flush_blocking(&mut self) -> Result<(), String> {
        self.join_inflight()?;
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
        if self.rows.is_empty() && self.pending_dates.is_empty() && self.inflight.is_none() {
            return Ok(());
        }
        let result = self.flush_blocking();
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

impl Drop for DirectSegmentWriter {
    /// 关掉写入器时收掉后台那一段，保证"已封存"在 close 之后就是持久的。
    /// 收尾要等那一段写完，最多等一次封段；错误不在这里抛，调用方走 finish/append 已经拿到了。
    fn drop(&mut self) {
        if self.inflight.is_some() {
            let _ = self.join_inflight();
        }
    }
}

fn write_segment(path: &Path, rows: &[(u32, u32)], columns: &[Vec<f32>]) -> Result<(), String> {
    for c in columns {
        if c.len() != rows.len() {
            return Err("直接段列长度不一致".into());
        }
    }
    let mut raw = Vec::with_capacity(rows.len() * 8);
    for (d, c) in rows {
        raw.extend_from_slice(&d.to_le_bytes());
        raw.extend_from_slice(&c.to_le_bytes());
    }
    let ro = zstd::encode_all(&raw[..], 1).map_err(|e| e.to_string())?;
    drop(raw);

    let enable_compress = rows.len() >= 64
        && std::env::var("RUST_PYFUNC_DISABLE_SEGMENT_COMPRESS")
            .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
    let seg_version: u32 = if enable_compress { 4 } else { 3 };

    let header_len = 32 + columns.len() as u64 * 16;
    let raw_col_len = rows.len() as u64 * 4;
    let mut header = vec![0u8; header_len as usize];
    header[0..4].copy_from_slice(&seg_version.to_le_bytes());
    header[4..8].copy_from_slice(&(columns.len() as u32).to_le_bytes());
    header[8..16].copy_from_slice(&(rows.len() as u64).to_le_bytes());
    header[16..24].copy_from_slice(&header_len.to_le_bytes());
    header[24..32].copy_from_slice(&(ro.len() as u64).to_le_bytes());

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let profile = StorageProfile::detect(path)?;
    #[cfg(target_os = "linux")]
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

    // 先跳过头部与压缩行序，顺序流式写出各列（v4 按 256 列分批并行自适应压缩，内存峰值仅几十 MB），
    // 最后精确回填 header 索引并截齐文件长度。
    let data_start = header_len + ro.len() as u64;
    file.seek(SeekFrom::Start(data_start))
        .map_err(|e| e.to_string())?;
    let mut out = std::io::BufWriter::with_capacity(8 * 1024 * 1024, file);
    let mut off = data_start;
    let mut unsynced = 0usize;

    if enable_compress {
        use rayon::prelude::*;
        let comp_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8.min(columns.len().max(1)))
            .build()
            .map_err(|e| format!("建列段压缩线程池失败: {e}"))?;
        const COMP_BATCH: usize = 256;
        for (batch_idx, col_batch) in columns.chunks(COMP_BATCH).enumerate() {
            let encoded: Vec<Option<Vec<u8>>> = comp_pool.install(|| {
                col_batch
                    .par_iter()
                    .map(|c| encode_f32_column_adaptive(c))
                    .collect()
            });
            for (local_i, (c, enc_opt)) in col_batch.iter().zip(encoded.iter()).enumerate() {
                let global_col = batch_idx * COMP_BATCH + local_i;
                let ix_pos = 32 + global_col * 16;
                let written_len: u64 = if let Some(comp_bytes) = enc_opt {
                    out.write_all(comp_bytes).map_err(|e| e.to_string())?;
                    unsynced += comp_bytes.len();
                    comp_bytes.len() as u64
                } else {
                    #[cfg(target_endian = "little")]
                    {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(c.as_ptr().cast::<u8>(), c.len() * 4)
                        };
                        out.write_all(bytes).map_err(|e| e.to_string())?;
                        unsynced += bytes.len();
                    }
                    #[cfg(target_endian = "big")]
                    for value in c {
                        out.write_all(&value.to_le_bytes())
                            .map_err(|e| e.to_string())?;
                        unsynced += 4;
                    }
                    raw_col_len
                };
                header[ix_pos..ix_pos + 8].copy_from_slice(&off.to_le_bytes());
                header[ix_pos + 8..ix_pos + 16].copy_from_slice(&written_len.to_le_bytes());
                off = off.checked_add(written_len).ok_or("直接段文件长度溢出")?;
                if unsynced >= 1024 * 1024 * 1024 {
                    out.flush().map_err(|e| e.to_string())?;
                    crate::storage_policy::sync_file(out.get_ref(), "直接段")?;
                    unsynced = 0;
                }
            }
        }
    } else {
        for (global_col, c) in columns.iter().enumerate() {
            let ix_pos = 32 + global_col * 16;
            header[ix_pos..ix_pos + 8].copy_from_slice(&off.to_le_bytes());
            header[ix_pos + 8..ix_pos + 16].copy_from_slice(&raw_col_len.to_le_bytes());
            off = off.checked_add(raw_col_len).ok_or("直接段文件长度溢出")?;
            #[cfg(target_endian = "little")]
            {
                let bytes =
                    unsafe { std::slice::from_raw_parts(c.as_ptr().cast::<u8>(), c.len() * 4) };
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
    }

    out.flush().map_err(|e| e.to_string())?;
    let mut file = out.into_inner().map_err(|e| e.to_string())?;
    file.set_len(off).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.write_all(&header)
        .and_then(|_| file.write_all(&ro))
        .map_err(|e| e.to_string())?;
    crate::storage_policy::sync_file(&file, "直接段")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用临时目录；按项目约定，因子测试文件一律放 /hdd。
    struct TempStore(PathBuf);
    impl TempStore {
        fn new(tag: &str) -> Self {
            let dir = Path::new("/hdd/user_home_unsafe/chenzongwei")
                .join(format!("codex_seg_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }
    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// 造 count 条互不重复的 (date, code)，从序号 start 开始。
    fn make_rows(start: usize, count: usize, cols: usize) -> Vec<TaskResult> {
        (start..start + count)
            .map(|i| TaskResult {
                date: 20220101 + (i / 100) as i64,
                code: format!("{:06}", i % 100),
                timestamp: 0,
                facs: (0..cols).map(|c| (i * cols + c) as f32).collect(),
            })
            .collect()
    }

    /// 模拟真实节奏：算一阵、攒满一段、再接着算。用 sleep 代替计算，
    /// 比例按实测来（同一个 32 GiB 段，算约 165 秒，写约 181 秒）。
    #[test]
    fn seal_overlap_with_compute() {
        let store = TempStore::new("overlap");
        let cols = 2000usize;
        let names: Vec<String> = (0..cols).map(|i| format!("f{i}")).collect();
        let buffer_bytes = 128 * 1024 * 1024;
        let max_rows = buffer_bytes / (cols * 4);
        let slice = max_rows / 8;
        let mut w = DirectSegmentWriter::open(store.0.to_str().unwrap(), &names, buffer_bytes).unwrap();
        let mut written = 0usize;
        let start = std::time::Instant::now();
        let mut cycle_rows = 0usize;
        let mut cycle_start = std::time::Instant::now();
        for cycle in 0..4 {
            for _ in 0..8 {
                let batch = make_rows(written, slice, cols);
                w.append_batch(&batch).unwrap();
                written += slice;
                cycle_rows += slice;
                std::thread::sleep(std::time::Duration::from_millis(150)); // 代替计算
            }
            println!(
                "第 {cycle} 段：{} 行，含计算共 {:.2} s",
                cycle_rows,
                cycle_start.elapsed().as_secs_f64()
            );
            cycle_rows = 0;
            cycle_start = std::time::Instant::now();
        }
        let t = std::time::Instant::now();
        w.finish().unwrap();
        println!("finish：{:.2} s，总墙钟 {:.2} s", t.elapsed().as_secs_f64(), start.elapsed().as_secs_f64());
        assert_eq!(w.record_count() as usize, written);
    }

    /// 封段这一下不能等写盘：append 返回时，段还没写完、清单还没动，
    /// 但下一批数据已经能往新缓冲里写。
    #[test]
    fn seal_returns_before_segment_is_written() {
        let store = TempStore::new("async");
        let cols = 2000usize;
        let names: Vec<String> = (0..cols).map(|i| format!("f{i}")).collect();
        let buffer_bytes = 128 * 1024 * 1024;
        let max_rows = buffer_bytes / (cols * 4);
        let mut w =
            DirectSegmentWriter::open(store.0.to_str().unwrap(), &names, buffer_bytes).unwrap();
        w.append_batch(&make_rows(0, max_rows, cols)).unwrap();
        // 刚好攒满：这一段已经在后台写了，计算侧立刻能干别的
        assert!(w.inflight.is_some(), "攒满后应交给后台写");
        assert!(w.rows.is_empty(), "缓冲应已换新");
        assert!(w.committed.is_empty(), "清单还没更新，不算完成");
        assert_eq!(w.completed_dates().len(), 0);
        // 后台还在写的时候继续收数据
        w.append_batch(&make_rows(max_rows, 1000, cols)).unwrap();
        assert_eq!(w.rows.len(), 1000);
        // 收尾会把后台那段接上，并把它写进清单
        w.finish().unwrap();
        assert!(w.inflight.is_none());
        assert_eq!(w.record_count() as usize, max_rows + 1000);
        assert_eq!(load_manifest(&store.0).unwrap().unwrap().parts.len(), 2);
    }

    /// 一整天的数据刚好攒满一段时，这一天要能在收尾后标记完成。
    #[test]
    fn exact_boundary_date_is_committed() {
        let store = TempStore::new("boundary");
        let cols = 100usize;
        let names: Vec<String> = (0..cols).map(|i| format!("f{i}")).collect();
        let buffer_bytes = 64 * 100 * 4; // 正好 64 行
        let mut w =
            DirectSegmentWriter::open(store.0.to_str().unwrap(), &names, buffer_bytes).unwrap();
        let day = 20220104i64;
        let batch = make_rows(0, 64, cols);
        let batch: Vec<TaskResult> = batch
            .into_iter()
            .map(|mut r| {
                r.date = day;
                r
            })
            .collect();
        w.append_date(day, &batch).unwrap();
        assert_eq!(w.completed_dates(), HashSet::from([day]));
        w.finish().unwrap();
        let m = load_manifest(&store.0).unwrap().unwrap();
        assert_eq!(m.parts.len(), 1);
        assert_eq!(m.completed_dates, vec![day]);
    }

    /// 后台写盘被强杀会留下清单里没有的孤儿段：重开要能容忍，重算时覆盖同名文件。
    #[test]
    fn orphan_segment_file_is_tolerated_and_overwritten() {
        let store = TempStore::new("orphan");
        let cols = 100usize;
        let names: Vec<String> = (0..cols).map(|i| format!("f{i}")).collect();
        let buffer_bytes = 64 * 100 * 4;
        {
            let mut w =
                DirectSegmentWriter::open(store.0.to_str().unwrap(), &names, buffer_bytes).unwrap();
            w.append_batch(&make_rows(0, 64, cols)).unwrap();
            w.finish().unwrap();
        }
        // 造一个"文件写完、清单还没落账"的孤儿段
        std::fs::copy(
            store.0.join("segment_00000000.bin"),
            store.0.join("segment_00000001.bin"),
        )
        .unwrap();
        assert_eq!(load_manifest(&store.0).unwrap().unwrap().parts.len(), 1);
        // 重开：不认孤儿段，行数只按清单算
        let mut w =
            DirectSegmentWriter::open(store.0.to_str().unwrap(), &names, buffer_bytes).unwrap();
        assert_eq!(w.record_count(), 64);
        // 再算一段，落在同名文件上，把它覆盖掉
        w.append_batch(&make_rows(64, 64, cols)).unwrap();
        w.finish().unwrap();
        let m = load_manifest(&store.0).unwrap().unwrap();
        assert_eq!(m.parts.len(), 2);
        assert_eq!(m.parts[1].rows, 64);
        assert_eq!(w.record_count(), 128);
    }

    /// 封段时 append 会不会被写盘挡住：逐批计时，最后核对行数与取值。
    #[test]
    fn seal_latency_report() {
        let store = TempStore::new("latency");
        let cols = 2000usize;
        let names: Vec<String> = (0..cols).map(|i| format!("f{i}")).collect();
        let buffer_bytes = 128 * 1024 * 1024;
        let max_rows = buffer_bytes / (cols * 4);
        let mut w = DirectSegmentWriter::open(store.0.to_str().unwrap(), &names, buffer_bytes).unwrap();
        let mut written = 0usize;
        for batch_idx in 0..4 {
            let batch = make_rows(written, max_rows, cols);
            let t = std::time::Instant::now();
            w.append_batch(&batch).unwrap();
            println!(
                "append 第 {batch_idx} 批 {} 行：{:.3} s",
                batch.len(),
                t.elapsed().as_secs_f64()
            );
            written += batch.len();
        }
        let t = std::time::Instant::now();
        w.finish().unwrap();
        println!("finish：{:.3} s", t.elapsed().as_secs_f64());
        assert_eq!(w.record_count() as usize, written);
        let parts = w.manifest.parts.len();
        assert!(parts >= 4, "预期每批都封一段，实际 {parts} 段");
        assert_eq!(load_manifest(&store.0).unwrap().unwrap().parts.len(), parts);
    }
}
