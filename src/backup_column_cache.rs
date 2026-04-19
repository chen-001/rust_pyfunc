use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

const HEADER_SIZE: usize = 64;

// v2 备份格式偏移（用于从备份文件读取原始数据）
const V2_CODE_LEN_OFFSET: usize = 8 + 8 + 8 + 4;
const V2_CODE_BYTES_OFFSET: usize = V2_CODE_LEN_OFFSET + 4;
const V2_FACTOR_BASE_OFFSET: usize = 8 + 8 + 8 + 4 + 4 + 32;
const V2_FACTOR_SIZE: usize = 8; // f64

// v3 备份格式偏移
const V3_CODE_OFFSET: usize = 8;
const V3_CODE_SIZE: usize = 16;
const V3_FACTOR_BASE_OFFSET: usize = 32;
const V3_FACTOR_SIZE: usize = 4; // f32

// 列缓存 v2 格式常量
const V2_MANIFEST_MAGIC: [u8; 8] = *b"RPCLBKV2";
const V2_MANIFEST_VERSION: u32 = 2;
const V2_META_ROW_SIZE: usize = 4; // date_id:u16 + code_id:u16
const V2_CODE_ROW_SIZE: usize = 16; // [u8; 16] null-terminated

// 列缓存 v1 格式常量（向后兼容）
const V1_MANIFEST_MAGIC: [u8; 8] = *b"RPCLBKV1";
const V1_MANIFEST_VERSION: u32 = 1;
const V1_META_ROW_SIZE: usize = 8 + 4;
const V1_CODE_ROW_SIZE: usize = 4 + 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheManifest {
    magic: [u8; 8],
    version: u32,
    source_size: u64,
    source_mtime_sec: u64,
    source_mtime_nsec: u32,
    record_count: u64,
    factor_count: u32,
    record_size: u32,
    block_cols: u32,
    code_count: u32,
    date_count: u32, // v2 新增：日期字典大小
}

#[derive(Debug, Clone, Copy)]
struct ParsedHeader {
    version: u32,
    record_count: usize,
    record_size: usize,
    factor_count: usize,
}

fn calculate_v2_record_size(factor_count: usize) -> usize {
    8 + 8 + 8 + 4 + 4 + 32 + factor_count * 8 + 4
}

fn calculate_v3_record_size(factor_count: usize) -> usize {
    8 + V3_CODE_SIZE + 8 + factor_count * V3_FACTOR_SIZE + 4
}

fn cache_dir_path(backup_file: &str) -> PathBuf {
    PathBuf::from(format!("{backup_file}.colblk_cache_v2"))
}

fn v1_cache_dir_path(backup_file: &str) -> PathBuf {
    PathBuf::from(format!("{backup_file}.colblk_cache_v1"))
}

fn manifest_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("manifest.bin")
}

fn meta_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("meta.bin")
}

fn codes_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("codes.bin")
}

fn dates_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("dict_dates.bin")
}

fn dict_codes_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("dict_codes.bin")
}

fn block_path(cache_dir: &Path, block_idx: usize) -> PathBuf {
    cache_dir.join(format!("blk_{block_idx:03}.bin"))
}

fn read_i64_le(buf: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
}

fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap())
}

fn read_f64_le(buf: &[u8], offset: usize) -> f64 {
    f64::from_bits(u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap()))
}

/// 从 v3 备份记录读取 null-terminated code
fn read_v3_code_from_record(record: &[u8]) -> String {
    let end = record[V3_CODE_OFFSET..V3_CODE_OFFSET + V3_CODE_SIZE]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(V3_CODE_SIZE);
    String::from_utf8_lossy(&record[V3_CODE_OFFSET..V3_CODE_OFFSET + end]).to_string()
}

/// 从 dict_codes.bin 的 16 字节条目读取 null-terminated code
fn read_code_from_dict16(entry: &[u8]) -> String {
    let end = entry.iter().position(|&b| b == 0).unwrap_or(V3_CODE_SIZE);
    String::from_utf8_lossy(&entry[..end]).to_string()
}

/// v4 chunk 索引条目（列缓存内部使用）
struct ChunkInfoV4 {
    compressed_size: usize,
    record_count: usize,
    data_offset: usize,
}

/// 扫描 v4 文件数据区构建 chunk 索引
fn build_chunk_index_v4(mmap: &[u8]) -> Result<Vec<ChunkInfoV4>, String> {
    let chunk_count = u32::from_le_bytes(mmap[28..32].try_into().unwrap()) as usize;
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut offset = HEADER_SIZE;

    for _ in 0..chunk_count {
        if offset + 8 > mmap.len() {
            return Err(format!("chunk 头部越界: offset={}, len={}", offset, mmap.len()));
        }
        let compressed_size = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
        let record_count = u32::from_le_bytes(mmap[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let data_offset = offset + 8;
        if data_offset + compressed_size > mmap.len() {
            return Err(format!(
                "chunk 数据越界: offset={}, compressed_size={}, len={}",
                data_offset, compressed_size, mmap.len()
            ));
        }
        chunks.push(ChunkInfoV4 {
            compressed_size,
            record_count,
            data_offset,
        });
        offset = data_offset + compressed_size;
    }
    Ok(chunks)
}

fn parse_header(mmap: &[u8]) -> PyResult<ParsedHeader> {
    if mmap.len() < HEADER_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件头部不足64字节",
        ));
    }
    if &mmap[0..8] != b"RPBACKUP" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "仅支持 RPBACKUP 格式构建列缓存",
        ));
    }

    let version = u32::from_le_bytes(mmap[8..12].try_into().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析版本号")
    })?);
    let record_count = u64::from_le_bytes(mmap[12..20].try_into().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析记录数")
    })?) as usize;
    let record_size = u32::from_le_bytes(mmap[20..24].try_into().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析记录大小")
    })?) as usize;
    let factor_count = u32::from_le_bytes(mmap[24..28].try_into().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析因子列数")
    })?) as usize;

    if version != 2 && version != 3 && version != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列缓存仅支持 RPBACKUP v2/v3/v4 格式，当前版本为 {version}"
        )));
    }

    let expected_record_size = if version == 3 || version == 4 {
        calculate_v3_record_size(factor_count)
    } else {
        calculate_v2_record_size(factor_count)
    };
    if record_size != expected_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 头部为 {record_size}, 计算为 {expected_record_size}"
        )));
    }

    // v4 不做简单的 expected_file_size 检查（压缩后大小不同）
    if version != 4 {
        let expected_file_size = HEADER_SIZE + record_count.saturating_mul(record_size);
        if mmap.len() < expected_file_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "备份文件被截断，无法构建列缓存",
            ));
        }
    }

    Ok(ParsedHeader {
        version,
        record_count,
        record_size,
        factor_count,
    })
}

fn source_fingerprint(path: &Path) -> PyResult<(u64, u64, u32)> {
    let meta = fs::metadata(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取源文件元数据失败: {e}"))
    })?;
    let source_size = meta.len();
    let modified = meta.modified().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取源文件修改时间失败: {e}"))
    })?;
    let duration = modified.duration_since(UNIX_EPOCH).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("源文件时间戳无效: {e}"))
    })?;
    Ok((source_size, duration.as_secs(), duration.subsec_nanos()))
}

fn load_manifest(cache_dir: &Path) -> PyResult<CacheManifest> {
    let bytes = fs::read(manifest_path(cache_dir)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 manifest.bin 失败: {e}"))
    })?;
    bincode::deserialize::<CacheManifest>(&bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("解析 manifest.bin 失败: {e}"))
    })
}

fn save_manifest(cache_dir: &Path, manifest: &CacheManifest) -> PyResult<()> {
    let bytes = bincode::serialize(manifest).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("序列化 manifest 失败: {e}"))
    })?;
    fs::write(manifest_path(cache_dir), bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入 manifest.bin 失败: {e}"))
    })
}

fn expected_block_count(factor_count: usize, block_cols: usize) -> usize {
    (factor_count + block_cols - 1) / block_cols
}

/// 校验 v2 缓存文件完整性
fn validate_v2_cache_files(cache_dir: &Path, manifest: &CacheManifest) -> bool {
    if manifest.magic != V2_MANIFEST_MAGIC {
        return false;
    }
    if manifest.version != V2_MANIFEST_VERSION {
        return false;
    }
    if manifest.block_cols == 0 {
        return false;
    }

    let record_count = manifest.record_count as usize;
    let factor_count = manifest.factor_count as usize;
    let block_cols = manifest.block_cols as usize;
    let blocks = expected_block_count(factor_count, block_cols);

    // meta.bin: record_count * 4 bytes
    if let Ok(m) = fs::metadata(meta_path(cache_dir)) {
        if m.len() != (record_count * V2_META_ROW_SIZE) as u64 {
            return false;
        }
    } else {
        return false;
    }

    // dict_codes.bin: code_count * 16 bytes
    if let Ok(m) = fs::metadata(dict_codes_path(cache_dir)) {
        if m.len() != (manifest.code_count as usize * V2_CODE_ROW_SIZE) as u64 {
            return false;
        }
    } else {
        return false;
    }

    // dict_dates.bin: date_count * 8 bytes
    if let Ok(m) = fs::metadata(dates_path(cache_dir)) {
        if m.len() != (manifest.date_count as usize * 8) as u64 {
            return false;
        }
    } else {
        return false;
    }

    // blk 文件: [u64原始大小][zstd压缩数据]（不校验精确大小，只存在即可）
    for blk_idx in 0..blocks {
        if !block_path(cache_dir, blk_idx).exists() {
            return false;
        }
    }

    true
}

/// 校验 v1 缓存文件完整性（向后兼容）
fn validate_v1_cache_files(cache_dir: &Path, manifest: &CacheManifest) -> bool {
    if manifest.magic != V1_MANIFEST_MAGIC {
        return false;
    }
    if manifest.version != V1_MANIFEST_VERSION {
        return false;
    }
    if manifest.block_cols == 0 {
        return false;
    }

    let record_count = manifest.record_count as usize;
    let factor_count = manifest.factor_count as usize;
    let block_cols = manifest.block_cols as usize;
    let blocks = expected_block_count(factor_count, block_cols);

    if let Ok(m) = fs::metadata(meta_path(cache_dir)) {
        if m.len() != (record_count * V1_META_ROW_SIZE) as u64 {
            return false;
        }
    } else {
        return false;
    }

    if let Ok(m) = fs::metadata(codes_path(cache_dir)) {
        if m.len() != (manifest.code_count as usize * V1_CODE_ROW_SIZE) as u64 {
            return false;
        }
    } else {
        return false;
    }

    for blk_idx in 0..blocks {
        let block_width = min(block_cols, factor_count - blk_idx * block_cols);
        if let Ok(m) = fs::metadata(block_path(cache_dir, blk_idx)) {
            if m.len() != (record_count * block_width * 8) as u64 {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

fn validate_manifest(
    cache_dir: &Path,
    backup_file: &Path,
    manifest: &CacheManifest,
) -> PyResult<bool> {
    let valid = if manifest.magic == V2_MANIFEST_MAGIC {
        validate_v2_cache_files(cache_dir, manifest)
    } else if manifest.magic == V1_MANIFEST_MAGIC {
        validate_v1_cache_files(cache_dir, manifest)
    } else {
        false
    };

    if !valid {
        return Ok(false);
    }

    let (source_size, sec, nsec) = source_fingerprint(backup_file)?;
    if source_size != manifest.source_size
        || sec != manifest.source_mtime_sec
        || nsec != manifest.source_mtime_nsec
    {
        return Ok(false);
    }

    Ok(true)
}

fn try_get_valid_manifest(cache_dir: &Path, backup_file: &Path) -> PyResult<Option<CacheManifest>> {
    if !cache_dir.exists() {
        return Ok(None);
    }
    let manifest = match load_manifest(cache_dir) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if validate_manifest(cache_dir, backup_file, &manifest)? {
        Ok(Some(manifest))
    } else {
        Ok(None)
    }
}

#[derive(Debug, Clone)]
struct BuildSummary {
    cache_dir: PathBuf,
    rebuilt: bool,
    manifest: CacheManifest,
}

/// 从 v2 备份构建 v2 列缓存（排序优化 + zstd level 9）
fn build_cache_from_v2(
    mmap: &[u8],
    header: &ParsedHeader,
    cache_dir: &Path,
    block_cols: usize,
) -> PyResult<CacheManifest> {
    let record_count = header.record_count;
    let factor_count = header.factor_count;
    let record_size = header.record_size;
    let block_count = expected_block_count(factor_count, block_cols);

    // 第一遍：提取唯一日期和代码，构建字典
    let mut date_to_id: HashMap<i64, u16> = HashMap::new();
    let mut code_to_id: HashMap<String, u16> = HashMap::new();
    let mut datebook: Vec<i64> = Vec::new();
    let mut codebook: Vec<String> = Vec::new();
    let mut row_date_ids: Vec<u16> = Vec::with_capacity(record_count);
    let mut row_code_ids: Vec<u16> = Vec::with_capacity(record_count);

    for row in 0..record_count {
        let offset = HEADER_SIZE + row * record_size;
        let record = &mmap[offset..offset + record_size];

        let date = read_i64_le(record, 0);
        let date_id = *date_to_id.entry(date).or_insert_with(|| {
            let id = datebook.len() as u16;
            datebook.push(date);
            id
        });
        row_date_ids.push(date_id);

        let code_len = min(read_u32_le(record, V2_CODE_LEN_OFFSET) as usize, 32);
        let code = String::from_utf8_lossy(&record[V2_CODE_BYTES_OFFSET..V2_CODE_BYTES_OFFSET + code_len]).to_string();
        let code_id = *code_to_id.entry(code.clone()).or_insert_with(|| {
            let id = codebook.len() as u16;
            codebook.push(code);
            id
        });
        row_code_ids.push(code_id);
    }

    if datebook.len() > 65535 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "日期数量 {} 超过 65535 限制", datebook.len()
        )));
    }
    if codebook.len() > 65535 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "代码数量 {} 超过 65535 限制", codebook.len()
        )));
    }

    // 构建 sort_index：按 (date_id, code_id) 排序
    let mut sort_index: Vec<usize> = (0..record_count).collect();
    sort_index.sort_unstable_by_key(|&row| (row_date_ids[row], row_code_ids[row]));

    // 写入 dict_dates.bin
    {
        let mut w = BufWriter::with_capacity(4 * 1024 * 1024, File::create(dates_path(cache_dir))?);
        for &d in &datebook {
            w.write_all(&d.to_le_bytes())?;
        }
        w.flush()?;
    }

    // 写入 dict_codes.bin (每个代码 16 字节 null-terminated)
    {
        let mut w = BufWriter::with_capacity(4 * 1024 * 1024, File::create(dict_codes_path(cache_dir))?);
        for code in &codebook {
            let mut buf = [0u8; V3_CODE_SIZE];
            let bytes = code.as_bytes();
            let copy_len = min(bytes.len(), 15);
            buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
            w.write_all(&buf)?;
        }
        w.flush()?;
    }

    // 写入 meta.bin: [date_id:u16, code_id:u16] 每行 4 字节（按 sort_index 顺序）
    // 同时收集每块的 f32 数据
    let mut meta_writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(meta_path(cache_dir))?);
    let mut block_f32_data: Vec<Vec<f32>> = vec![Vec::new(); block_count];

    for &row in &sort_index {
        let date_id = row_date_ids[row];
        let code_id = row_code_ids[row];

        meta_writer.write_all(&date_id.to_le_bytes())?;
        meta_writer.write_all(&code_id.to_le_bytes())?;

        let offset = HEADER_SIZE + row * record_size;
        let record = &mmap[offset..offset + record_size];

        // 读取 f64 因子并转换为 f32
        for blk_idx in 0..block_count {
            let start_col = blk_idx * block_cols;
            let end_col = min(start_col + block_cols, factor_count);
            for col in start_col..end_col {
                let byte_offset = V2_FACTOR_BASE_OFFSET + col * V2_FACTOR_SIZE;
                let f = read_f64_le(record, byte_offset);
                block_f32_data[blk_idx].push(f as f32);
            }
        }
    }
    meta_writer.flush()?;

    // 写入块文件：[u64 原始大小][zstd 压缩数据]，压缩级别 9
    for blk_idx in 0..block_count {
        let f32_slice = &block_f32_data[blk_idx];
        let raw_bytes = unsafe {
            std::slice::from_raw_parts(
                f32_slice.as_ptr() as *const u8,
                f32_slice.len() * 4,
            )
        };

        let compressed = zstd::encode_all(raw_bytes, 9).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("zstd 压缩失败: {e}"))
        })?;

        let mut w = BufWriter::with_capacity(16 * 1024 * 1024, File::create(block_path(cache_dir, blk_idx))?);
        w.write_all(&(raw_bytes.len() as u64).to_le_bytes())?;
        w.write_all(&compressed)?;
        w.flush()?;
    }

    Ok(CacheManifest {
        magic: V2_MANIFEST_MAGIC,
        version: V2_MANIFEST_VERSION,
        source_size: 0, // 稍后设置
        source_mtime_sec: 0,
        source_mtime_nsec: 0,
        record_count: record_count as u64,
        factor_count: factor_count as u32,
        record_size: record_size as u32,
        block_cols: block_cols as u32,
        code_count: codebook.len() as u32,
        date_count: datebook.len() as u32,
    })
}

/// 从 v3 备份构建 v2 列缓存（支持排序优化 + zstd level 9）
fn build_cache_from_v3(
    mmap: &[u8],
    header: &ParsedHeader,
    cache_dir: &Path,
    block_cols: usize,
) -> PyResult<CacheManifest> {
    let record_count = header.record_count;
    let factor_count = header.factor_count;
    let record_size = header.record_size;

    build_cache_from_v3_buffer(
        mmap,
        HEADER_SIZE,
        record_count,
        record_size,
        factor_count,
        cache_dir,
        block_cols,
    )
}

/// 从连续的 v3 格式记录缓冲区构建列缓存（v3/v4 共用）
fn build_cache_from_v3_buffer(
    data: &[u8],
    data_start: usize,
    record_count: usize,
    record_size: usize,
    factor_count: usize,
    cache_dir: &Path,
    block_cols: usize,
) -> PyResult<CacheManifest> {
    let block_count = expected_block_count(factor_count, block_cols);

    // 第一遍：提取唯一日期和代码
    let mut date_to_id: HashMap<i64, u16> = HashMap::new();
    let mut code_to_id: HashMap<String, u16> = HashMap::new();
    let mut datebook: Vec<i64> = Vec::new();
    let mut codebook: Vec<String> = Vec::new();
    let mut row_date_ids: Vec<u16> = Vec::with_capacity(record_count);
    let mut row_code_ids: Vec<u16> = Vec::with_capacity(record_count);

    for row in 0..record_count {
        let offset = data_start + row * record_size;
        let record = &data[offset..offset + record_size];

        let date = read_i64_le(record, 0);
        let date_id = *date_to_id.entry(date).or_insert_with(|| {
            let id = datebook.len() as u16;
            datebook.push(date);
            id
        });
        row_date_ids.push(date_id);

        let code = read_v3_code_from_record(record);
        let code_id = *code_to_id.entry(code.clone()).or_insert_with(|| {
            let id = codebook.len() as u16;
            codebook.push(code);
            id
        });
        row_code_ids.push(code_id);
    }

    if datebook.len() > 65535 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "日期数量 {} 超过 65535 限制", datebook.len()
        )));
    }
    if codebook.len() > 65535 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "代码数量 {} 超过 65535 限制", codebook.len()
        )));
    }

    // 构建 sort_index：按 (date_id, code_id) 排序
    let mut sort_index: Vec<usize> = (0..record_count).collect();
    sort_index.sort_unstable_by_key(|&row| (row_date_ids[row], row_code_ids[row]));

    // 写入 dict_dates.bin
    {
        let mut w = BufWriter::with_capacity(4 * 1024 * 1024, File::create(dates_path(cache_dir))?);
        for &d in &datebook {
            w.write_all(&d.to_le_bytes())?;
        }
        w.flush()?;
    }

    // 写入 dict_codes.bin
    {
        let mut w = BufWriter::with_capacity(4 * 1024 * 1024, File::create(dict_codes_path(cache_dir))?);
        for code in &codebook {
            let mut buf = [0u8; V3_CODE_SIZE];
            let bytes = code.as_bytes();
            let copy_len = min(bytes.len(), 15);
            buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
            w.write_all(&buf)?;
        }
        w.flush()?;
    }

    // 写入 meta.bin 并收集 f32 数据（按 sort_index 顺序）
    let mut meta_writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(meta_path(cache_dir))?);
    let mut block_f32_data: Vec<Vec<f32>> = vec![Vec::new(); block_count];

    for &row in &sort_index {
        let date_id = row_date_ids[row];
        let code_id = row_code_ids[row];

        meta_writer.write_all(&date_id.to_le_bytes())?;
        meta_writer.write_all(&code_id.to_le_bytes())?;

        let offset = data_start + row * record_size;
        let record = &data[offset..offset + record_size];

        for blk_idx in 0..block_count {
            let start_col = blk_idx * block_cols;
            let end_col = min(start_col + block_cols, factor_count);
            for col in start_col..end_col {
                let byte_offset = V3_FACTOR_BASE_OFFSET + col * V3_FACTOR_SIZE;
                let bits = u32::from_le_bytes(record[byte_offset..byte_offset + 4].try_into().unwrap());
                block_f32_data[blk_idx].push(f32::from_bits(bits));
            }
        }
    }
    meta_writer.flush()?;

    // 写入块文件：[u64 原始大小][zstd 压缩数据]，压缩级别 9
    for blk_idx in 0..block_count {
        let f32_slice = &block_f32_data[blk_idx];
        let raw_bytes = unsafe {
            std::slice::from_raw_parts(
                f32_slice.as_ptr() as *const u8,
                f32_slice.len() * 4,
            )
        };

        let compressed = zstd::encode_all(raw_bytes, 9).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("zstd 压缩失败: {e}"))
        })?;

        let mut w = BufWriter::with_capacity(16 * 1024 * 1024, File::create(block_path(cache_dir, blk_idx))?);
        w.write_all(&(raw_bytes.len() as u64).to_le_bytes())?;
        w.write_all(&compressed)?;
        w.flush()?;
    }

    Ok(CacheManifest {
        magic: V2_MANIFEST_MAGIC,
        version: V2_MANIFEST_VERSION,
        source_size: 0,
        source_mtime_sec: 0,
        source_mtime_nsec: 0,
        record_count: record_count as u64,
        factor_count: factor_count as u32,
        record_size: record_size as u32,
        block_cols: block_cols as u32,
        code_count: codebook.len() as u32,
        date_count: datebook.len() as u32,
    })
}

/// 从 v4 分块压缩备份构建 v2 列缓存
fn build_cache_from_v4(
    mmap: &[u8],
    header: &ParsedHeader,
    cache_dir: &Path,
    block_cols: usize,
) -> PyResult<CacheManifest> {
    let factor_count = header.factor_count;
    let record_size = header.record_size;
    let record_count = header.record_count;

    // 解压所有 chunk 到连续缓冲区
    let chunks = build_chunk_index_v4(mmap).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
    })?;

    let mut decompressed = Vec::with_capacity(record_count * record_size);
    for chunk in &chunks {
        let compressed_data = &mmap[chunk.data_offset..chunk.data_offset + chunk.compressed_size];
        let decompressed_chunk = zstd::decode_all(compressed_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("zstd 解压失败: {e}"))
        })?;
        decompressed.extend_from_slice(&decompressed_chunk);
    }

    if decompressed.len() != record_count * record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "解压后大小不匹配: 期望 {}, 实际 {}",
            record_count * record_size,
            decompressed.len()
        )));
    }

    // 复用 v3 缓冲区构建逻辑
    build_cache_from_v3_buffer(
        &decompressed,
        0, // data_start=0，因为 decompressed 是纯数据（没有文件头）
        record_count,
        record_size,
        factor_count,
        cache_dir,
        block_cols,
    )
}

fn build_cache_internal(
    backup_file: &str,
    block_cols: usize,
    force_rebuild: bool,
) -> PyResult<BuildSummary> {
    if block_cols == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "block_cols 必须大于 0",
        ));
    }

    let backup_path = Path::new(backup_file);
    if !backup_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在",
        ));
    }

    let cache_dir = cache_dir_path(backup_file);
    if !force_rebuild {
        if let Some(manifest) = try_get_valid_manifest(&cache_dir, backup_path)? {
            return Ok(BuildSummary {
                cache_dir,
                rebuilt: false,
                manifest,
            });
        }
    }

    if cache_dir.exists() {
        fs::remove_dir_all(&cache_dir).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("删除旧缓存目录失败: {e}"))
        })?;
    }
    fs::create_dir_all(&cache_dir).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("创建缓存目录失败: {e}"))
    })?;

    let src_file = File::open(backup_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法打开备份文件: {e}"))
    })?;
    let mmap = unsafe {
        Mmap::map(&src_file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("无法映射备份文件: {e}"))
        })?
    };

    let header = parse_header(&mmap)?;

    let mut manifest = if header.version == 4 {
        build_cache_from_v4(&mmap, &header, &cache_dir, block_cols)?
    } else if header.version == 3 {
        build_cache_from_v3(&mmap, &header, &cache_dir, block_cols)?
    } else {
        build_cache_from_v2(&mmap, &header, &cache_dir, block_cols)?
    };

    let (source_size, source_mtime_sec, source_mtime_nsec) = source_fingerprint(backup_path)?;
    manifest.source_size = source_size;
    manifest.source_mtime_sec = source_mtime_sec;
    manifest.source_mtime_nsec = source_mtime_nsec;
    save_manifest(&cache_dir, &manifest)?;

    Ok(BuildSummary {
        cache_dir,
        rebuilt: true,
        manifest,
    })
}

fn ensure_cache(backup_file: &str, build_if_missing: bool) -> PyResult<(PathBuf, CacheManifest)> {
    let backup_path = Path::new(backup_file);
    if !backup_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "备份文件不存在",
        ));
    }

    // 先尝试 v2 缓存
    let cache_dir = cache_dir_path(backup_file);
    if let Some(manifest) = try_get_valid_manifest(&cache_dir, backup_path)? {
        return Ok((cache_dir, manifest));
    }

    // 再尝试 v1 缓存（向后兼容）
    let v1_dir = v1_cache_dir_path(backup_file);
    if let Some(manifest) = try_get_valid_manifest(&v1_dir, backup_path)? {
        return Ok((v1_dir, manifest));
    }

    if !build_if_missing {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "缓存不存在或已失效，请先构建缓存",
        ));
    }

    let summary = build_cache_internal(backup_file, 32, false)?;
    Ok((summary.cache_dir, summary.manifest))
}

/// 从 v2 格式块文件中解压并读取指定列
fn read_factor_from_v2_block(
    blk_mmap: &[u8],
    record_count: usize,
    block_width: usize,
    local_col: usize,
) -> Vec<f64> {
    let raw_size = u64::from_le_bytes(blk_mmap[0..8].try_into().unwrap()) as usize;
    let compressed = &blk_mmap[8..];
    let decompressed = zstd::decode_all(compressed).expect("zstd 解压失败");

    let f32_slice = unsafe {
        std::slice::from_raw_parts(decompressed.as_ptr() as *const f32, raw_size / 4)
    };

    let mut factors = Vec::with_capacity(record_count);
    for row in 0..record_count {
        let idx = row * block_width + local_col;
        factors.push(f32_slice[idx] as f64);
    }
    factors
}

#[pyfunction]
#[pyo3(signature = (backup_file, block_cols=32, force_rebuild=false))]
pub fn build_backup_column_block_cache_single_thread(
    backup_file: String,
    block_cols: usize,
    force_rebuild: bool,
) -> PyResult<PyObject> {
    let summary = build_cache_internal(&backup_file, block_cols, force_rebuild)?;
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("cache_dir", summary.cache_dir.to_string_lossy().to_string())?;
        dict.set_item("record_count", summary.manifest.record_count)?;
        dict.set_item("factor_count", summary.manifest.factor_count)?;
        dict.set_item("block_cols", summary.manifest.block_cols)?;
        dict.set_item("code_count", summary.manifest.code_count)?;
        dict.set_item("date_count", summary.manifest.date_count)?;
        dict.set_item("rebuilt", summary.rebuilt)?;
        dict.set_item("cache_version", summary.manifest.version)?;
        Ok(dict.into())
    })
}

#[pyfunction]
#[pyo3(signature = (backup_file, column_index, build_if_missing=true))]
pub fn query_backup_single_column_compact_cached(
    backup_file: String,
    column_index: usize,
    build_if_missing: bool,
) -> PyResult<PyObject> {
    let (cache_dir, manifest) = ensure_cache(&backup_file, build_if_missing)?;
    let record_count = manifest.record_count as usize;
    let factor_count = manifest.factor_count as usize;
    let block_cols = manifest.block_cols as usize;

    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列索引 {column_index} 超出范围，因子列数为 {factor_count}"
        )));
    }

    let block_idx = column_index / block_cols;
    let local_col = column_index % block_cols;
    let block_width = min(block_cols, factor_count - block_idx * block_cols);

    // 读取 meta.bin
    let meta_bytes = fs::read(meta_path(&cache_dir)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 meta.bin 失败: {e}"))
    })?;

    if manifest.magic == V2_MANIFEST_MAGIC {
        // v2 缓存格式
        let blk_bytes = fs::read(block_path(&cache_dir, block_idx)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取块文件失败: {e}"))
        })?;

        let factors = read_factor_from_v2_block(&blk_bytes, record_count, block_width, local_col);

        let mut dates = Vec::with_capacity(record_count);
        let mut code_ids = Vec::with_capacity(record_count);
        for row in 0..record_count {
            let meta_offset = row * V2_META_ROW_SIZE;
            let date_id = read_u16_le(&meta_bytes, meta_offset);
            let code_id = read_u16_le(&meta_bytes, meta_offset + 2);
            dates.push(date_id as u32);
            code_ids.push(code_id);
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("date_id", PyArray1::from_vec(py, dates))?;
            dict.set_item("code_id", PyArray1::from_vec(py, code_ids))?;
            dict.set_item("factor", PyArray1::from_vec(py, factors))?;
            Ok(dict.into())
        })
    } else {
        // v1 缓存格式（向后兼容）
        let meta_mmap = unsafe {
            Mmap::map(&File::open(meta_path(&cache_dir))?).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("映射 meta.bin 失败: {e}"))
            })?
        };
        let blk_mmap = unsafe {
            Mmap::map(&File::open(block_path(&cache_dir, block_idx))?).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("映射块文件失败: {e}"))
            })?
        };

        let mut dates = Vec::with_capacity(record_count);
        let mut code_ids = Vec::with_capacity(record_count);
        let mut factors = Vec::with_capacity(record_count);
        let block_row_bytes = block_width * 8;

        for row in 0..record_count {
            let meta_offset = row * V1_META_ROW_SIZE;
            let date = read_i64_le(&meta_mmap, meta_offset);
            let code_id = read_u32_le(&meta_mmap, meta_offset + 8);
            let factor_offset = row * block_row_bytes + local_col * 8;
            let factor = read_f64_le(&blk_mmap, factor_offset);
            dates.push(date);
            code_ids.push(code_id);
            factors.push(factor);
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("date", PyArray1::from_vec(py, dates))?;
            dict.set_item("code_id", PyArray1::from_vec(py, code_ids))?;
            dict.set_item("factor", PyArray1::from_vec(py, factors))?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
#[pyo3(signature = (cache_dir, column_index))]
pub fn query_backup_single_column_compact_cached_from_cache_dir(
    cache_dir: String,
    column_index: usize,
) -> PyResult<PyObject> {
    let cache_dir_path = PathBuf::from(cache_dir);
    if !cache_dir_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "缓存目录不存在",
        ));
    }
    if !cache_dir_path.is_dir() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cache_dir 必须是目录路径",
        ));
    }

    let manifest = load_manifest(&cache_dir_path)?;
    let valid = if manifest.magic == V2_MANIFEST_MAGIC {
        validate_v2_cache_files(&cache_dir_path, &manifest)
    } else {
        validate_v1_cache_files(&cache_dir_path, &manifest)
    };
    if !valid {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "缓存目录结构或文件大小无效，缓存可能损坏",
        ));
    }

    let record_count = manifest.record_count as usize;
    let factor_count = manifest.factor_count as usize;
    let block_cols = manifest.block_cols as usize;

    if column_index >= factor_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "列索引 {column_index} 超出范围，因子列数为 {factor_count}"
        )));
    }

    let block_idx = column_index / block_cols;
    let local_col = column_index % block_cols;
    let block_width = min(block_cols, factor_count - block_idx * block_cols);

    let meta_bytes = fs::read(meta_path(&cache_dir_path)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 meta.bin 失败: {e}"))
    })?;

    if manifest.magic == V2_MANIFEST_MAGIC {
        // v2 缓存格式
        let blk_bytes = fs::read(block_path(&cache_dir_path, block_idx)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取块文件失败: {e}"))
        })?;

        let factors = read_factor_from_v2_block(&blk_bytes, record_count, block_width, local_col);

        let mut dates = Vec::with_capacity(record_count);
        let mut code_ids = Vec::with_capacity(record_count);
        for row in 0..record_count {
            let meta_offset = row * V2_META_ROW_SIZE;
            let date_id = read_u16_le(&meta_bytes, meta_offset);
            let code_id = read_u16_le(&meta_bytes, meta_offset + 2);
            dates.push(date_id as u32);
            code_ids.push(code_id);
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("date_id", PyArray1::from_vec(py, dates))?;
            dict.set_item("code_id", PyArray1::from_vec(py, code_ids))?;
            dict.set_item("factor", PyArray1::from_vec(py, factors))?;
            Ok(dict.into())
        })
    } else {
        // v1 缓存格式（向后兼容）
        let blk_mmap = unsafe {
            Mmap::map(&File::open(block_path(&cache_dir_path, block_idx))?).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("映射块文件失败: {e}"))
            })?
        };
        let block_row_bytes = block_width * 8;

        let mut dates = Vec::with_capacity(record_count);
        let mut code_ids = Vec::with_capacity(record_count);
        let mut factors = Vec::with_capacity(record_count);

        for row in 0..record_count {
            let meta_offset = row * V1_META_ROW_SIZE;
            let date = read_i64_le(&meta_bytes, meta_offset);
            let code_id = read_u32_le(&meta_bytes, meta_offset + 8);
            let factor_offset = row * block_row_bytes + local_col * 8;
            let factor = read_f64_le(&blk_mmap, factor_offset);
            dates.push(date);
            code_ids.push(code_id);
            factors.push(factor);
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("date", PyArray1::from_vec(py, dates))?;
            dict.set_item("code_id", PyArray1::from_vec(py, code_ids))?;
            dict.set_item("factor", PyArray1::from_vec(py, factors))?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
#[pyo3(signature = (backup_file))]
pub fn query_backup_codebook_cached(backup_file: String) -> PyResult<PyObject> {
    let (cache_dir, manifest) = ensure_cache(&backup_file, true)?;

    if manifest.magic == V2_MANIFEST_MAGIC {
        // v2 缓存：从 dict_codes.bin 读取
        let code_bytes = fs::read(dict_codes_path(&cache_dir)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 dict_codes.bin 失败: {e}"))
        })?;

        let expected_size = manifest.code_count as usize * V2_CODE_ROW_SIZE;
        if code_bytes.len() != expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dict_codes.bin 大小不符合 manifest 记录，缓存可能损坏",
            ));
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for idx in 0..manifest.code_count as usize {
                let offset = idx * V2_CODE_ROW_SIZE;
                let code = read_code_from_dict16(&code_bytes[offset..offset + V2_CODE_ROW_SIZE]);
                dict.set_item(idx as u32, code)?;
            }
            Ok(dict.into())
        })
    } else {
        // v1 缓存：从 codes.bin 读取
        let code_bytes = fs::read(codes_path(&cache_dir)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 codes.bin 失败: {e}"))
        })?;

        let expected_size = manifest.code_count as usize * V1_CODE_ROW_SIZE;
        if code_bytes.len() != expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "codes.bin 大小不符合 manifest 记录，缓存可能损坏",
            ));
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for idx in 0..manifest.code_count as usize {
                let offset = idx * V1_CODE_ROW_SIZE;
                let code_len = min(read_u32_le(&code_bytes, offset) as usize, 32);
                let bytes = &code_bytes[offset + 4..offset + 4 + 32];
                let code = String::from_utf8_lossy(&bytes[..code_len]).into_owned();
                dict.set_item(idx as u32, code)?;
            }
            Ok(dict.into())
        })
    }
}

#[pyfunction]
#[pyo3(signature = (cache_dir))]
pub fn query_backup_codebook_cached_from_cache_dir(cache_dir: String) -> PyResult<PyObject> {
    let cache_dir_path = PathBuf::from(cache_dir);
    if !cache_dir_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "缓存目录不存在",
        ));
    }
    if !cache_dir_path.is_dir() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cache_dir 必须是目录路径",
        ));
    }

    let manifest = load_manifest(&cache_dir_path)?;
    let valid = if manifest.magic == V2_MANIFEST_MAGIC {
        validate_v2_cache_files(&cache_dir_path, &manifest)
    } else {
        validate_v1_cache_files(&cache_dir_path, &manifest)
    };
    if !valid {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "缓存目录结构或文件大小无效，缓存可能损坏",
        ));
    }

    if manifest.magic == V2_MANIFEST_MAGIC {
        let code_bytes = fs::read(dict_codes_path(&cache_dir_path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 dict_codes.bin 失败: {e}"))
        })?;

        let expected_size = manifest.code_count as usize * V2_CODE_ROW_SIZE;
        if code_bytes.len() != expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dict_codes.bin 大小不符合 manifest 记录，缓存可能损坏",
            ));
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for idx in 0..manifest.code_count as usize {
                let offset = idx * V2_CODE_ROW_SIZE;
                let code = read_code_from_dict16(&code_bytes[offset..offset + V2_CODE_ROW_SIZE]);
                dict.set_item(idx as u32, code)?;
            }
            Ok(dict.into())
        })
    } else {
        let code_bytes = fs::read(codes_path(&cache_dir_path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 codes.bin 失败: {e}"))
        })?;

        let expected_size = manifest.code_count as usize * V1_CODE_ROW_SIZE;
        if code_bytes.len() != expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "codes.bin 大小不符合 manifest 记录，缓存可能损坏",
            ));
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for idx in 0..manifest.code_count as usize {
                let offset = idx * V1_CODE_ROW_SIZE;
                let code_len = min(read_u32_le(&code_bytes, offset) as usize, 32);
                let bytes = &code_bytes[offset + 4..offset + 4 + 32];
                let code = String::from_utf8_lossy(&bytes[..code_len]).into_owned();
                dict.set_item(idx as u32, code)?;
            }
            Ok(dict.into())
        })
    }
}
