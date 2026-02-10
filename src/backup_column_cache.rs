use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

const HEADER_SIZE: usize = 64;
const DATE_OFFSET: usize = 0;
const CODE_LEN_OFFSET: usize = 8 + 8 + 8 + 4;
const CODE_BYTES_OFFSET: usize = CODE_LEN_OFFSET + 4;
const FACTOR_BASE_OFFSET: usize = 8 + 8 + 8 + 4 + 4 + 32;
const META_ROW_SIZE: usize = 8 + 4;
const CODE_ROW_SIZE: usize = 4 + 32;
const MANIFEST_VERSION: u32 = 1;
const MANIFEST_MAGIC: [u8; 8] = *b"RPCLBKV1";

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
}

#[derive(Debug, Clone, Copy)]
struct ParsedHeader {
    record_count: usize,
    record_size: usize,
    factor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CodeKey {
    code_len: u8,
    code_bytes: [u8; 32],
}

#[derive(Debug, Clone)]
struct BuildSummary {
    cache_dir: PathBuf,
    rebuilt: bool,
    manifest: CacheManifest,
}

fn calculate_record_size(factor_count: usize) -> usize {
    8 + 8 + 8 + 4 + 4 + 32 + factor_count * 8 + 4
}

fn cache_dir_path(backup_file: &str) -> PathBuf {
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

fn block_path(cache_dir: &Path, block_idx: usize) -> PathBuf {
    cache_dir.join(format!("blk_{block_idx:03}.bin"))
}

fn parse_header(mmap: &[u8]) -> PyResult<ParsedHeader> {
    if mmap.len() < HEADER_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件头部不足64字节",
        ));
    }
    if &mmap[0..8] != b"RPBACKUP" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "仅支持 RPBACKUP v2 格式构建列缓存",
        ));
    }

    let version = u32::from_le_bytes(
        mmap[8..12]
            .try_into()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析版本号"))?,
    );
    let record_count = u64::from_le_bytes(
        mmap[12..20]
            .try_into()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析记录数"))?,
    ) as usize;
    let record_size = u32::from_le_bytes(
        mmap[20..24]
            .try_into()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析记录大小"))?,
    ) as usize;
    let factor_count = u32::from_le_bytes(
        mmap[24..28]
            .try_into()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("无法解析因子列数"))?,
    ) as usize;

    if version != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "列缓存仅支持 RPBACKUP v2 格式",
        ));
    }

    let expected_record_size = calculate_record_size(factor_count);
    if record_size != expected_record_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "记录大小不匹配: 头部为 {record_size}, 计算为 {expected_record_size}"
        )));
    }

    let expected_file_size = HEADER_SIZE + record_count.saturating_mul(record_size);
    if mmap.len() < expected_file_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "备份文件被截断，无法构建列缓存",
        ));
    }

    Ok(ParsedHeader {
        record_count,
        record_size,
        factor_count,
    })
}

fn read_i64_le(buf: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
}

fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn read_f64_le(buf: &[u8], offset: usize) -> f64 {
    f64::from_bits(u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap()))
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

fn validate_cache_files(cache_dir: &Path, manifest: &CacheManifest) -> bool {
    if manifest.magic != MANIFEST_MAGIC {
        return false;
    }
    if manifest.version != MANIFEST_VERSION {
        return false;
    }
    if manifest.block_cols == 0 {
        return false;
    }

    let record_count = manifest.record_count as usize;
    let factor_count = manifest.factor_count as usize;
    let block_cols = manifest.block_cols as usize;
    let blocks = expected_block_count(factor_count, block_cols);

    let meta_meta = match fs::metadata(meta_path(cache_dir)) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if meta_meta.len() != (record_count * META_ROW_SIZE) as u64 {
        return false;
    }

    let code_meta = match fs::metadata(codes_path(cache_dir)) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if code_meta.len() != (manifest.code_count as usize * CODE_ROW_SIZE) as u64 {
        return false;
    }

    for blk_idx in 0..blocks {
        let block_width = min(block_cols, factor_count - blk_idx * block_cols);
        let blk_meta = match fs::metadata(block_path(cache_dir, blk_idx)) {
            Ok(m) => m,
            Err(_) => return false,
        };
        let expected_size = (record_count * block_width * 8) as u64;
        if blk_meta.len() != expected_size {
            return false;
        }
    }

    true
}

fn validate_manifest(cache_dir: &Path, backup_file: &Path, manifest: &CacheManifest) -> PyResult<bool> {
    if !validate_cache_files(cache_dir, manifest) {
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

fn build_cache_internal(backup_file: &str, block_cols: usize, force_rebuild: bool) -> PyResult<BuildSummary> {
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
    let record_count = header.record_count;
    let factor_count = header.factor_count;
    let record_size = header.record_size;
    let block_count = expected_block_count(factor_count, block_cols);

    let mut meta_writer = BufWriter::with_capacity(
        16 * 1024 * 1024,
        File::create(meta_path(&cache_dir)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("创建 meta.bin 失败: {e}"))
        })?,
    );

    let mut block_writers = Vec::with_capacity(block_count);
    for blk_idx in 0..block_count {
        let file = File::create(block_path(&cache_dir, blk_idx)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "创建块文件 blk_{blk_idx:03}.bin 失败: {e}"
            ))
        })?;
        block_writers.push(BufWriter::with_capacity(16 * 1024 * 1024, file));
    }

    let mut code_to_id: HashMap<CodeKey, u32> = HashMap::new();
    let mut codebook: Vec<CodeKey> = Vec::new();

    for row in 0..record_count {
        let record_offset = HEADER_SIZE + row * record_size;
        let record = &mmap[record_offset..record_offset + record_size];

        let date = read_i64_le(record, DATE_OFFSET);
        let code_len = min(read_u32_le(record, CODE_LEN_OFFSET) as usize, 32);
        let mut code_bytes = [0u8; 32];
        code_bytes.copy_from_slice(&record[CODE_BYTES_OFFSET..CODE_BYTES_OFFSET + 32]);

        let key = CodeKey {
            code_len: code_len as u8,
            code_bytes,
        };
        let code_id = if let Some(existing) = code_to_id.get(&key) {
            *existing
        } else {
            let new_id = codebook.len() as u32;
            code_to_id.insert(key.clone(), new_id);
            codebook.push(key);
            new_id
        };

        meta_writer
            .write_all(&date.to_le_bytes())
            .and_then(|_| meta_writer.write_all(&code_id.to_le_bytes()))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入 meta.bin 失败: {e}"))
            })?;

        for blk_idx in 0..block_count {
            let start_col = blk_idx * block_cols;
            let end_col = min(start_col + block_cols, factor_count);
            let start_byte = FACTOR_BASE_OFFSET + start_col * 8;
            let end_byte = FACTOR_BASE_OFFSET + end_col * 8;
            block_writers[blk_idx]
                .write_all(&record[start_byte..end_byte])
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "写入块文件 blk_{blk_idx:03}.bin 失败: {e}"
                    ))
                })?;
        }
    }

    meta_writer.flush().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("刷新 meta.bin 失败: {e}"))
    })?;
    for (blk_idx, writer) in block_writers.iter_mut().enumerate() {
        writer.flush().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "刷新块文件 blk_{blk_idx:03}.bin 失败: {e}"
            ))
        })?;
    }

    let mut codes_writer = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(codes_path(&cache_dir)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("创建 codes.bin 失败: {e}"))
        })?,
    );
    for code in &codebook {
        codes_writer
            .write_all(&(code.code_len as u32).to_le_bytes())
            .and_then(|_| codes_writer.write_all(&code.code_bytes))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("写入 codes.bin 失败: {e}"))
            })?;
    }
    codes_writer.flush().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("刷新 codes.bin 失败: {e}"))
    })?;

    let (source_size, source_mtime_sec, source_mtime_nsec) = source_fingerprint(backup_path)?;
    let manifest = CacheManifest {
        magic: MANIFEST_MAGIC,
        version: MANIFEST_VERSION,
        source_size,
        source_mtime_sec,
        source_mtime_nsec,
        record_count: record_count as u64,
        factor_count: factor_count as u32,
        record_size: record_size as u32,
        block_cols: block_cols as u32,
        code_count: codebook.len() as u32,
    };
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
    let cache_dir = cache_dir_path(backup_file);

    if let Some(manifest) = try_get_valid_manifest(&cache_dir, backup_path)? {
        return Ok((cache_dir, manifest));
    }

    if !build_if_missing {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            "缓存不存在或已失效，请先构建缓存",
        ));
    }

    let summary = build_cache_internal(backup_file, 32, false)?;
    Ok((summary.cache_dir, summary.manifest))
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
        dict.set_item("rebuilt", summary.rebuilt)?;
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

    let meta_file = File::open(meta_path(&cache_dir)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("打开 meta.bin 失败: {e}"))
    })?;
    let blk_file = File::open(block_path(&cache_dir, block_idx)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("打开块文件失败: {e}"))
    })?;

    let meta_mmap = unsafe {
        Mmap::map(&meta_file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("映射 meta.bin 失败: {e}"))
        })?
    };
    let blk_mmap = unsafe {
        Mmap::map(&blk_file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("映射块文件失败: {e}"))
        })?
    };

    let mut dates = Vec::with_capacity(record_count);
    let mut code_ids = Vec::with_capacity(record_count);
    let mut factors = Vec::with_capacity(record_count);
    let block_row_bytes = block_width * 8;

    for row in 0..record_count {
        let meta_offset = row * META_ROW_SIZE;
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
    if !validate_cache_files(&cache_dir_path, &manifest) {
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
    let block_row_bytes = block_width * 8;

    let meta_file = File::open(meta_path(&cache_dir_path)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("打开 meta.bin 失败: {e}"))
    })?;
    let blk_file = File::open(block_path(&cache_dir_path, block_idx)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("打开块文件失败: {e}"))
    })?;

    let meta_mmap = unsafe {
        Mmap::map(&meta_file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("映射 meta.bin 失败: {e}"))
        })?
    };
    let blk_mmap = unsafe {
        Mmap::map(&blk_file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("映射块文件失败: {e}"))
        })?
    };

    let mut dates = Vec::with_capacity(record_count);
    let mut code_ids = Vec::with_capacity(record_count);
    let mut factors = Vec::with_capacity(record_count);

    for row in 0..record_count {
        let meta_offset = row * META_ROW_SIZE;
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

#[pyfunction]
#[pyo3(signature = (backup_file))]
pub fn query_backup_codebook_cached(backup_file: String) -> PyResult<PyObject> {
    let (cache_dir, manifest) = ensure_cache(&backup_file, true)?;
    let code_bytes = fs::read(codes_path(&cache_dir)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 codes.bin 失败: {e}"))
    })?;

    let expected_size = manifest.code_count as usize * CODE_ROW_SIZE;
    if code_bytes.len() != expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "codes.bin 大小不符合 manifest 记录，缓存可能损坏",
        ));
    }

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        for idx in 0..manifest.code_count as usize {
            let offset = idx * CODE_ROW_SIZE;
            let code_len = min(read_u32_le(&code_bytes, offset) as usize, 32);
            let bytes = &code_bytes[offset + 4..offset + 4 + 32];
            let code = String::from_utf8_lossy(&bytes[..code_len]).into_owned();
            dict.set_item(idx as u32, code)?;
        }
        Ok(dict.into())
    })
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
    if !validate_cache_files(&cache_dir_path, &manifest) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "缓存目录结构或文件大小无效，缓存可能损坏",
        ));
    }

    let code_bytes = fs::read(codes_path(&cache_dir_path)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("读取 codes.bin 失败: {e}"))
    })?;

    let expected_size = manifest.code_count as usize * CODE_ROW_SIZE;
    if code_bytes.len() != expected_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "codes.bin 大小不符合 manifest 记录，缓存可能损坏",
        ));
    }

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        for idx in 0..manifest.code_count as usize {
            let offset = idx * CODE_ROW_SIZE;
            let code_len = min(read_u32_le(&code_bytes, offset) as usize, 32);
            let bytes = &code_bytes[offset + 4..offset + 4 + 32];
            let code = String::from_utf8_lossy(&bytes[..code_len]).into_owned();
            dict.set_item(idx as u32, code)?;
        }
        Ok(dict.into())
    })
}
