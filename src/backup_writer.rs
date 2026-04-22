use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::backup_reader::TaskResult;

const HEADER_SIZE: usize = 64;
const V3_CODE_SIZE: usize = 16;

#[repr(C)]
struct FileHeaderV3 {
    magic: [u8; 8],
    version: u32,
    record_count: u64,
    record_size: u32,
    factor_count: u32,
    code_size: u32,
    reserved: [u8; 32],
}

/// v4 文件头：用 chunk_count 替换 code_size
#[repr(C, packed)]
struct FileHeaderV4 {
    magic: [u8; 8],
    version: u32,
    record_count: u64,
    record_size: u32,
    factor_count: u32,
    chunk_count: u32,
    reserved: [u8; 32],
}

/// 计算 v3 格式单条记录大小：36 + F*4
pub fn calculate_v3_record_size(factor_count: usize) -> usize {
    8 + V3_CODE_SIZE + 8 + factor_count * 4 + 4 // date + code + timestamp + factors(f32) + checksum
}

/// 计算 v3 校验和
fn calculate_v3_checksum(date: i64, code: &[u8; V3_CODE_SIZE], timestamp: i64, factors_f32: &[f32]) -> u32 {
    let mut sum = 0u32;
    sum = sum.wrapping_add(date as u32);
    sum = sum.wrapping_add((date >> 32) as u32);
    for &b in code {
        sum = sum.wrapping_add(b as u32);
    }
    sum = sum.wrapping_add(timestamp as u32);
    sum = sum.wrapping_add((timestamp >> 32) as u32);
    for &f in factors_f32 {
        sum = sum.wrapping_add(f.to_bits());
    }
    sum
}

/// 将 TaskResult 序列化为 v3 字节（v4 chunk 内部仍然使用 v3 记录格式）
fn serialize_record_v3(result: &TaskResult, factor_count: usize) -> Result<Vec<u8>, String> {
    let code_bytes = result.code.as_bytes();
    if code_bytes.len() > 15 {
        return Err(format!(
            "代码 '{}' 长度 {} 超过 15 字节限制",
            result.code,
            code_bytes.len()
        ));
    }

    let record_size = calculate_v3_record_size(factor_count);
    let mut buf = vec![0u8; record_size];

    // date (offset 0, 8 bytes)
    buf[0..8].copy_from_slice(&result.date.to_le_bytes());

    // code (offset 8, 16 bytes, null-terminated)
    let copy_len = std::cmp::min(code_bytes.len(), 15);
    buf[8..8 + copy_len].copy_from_slice(&code_bytes[..copy_len]);

    // timestamp (offset 24, 8 bytes)
    buf[24..32].copy_from_slice(&result.timestamp.to_le_bytes());

    // factors f64→f32 (offset 32, F*4 bytes)
    let factors_f32: Vec<f32> = result.facs.iter().map(|&v| v as f32).collect();
    for (i, &f) in factors_f32.iter().enumerate() {
        if i >= factor_count {
            break;
        }
        let offset = 32 + i * 4;
        buf[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
    }

    // checksum (offset 32 + F*4, 4 bytes)
    let checksum = calculate_v3_checksum(
        result.date,
        unsafe { &*(buf.as_ptr().add(8) as *const [u8; V3_CODE_SIZE]) },
        result.timestamp,
        &factors_f32,
    );
    let cksum_offset = 32 + factor_count * 4;
    buf[cksum_offset..cksum_offset + 4].copy_from_slice(&checksum.to_le_bytes());

    Ok(buf)
}

/// 将 TaskResult 数组写入 v4 分块压缩格式备份文件
/// - 文件不存在或无效 → 创建 v4 文件
/// - 文件存在且 version=4 → 追加（校验 factor_count 一致）
/// - 文件存在且 version=3 → 报错（v3 文件不支持追加 v4 数据）
/// - 文件存在且 version≠3/4 → 报错
pub fn save_results_to_backup(
    results: &[TaskResult],
    backup_file: &str,
    expected_result_length: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if results.is_empty() {
        return Ok(());
    }

    let factor_count = expected_result_length;
    let record_size = calculate_v3_record_size(factor_count);

    let file_path = Path::new(backup_file);
    let file_exists = file_path.exists();
    let file_valid = if file_exists {
        file_path
            .metadata()
            .map(|m| m.len() >= HEADER_SIZE as u64)
            .unwrap_or(false)
    } else {
        false
    };

    // 序列化本批所有记录为连续字节缓冲
    let mut raw_buf = Vec::with_capacity(results.len() * record_size);
    let mut written_count: usize = 0;
    for result in results.iter() {
        match serialize_record_v3(result, factor_count) {
            Ok(record_bytes) => {
                raw_buf.extend_from_slice(&record_bytes);
                written_count += 1;
            }
            Err(e) => {
                eprintln!("WARNING: 跳过无效记录: {}", e);
            }
        }
    }

    if written_count == 0 {
        return Ok(());
    }

    // zstd 压缩
    let compressed = zstd::encode_all(&raw_buf[..], 9)?;
    let compressed_size = compressed.len() as u32;

    if !file_valid {
        // 创建新文件，写入 v4 文件头 + 第一个 chunk
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(backup_file)?;

        let header = FileHeaderV4 {
            magic: *b"RPBACKUP",
            version: 4,
            record_count: written_count as u64,
            record_size: record_size as u32,
            factor_count: factor_count as u32,
            chunk_count: 1,
            reserved: [0; 32],
        };

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const FileHeaderV4 as *const u8,
                std::mem::size_of::<FileHeaderV4>(),
            )
        };

        file.write_all(header_bytes)?;

        // 写入 chunk: [compressed_size: u32][record_count: u32][compressed_data]
        file.write_all(&compressed_size.to_le_bytes())?;
        file.write_all(&(written_count as u32).to_le_bytes())?;
        file.write_all(&compressed)?;
        file.flush()?;
        return Ok(());
    }

    // 文件已存在，读取头部判断版本
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(backup_file)?;

    let file_len = file.metadata()?.len() as usize;
    if file_len < HEADER_SIZE {
        return Err(format!("File too small: {} < {}", file_len, HEADER_SIZE).into());
    }

    let mut header_bytes = [0u8; 64];
    file.read_exact(&mut header_bytes)?;

    let version = u32::from_le_bytes(header_bytes[8..12].try_into()?);
    if version != 4 {
        return Err(format!(
            "不支持的备份文件版本 {}，仅支持 v4 格式写入",
            version
        )
        .into());
    }

    // 检查 factor_count
    let file_factor_count = u32::from_le_bytes(header_bytes[24..28].try_into()?) as usize;
    if file_factor_count != factor_count {
        return Err(format!(
            "Factor count mismatch: file has {}, expected {}",
            file_factor_count, factor_count
        )
        .into());
    }

    let current_record_count = u64::from_le_bytes(header_bytes[12..20].try_into()?);
    let current_chunk_count = u32::from_le_bytes(header_bytes[28..32].try_into()?);

    // 追加新 chunk 到文件末尾
    file.seek(SeekFrom::End(0))?;
    file.write_all(&compressed_size.to_le_bytes())?;
    file.write_all(&(written_count as u32).to_le_bytes())?;
    file.write_all(&compressed)?;
    file.flush()?;

    // 更新文件头：record_count 和 chunk_count
    let new_record_count = current_record_count + written_count as u64;
    let new_chunk_count = current_chunk_count + 1;

    file.seek(SeekFrom::Start(12))?;
    file.write_all(&new_record_count.to_le_bytes())?;
    file.seek(SeekFrom::Start(28))?;
    file.write_all(&new_chunk_count.to_le_bytes())?;
    file.flush()?;

    Ok(())
}
