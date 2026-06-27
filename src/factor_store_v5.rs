//! # 列式因子存储 RPFBINV5
//!
//! 统一"计算时增量备份"与"回测时并发顺序读取"的存储格式。
//!
//! ## 设计目标
//! - **计算时**：collector 线程攒批后把 N 条任务结果按列重排，追加成 1 个 chunk（纯顺序追加）。
//! - **回测时**：计算结束后投影一次，让"同一因子的所有行"连续存放，单线程顺序读零寻道。
//! - **断点续算**：重启时从 idx 字典 + colblk header 恢复已写记录集合。
//!
//! ## 文件布局
//! - `factors.colblk`：数据文件（顺序追加）。Header + 多个 zstd chunk（列优先）+ 投影列区。
//! - `factors.idx`：字典/索引文件。因子名 + 日期字典 + 股票字典 + projected_flag。
//!
//! ## 布局选择的理由
//! 追加阶段任务乱序到达（多进程并行），无法预知完整排序，所以 chunk 内列优先（因子0全部→因子1全部…）。
//! 这样回测读因子 k 时，在单个 chunk 内只需读 k×n 段；跨 chunk 的跳读靠投影消除。
//! 投影把"按 chunk 组织"重排为"按因子组织"，让回测的列扫描变成单次连续顺序读。

use crate::backup_reader::TaskResult;
use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

// ============================ 常量 ============================

/// 数据文件魔数 "RPFBINV5"
const COLBLK_MAGIC: [u8; 8] = *b"RPFBINV5";
/// 索引文件魔数 "RPFBIDX5"
const IDX_MAGIC: [u8; 8] = *b"RPFBIDX5";
const COLBLK_VERSION: u32 = 5;
const IDX_VERSION: u32 = 5;
/// colblk header 固定 64 字节
const COLBLK_HEADER_SIZE: usize = 64;
/// idx header 固定 64 字节
const IDX_HEADER_SIZE: usize = 64;
/// zstd 压缩等级
const ZSTD_LEVEL: i32 = 1; // level 1：浮点数据压缩比loss小，速度快5-10倍
/// date_id / code_id 编码宽度（用 u32 留余量，突破旧 colblk_cache_v2 的 u16 上限）
const ID_BYTES: usize = 4;
/// 股票代码在字典里的固定槽位
const CODE_SLOT_BYTES: usize = 16;
/// 因子值宽度（f32）
const F32_BYTES: usize = 4;
/// 投影区格式版本，写在 colblk header reserved[44..48]。
/// 0 = 追加期未投影；2 = 去冗余新格式（共享行序 + 每因子 value 列）。旧 1.x 格式不兼容。
const PROJ_FORMAT_VERSION: u32 = 2;

// ============================ colblk Header ============================
// 字段布局（小端）：
//   [0..8]   magic "RPFBINV5"
//   [8..12]  version u32
//   [12..20] record_count u64  —— 已写入的总行数（date×code 对）
//   [20..24] factor_count u32  —— 原始因子数 F
//   [24..28] chunk_count u32   —— 已追加 chunk 数
//   [28..32] date_count u32    —— 字典日期数（追加期增量增长）
//   [32..36] code_count u32    —— 字典股票数
//   [36..44] projected_offset u64 —— 投影区起始偏移；0 表示未投影（追加期）
//            注意：追加期此字段恒为 0，下一 chunk 写入位置由"文件末尾"决定（seek End）。
//   [44..64] reserved

fn write_colblk_header_fields(
    file: &mut File,
    record_count: u64,
    factor_count: u32,
    chunk_count: u32,
    date_count: u32,
    code_count: u32,
    projected_offset: u64,
    proj_format_version: u32,
) -> Result<(), String> {
    let mut buf = [0u8; COLBLK_HEADER_SIZE];
    buf[0..8].copy_from_slice(&COLBLK_MAGIC);
    buf[8..12].copy_from_slice(&COLBLK_VERSION.to_le_bytes());
    buf[12..20].copy_from_slice(&record_count.to_le_bytes());
    buf[20..24].copy_from_slice(&factor_count.to_le_bytes());
    buf[24..28].copy_from_slice(&chunk_count.to_le_bytes());
    buf[28..32].copy_from_slice(&date_count.to_le_bytes());
    buf[32..36].copy_from_slice(&code_count.to_le_bytes());
    buf[36..44].copy_from_slice(&projected_offset.to_le_bytes());
    buf[44..48].copy_from_slice(&proj_format_version.to_le_bytes());
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek header 失败: {e}"))?;
    file.write_all(&buf)
        .map_err(|e| format!("写 header 失败: {e}"))?;
    file.flush()
        .map_err(|e| format!("flush header 失败: {e}"))?;
    Ok(())
}

/// 解析 colblk header
pub struct ColblkHeader {
    pub record_count: u64,
    pub factor_count: usize,
    pub chunk_count: u32,
    pub date_count: usize,
    pub code_count: usize,
    pub projected_offset: u64,
    pub proj_format_version: u32,
}

fn parse_colblk_header(bytes: &[u8]) -> Result<ColblkHeader, String> {
    if bytes.len() < COLBLK_HEADER_SIZE {
        return Err(format!(
            "colblk header 过短: {} < {}",
            bytes.len(),
            COLBLK_HEADER_SIZE
        ));
    }
    if &bytes[0..8] != &COLBLK_MAGIC {
        return Err(format!(
            "colblk 魔数错误: {:?}（期望 {:?}）",
            &bytes[0..8],
            COLBLK_MAGIC
        ));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != COLBLK_VERSION {
        return Err(format!(
            "不支持的 colblk 版本 {version}，期望 {COLBLK_VERSION}"
        ));
    }
    Ok(ColblkHeader {
        record_count: u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
        factor_count: u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize,
        chunk_count: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
        date_count: u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize,
        code_count: u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize,
        projected_offset: u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
        proj_format_version: u32::from_le_bytes(bytes[44..48].try_into().unwrap()),
    })
}

// ============================ idx 文件 ============================
// 布局：
//   [0..8]   magic "RPFBIDX5"
//   [8..12]  version u32
//   [12..16] factor_count u32
//   [16..20] date_count u32
//   [20..24] code_count u32
//   [24..32] projected_flag u64（0=追加中, 非0=已投影）
//   [32..64] reserved
//   因子名区：factor_count × 长度前缀字符串（u16 长度 + utf8 字节）
//   日期字典：date_count × i64（小端）
//   股票字典：code_count × 16 字节（null-terminated）

struct IdxHeader {
    factor_count: usize,
    date_count: usize,
    code_count: usize,
    projected_flag: u64,
}

fn parse_idx_header(bytes: &[u8]) -> Result<IdxHeader, String> {
    if bytes.len() < IDX_HEADER_SIZE {
        return Err("idx header 过短".to_string());
    }
    if &bytes[0..8] != &IDX_MAGIC {
        return Err(format!("idx 魔数错误: {:?}", &bytes[0..8]));
    }
    Ok(IdxHeader {
        factor_count: u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize,
        date_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize,
        code_count: u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize,
        projected_flag: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    })
}

/// 字典：因子名、日期、股票代码。所有 id 对应其在数组中的下标。
pub struct FactorDict {
    pub factor_names: Vec<String>,
    /// date_id → 真实日期 i64
    pub dates: Vec<i64>,
    /// code_id → 股票代码（不带交易所后缀）
    pub codes: Vec<String>,
    /// 反查：真实日期 → date_id
    pub date_to_id: HashMap<i64, u32>,
    /// 反查：股票代码 → code_id
    pub code_to_id: HashMap<String, u32>,
}

impl FactorDict {
    fn new(factor_names: Vec<String>) -> Self {
        Self {
            factor_names,
            dates: Vec::new(),
            codes: Vec::new(),
            date_to_id: HashMap::new(),
            code_to_id: HashMap::new(),
        }
    }

    /// 查/分配 date_id（不存在则分配）
    fn intern_date(&mut self, date: i64) -> u32 {
        if let Some(&id) = self.date_to_id.get(&date) {
            return id;
        }
        let id = self.dates.len() as u32;
        self.dates.push(date);
        self.date_to_id.insert(date, id);
        id
    }

    fn intern_code(&mut self, code: &str) -> u32 {
        if let Some(&id) = self.code_to_id.get(code) {
            return id;
        }
        let id = self.codes.len() as u32;
        self.codes.push(code.to_string());
        self.code_to_id.insert(code.to_string(), id);
        id
    }

    /// 序列化 idx 文件（整个重写，因为字典会增量增长且文件不大）
    fn write_idx(&self, path: &Path, projected_flag: u64) -> Result<(), String> {
        let mut buf: Vec<u8> = Vec::with_capacity(
            IDX_HEADER_SIZE
                + self.factor_names.len() * 40
                + self.dates.len() * 8
                + self.codes.len() * CODE_SLOT_BYTES,
        );
        // header
        let mut hdr = [0u8; IDX_HEADER_SIZE];
        hdr[0..8].copy_from_slice(&IDX_MAGIC);
        hdr[8..12].copy_from_slice(&IDX_VERSION.to_le_bytes());
        hdr[12..16].copy_from_slice(&(self.factor_names.len() as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(self.dates.len() as u32).to_le_bytes());
        hdr[20..24].copy_from_slice(&(self.codes.len() as u32).to_le_bytes());
        hdr[24..32].copy_from_slice(&projected_flag.to_le_bytes());
        buf.extend_from_slice(&hdr);
        // 因子名
        for name in &self.factor_names {
            let bytes = name.as_bytes();
            let len = bytes.len() as u16;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        // 日期字典
        for &d in &self.dates {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        // 股票字典（固定 16 字节槽，null-terminated）
        for code in &self.codes {
            let mut slot = [0u8; CODE_SLOT_BYTES];
            let bytes = code.as_bytes();
            let copy_len = bytes.len().min(CODE_SLOT_BYTES - 1);
            slot[..copy_len].copy_from_slice(&bytes[..copy_len]);
            buf.extend_from_slice(&slot);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("创建 idx 文件失败: {e}"))?;
        file.write_all(&buf)
            .map_err(|e| format!("写 idx 失败: {e}"))?;
        file.flush().map_err(|e| format!("flush idx 失败: {e}"))?;
        Ok(())
    }

    /// 读取 idx 文件
    fn read_idx(path: &Path) -> Result<(Self, u64), String> {
        let bytes = std::fs::read(path).map_err(|e| format!("读取 idx 文件失败 {path:?}: {e}"))?;
        let hdr = parse_idx_header(&bytes)?;
        let mut pos = IDX_HEADER_SIZE;
        // 因子名
        let mut factor_names = Vec::with_capacity(hdr.factor_count);
        for _ in 0..hdr.factor_count {
            if pos + 2 > bytes.len() {
                return Err("idx 因子名区截断".to_string());
            }
            let len = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            if pos + len > bytes.len() {
                return Err("idx 因子名截断".to_string());
            }
            let name = String::from_utf8(bytes[pos..pos + len].to_vec())
                .map_err(|e| format!("idx 因子名 utf8 失败: {e}"))?;
            factor_names.push(name);
            pos += len;
        }
        // 日期
        let mut dates = Vec::with_capacity(hdr.date_count);
        let mut date_to_id = HashMap::with_capacity(hdr.date_count);
        for i in 0..hdr.date_count {
            if pos + 8 > bytes.len() {
                return Err("idx 日期区截断".to_string());
            }
            let d = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            date_to_id.insert(d, i as u32);
            dates.push(d);
        }
        // 股票
        let mut codes = Vec::with_capacity(hdr.code_count);
        let mut code_to_id = HashMap::with_capacity(hdr.code_count);
        for i in 0..hdr.code_count {
            if pos + CODE_SLOT_BYTES > bytes.len() {
                return Err("idx 股票区截断".to_string());
            }
            let slot = &bytes[pos..pos + CODE_SLOT_BYTES];
            pos += CODE_SLOT_BYTES;
            let end = slot.iter().position(|&b| b == 0).unwrap_or(CODE_SLOT_BYTES);
            let code = String::from_utf8(slot[..end].to_vec())
                .map_err(|e| format!("idx 股票 utf8 失败: {e}"))?;
            code_to_id.insert(code.clone(), i as u32);
            codes.push(code);
        }
        Ok((
            Self {
                factor_names,
                dates,
                codes,
                date_to_id,
                code_to_id,
            },
            hdr.projected_flag,
        ))
    }
}

// ============================ Writer ============================

/// 列式因子存储写入器。
///
/// 生命周期：
/// 1. `open` / `create`：新建或断点续算加载
/// 2. `append_batch`：collector 线程反复调用，攒批按列重排追加 chunk
/// 3. `finish_and_project`：计算结束后投影，让回测顺序读
pub struct FactorStoreWriter {
    store_dir: PathBuf,
    colblk_path: PathBuf,
    idx_path: PathBuf,
    colblk_file: File,
    dict: FactorDict,
    factor_count: usize,
    record_count: u64,
    chunk_count: u32,
    projected_offset: u64,
    /// chunk 索引：每个 chunk 的 (data_file_offset, compressed_size, n_in_batch)
    /// 用于投影阶段快速定位
    chunk_index: Vec<(u64, u32, u32)>,
    /// 投影区格式版本（0=未投影/旧，PROJ_FORMAT_VERSION=新格式）。投影后置位。
    proj_format_version: u32,
}

impl FactorStoreWriter {
    /// 新建存储（或断点续算）。
    /// `factor_names`：原始因子名列表，长度必须等于每条结果的因子数。
    pub fn open(store_dir: &str, factor_names: &[String]) -> Result<Self, String> {
        let store_dir = PathBuf::from(store_dir);
        std::fs::create_dir_all(&store_dir).map_err(|e| format!("创建 store_dir 失败: {e}"))?;
        let colblk_path = store_dir.join("factors.colblk");
        let idx_path = store_dir.join("factors.idx");

        let factor_count = factor_names.len();

        // 断点续算：colblk 和 idx 都存在 → 加载并校验因子数一致
        if colblk_path.exists() && idx_path.exists() {
            return Self::resume(colblk_path, idx_path, factor_names);
        }

        // 全新创建
        let mut colblk_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&colblk_path)
            .map_err(|e| format!("创建 colblk 失败: {e}"))?;

        let dict = FactorDict::new(factor_names.to_vec());
        // 写空 header（record_count=0 等），然后定位到 header 之后等待追加
        write_colblk_header_fields(&mut colblk_file, 0, factor_count as u32, 0, 0, 0, 0, 0)?;
        colblk_file
            .seek(SeekFrom::Start(COLBLK_HEADER_SIZE as u64))
            .map_err(|e| format!("seek 起始位置失败: {e}"))?;
        // 写 idx（空字典）
        dict.write_idx(&idx_path, 0)?;

        Ok(Self {
            store_dir,
            colblk_path,
            idx_path,
            colblk_file,
            dict,
            factor_count,
            record_count: 0,
            chunk_count: 0,
            projected_offset: 0,
            chunk_index: Vec::new(),
            proj_format_version: 0,
        })
    }

    /// 断点续算：加载已有 colblk + idx，重建 chunk_index
    fn resume(
        colblk_path: PathBuf,
        idx_path: PathBuf,
        expected_factor_names: &[String],
    ) -> Result<Self, String> {
        let (dict, _projected_flag) = FactorDict::read_idx(&idx_path)?;
        if dict.factor_names.len() != expected_factor_names.len() {
            return Err(format!(
                "断点续算因子数不匹配: idx={}, 期望={}",
                dict.factor_names.len(),
                expected_factor_names.len()
            ));
        }
        let factor_count = dict.factor_names.len();

        // 解析 colblk header
        let mut colblk_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&colblk_path)
            .map_err(|e| format!("打开 colblk 失败: {e}"))?;
        let mut hdr_buf = [0u8; COLBLK_HEADER_SIZE];
        colblk_file
            .read_exact(&mut hdr_buf)
            .map_err(|e| format!("读 colblk header 失败: {e}"))?;
        let hdr = parse_colblk_header(&hdr_buf)?;
        if hdr.factor_count != factor_count {
            return Err(format!(
                "colblk factor_count={} 与 idx factor_count={} 不一致",
                hdr.factor_count, factor_count
            ));
        }

        // 扫描所有 chunk 重建 chunk_index（每个 chunk 头 = [compressed_size:u32][n_in_batch:u32]）
        let file_len = std::fs::metadata(&colblk_path)
            .map_err(|e| format!("读 colblk 元数据失败: {e}"))?
            .len();
        // chunk 起点上限：如果有投影区，只扫到 projected_offset（投影区不能当 chunk 解析）
        let scan_end = if hdr.projected_offset > 0 {
            hdr.projected_offset
        } else {
            file_len
        };
        let mut chunk_index: Vec<(u64, u32, u32)> = Vec::with_capacity(hdr.chunk_count as usize);
        let mut offset = COLBLK_HEADER_SIZE as u64;
        let mut chunk_count = 0u32;
        let mut record_count = 0u64;
        while offset + 8 <= scan_end {
            colblk_file
                .seek(SeekFrom::Start(offset))
                .map_err(|e| format!("seek chunk 失败: {e}"))?;
            let mut chunk_hdr = [0u8; 8];
            if colblk_file.read_exact(&mut chunk_hdr).is_err() {
                break; // 截断的不完整 chunk，停止
            }
            let compressed_size = u32::from_le_bytes(chunk_hdr[0..4].try_into().unwrap());
            let n_in_batch = u32::from_le_bytes(chunk_hdr[4..8].try_into().unwrap());
            let data_offset = offset + 8;
            let body_row_size = ID_BYTES * 2 + hdr.factor_count * F32_BYTES;
            // chunk_index 存原始 compressed_size（0=不压缩），offset 计算用实际 data 大小
            let data_size = if compressed_size == 0 {
                (n_in_batch as usize * body_row_size) as u32
            } else {
                compressed_size
            };
            chunk_index.push((data_offset, compressed_size, n_in_batch));
            offset = data_offset + data_size as u64;
            chunk_count += 1;
            record_count += n_in_batch as u64;
        }

        // projected_offset 从 header 取（未投影时为 0）；追加期下一写入位置 = 当前 offset（文件末尾）
        let projected_offset = hdr.projected_offset;

        // 定位到当前末尾，准备追加
        colblk_file
            .seek(SeekFrom::End(0))
            .map_err(|e| format!("seek 末尾失败: {e}"))?;

        Ok(Self {
            store_dir: colblk_path.parent().unwrap().to_path_buf(),
            colblk_path,
            idx_path,
            colblk_file,
            dict,
            factor_count,
            record_count,
            chunk_count,
            projected_offset,
            proj_format_version: hdr.proj_format_version,
            chunk_index,
        })
    }

    /// 追加一批任务结果。collector 线程调用，攒批后写入。
    /// 内部：按列重排（因子0全部 → 因子1全部…）+ zstd 压缩 + 追加 chunk + 更新 header/idx。
    pub fn append_batch(&mut self, results: &[TaskResult]) -> Result<(), String> {
        if results.is_empty() {
            return Ok(());
        }
        // 已投影后不允许再追加（会破坏投影区布局）
        if self.projected_offset > 0 {
            return Err("存储已投影，不能再追加数据".to_string());
        }
        let n = results.len();
        let factor_count = self.factor_count;

        // ---- 1. 分配 date_id / code_id（字典增量更新）----
        let mut date_ids = Vec::with_capacity(n);
        let mut code_ids = Vec::with_capacity(n);
        for r in results {
            date_ids.push(self.dict.intern_date(r.date));
            code_ids.push(self.dict.intern_code(&r.code));
        }

        // ---- 2. 列优先布局序列化 ----
        // chunk 体 = [n×u32 date_id][n×u32 code_id][n×F×f32 factors(列优先)]
        let body_size = n * ID_BYTES * 2 + n * factor_count * F32_BYTES;
        let mut body: Vec<u8> = Vec::with_capacity(body_size);
        for &d in &date_ids {
            body.extend_from_slice(&d.to_le_bytes());
        }
        for &c in &code_ids {
            body.extend_from_slice(&c.to_le_bytes());
        }
        // 因子值：列优先（先因子0的 n 个值，再因子1…）。f64→f32 截断（与现有 backup_writer 一致）
        for f_idx in 0..factor_count {
            for r in results {
                let v = if r.facs.len() > f_idx {
                    r.facs[f_idx] // 已是 f32
                } else {
                    f32::NAN
                };
                body.extend_from_slice(&v.to_le_bytes());
            }
        }

        // ---- 3. 不压缩（浮点数据 zstd-1 压缩比仅 1.15×，但解压 55min/遍太慢）----
        // chunk 头用 compressed_size=0 标记"不压缩"，data 直接是 body。
        let compressed_size = 0u32; // 0 = 不压缩标记
        let data_bytes = &body[..];

        // ---- 4. 追加 chunk 到 colblk 末尾 ----
        // 追加期 projected_offset==0，直接 seek End 追加。
        self.colblk_file
            .seek(SeekFrom::End(0))
            .map_err(|e| format!("seek End 失败: {e}"))?;
        let chunk_head_offset = self
            .colblk_file
            .stream_position()
            .map_err(|e| format!("获取 chunk 头偏移失败: {e}"))?;
        // 写 chunk 头 [compressed_size][n_in_batch] + 压缩数据
        self.colblk_file
            .write_all(&compressed_size.to_le_bytes())
            .map_err(|e| format!("写 chunk compressed_size 失败: {e}"))?;
        self.colblk_file
            .write_all(&(n as u32).to_le_bytes())
            .map_err(|e| format!("写 chunk n_in_batch 失败: {e}"))?;
        self.colblk_file
            .write_all(data_bytes)
            .map_err(|e| format!("写 chunk 数据失败: {e}"))?;
        self.colblk_file
            .flush()
            .map_err(|e| format!("flush chunk 失败: {e}"))?;

        // ---- 5. 更新内存状态 + header + idx ----
        // data 偏移 = chunk 头（8 字节）之后
        // chunk_index 第二元素 = chunk 头存的 compressed_size（0=不压缩标志）
        // 不压缩时 body 大小 = n × record_size，在 Reader 端按需计算
        self.chunk_index
            .push((chunk_head_offset + 8, 0u32, n as u32)); // 0 = 不压缩
        self.chunk_count += 1;
        self.record_count += n as u64;

        // 追加期 projected_offset 恒为 0
        write_colblk_header_fields(
            &mut self.colblk_file,
            self.record_count,
            self.factor_count as u32,
            self.chunk_count,
            self.dict.dates.len() as u32,
            self.dict.codes.len() as u32,
            0,
            self.proj_format_version,
        )?;
        // idx 字典会增长，整文件重写
        self.dict.write_idx(&self.idx_path, 0)?;

        Ok(())
    }

    /// 返回已写入记录的 (date, code) 集合，用于断点续算过滤。
    /// 扫描所有 chunk 解压，提取 date_id/code_id 并经字典还原。
    pub fn check_completed(&self) -> Result<HashSet<(i64, String)>, String> {
        let mut completed = HashSet::new();
        let file = File::open(&self.colblk_path).map_err(|e| format!("打开 colblk 失败: {e}"))?;
        let id_w = ID_BYTES;
        let mut skipped_out_of_range = 0u64;
        for (data_offset, compressed_size, n_in_batch) in &self.chunk_index {
            let n = *n_in_batch as usize;
            // 只读 date_id + code_id（每条 8 字节），用 pread 精确定位读，不触发 mmap readahead。
            // 不压缩时 id 区在 chunk 前 n*8 字节，pread 直接读；压缩时整体解压。
            let id_bytes: Vec<u8> = if *compressed_size == 0 {
                let id_size = n * id_w * 2;
                let mut buf = vec![0u8; id_size];
                if file
                    .read_at(&mut buf, *data_offset)
                    .map_err(|e| format!("读 id 区失败: {e}"))?
                    < id_size
                {
                    continue; // 崩溃残留：chunk 未完整落盘
                }
                buf
            } else {
                let comp_size = *compressed_size as usize;
                let mut buf = vec![0u8; comp_size];
                if file
                    .read_at(&mut buf, *data_offset)
                    .map_err(|e| format!("读压缩 chunk 失败: {e}"))?
                    < comp_size
                {
                    continue;
                }
                zstd::decode_all(&buf[..]).map_err(|e| format!("解压 chunk 失败: {e}"))?
            };
            // date_ids: [0..n*4], code_ids: [n*4..n*8]
            let code_base = n * id_w;
            for i in 0..n {
                let d_id = u32::from_le_bytes(id_bytes[i * id_w..i * id_w + 4].try_into().unwrap())
                    as usize;
                let c_id = u32::from_le_bytes(
                    id_bytes[code_base + i * id_w..code_base + i * id_w + 4]
                        .try_into()
                        .unwrap(),
                ) as usize;
                // 越界 id 跳过（崩溃残留：chunk 已落盘但 idx 字典未更新）
                match (self.dict.dates.get(d_id), self.dict.codes.get(c_id)) {
                    (Some(&date), Some(code)) => {
                        completed.insert((date, code.clone()));
                    }
                    _ => skipped_out_of_range += 1,
                }
            }
        }
        if skipped_out_of_range > 0 {
            eprintln!(
                "⚠️ 跳过 {} 条越界记录（崩溃残留，id 超出字典范围）",
                skipped_out_of_range
            );
        }
        Ok(completed)
    }

    /// 计算结束后投影：把"按 chunk 组织"重排为"按因子组织"（新格式 proj_format_version=2）。
    /// 新格式 = 共享行序段（[date_id,code_id]×total_rows 写一次）+ 每因子纯 value 列。
    /// 相比旧格式去掉每因子重复的 date_id/code_id（写量 3×→1×），并用 rayon 窗口化并行压缩。
    /// 投影完成后设置 colblk header projected_offset + proj_format_version=2，idx projected_flag=1。
    pub fn finish_and_project(&mut self, n_jobs: usize) -> Result<(), String> {
        let _ = n_jobs; // 投影用全局 rayon 池（计算已结束、全核可用）
        if self.record_count == 0 {
            // 空存储：写最小有效投影区（0因子0行），让 is_projected()=True。
            // 否则空分片 projected_offset=0 会使整个 sharded store 的 is_projected()=False，
            // 任何短日期回测（某些 date%8 桶为空）都会被误判为未投影。
            let project_start = COLBLK_HEADER_SIZE as u64; // 无 chunk，投影区紧接 header
            self.colblk_file
                .seek(SeekFrom::Start(project_start))
                .map_err(|e| format!("seek 空投影失败: {e}"))?;
            let mut hdr = Vec::with_capacity(32);
            hdr.extend_from_slice(&PROJ_FORMAT_VERSION.to_le_bytes());
            hdr.extend_from_slice(&0u32.to_le_bytes()); // factor_count=0（空投影无因子段）
            hdr.extend_from_slice(&0u64.to_le_bytes()); // total_rows=0
            hdr.extend_from_slice(&0u64.to_le_bytes()); // row_order_offset=0
            hdr.extend_from_slice(&0u64.to_le_bytes()); // row_order_csz=0
            self.colblk_file
                .write_all(&hdr)
                .map_err(|e| format!("写空投影头失败: {e}"))?;
            self.colblk_file
                .set_len(project_start + 32)
                .map_err(|e| format!("截断空投影失败: {e}"))?;
            self.projected_offset = project_start;
            self.proj_format_version = PROJ_FORMAT_VERSION;
            write_colblk_header_fields(
                &mut self.colblk_file,
                0,
                self.factor_count as u32,
                self.chunk_count,
                self.dict.dates.len() as u32,
                self.dict.codes.len() as u32,
                self.projected_offset,
                self.proj_format_version,
            )?;
            self.dict.write_idx(&self.idx_path, 1)?;
            return Ok(());
        }
        // 已是新格式投影 → 幂等返回（旧格式 version≠2 会落到下方重投影）
        if self.projected_offset > 0 && self.proj_format_version == PROJ_FORMAT_VERSION {
            return Ok(());
        }

        let factor_count = self.factor_count;
        let total_rows = self.record_count as usize;
        let id_w = ID_BYTES;

        // ---- 步骤①：顺序 pread 读所有 chunk，内存转置 + 收集共享行序 ----
        let colblk_file_read =
            File::open(&self.colblk_path).map_err(|e| format!("打开 colblk 读失败: {e}"))?;
        eprintln!(
            "🏗️ 投影①：顺序读 chunk 内存转置（峰值 ~{}GB）...",
            total_rows as u64 * factor_count as u64 * 4 / 1_000_000_000
        );
        let mut factor_values: Vec<Vec<f32>> =
            (0..factor_count).map(|_| Vec::with_capacity(total_rows)).collect();
        let mut date_ids: Vec<u32> = Vec::with_capacity(total_rows);
        let mut code_ids: Vec<u32> = Vec::with_capacity(total_rows);
        let n_chunks = self.chunk_index.len();
        for (ci, (data_offset, compressed_size, n_in_batch)) in self.chunk_index.iter().enumerate() {
            let n = *n_in_batch as usize;
            let body_row_size = id_w * 2 + factor_count * F32_BYTES;
            let data_size = if *compressed_size == 0 {
                n * body_row_size
            } else {
                *compressed_size as usize
            };
            let mut buf = vec![0u8; data_size];
            if colblk_file_read
                .read_at(&mut buf, *data_offset)
                .map_err(|e| format!("读 chunk 失败: {e}"))?
                < data_size
            {
                continue; // 崩溃残留
            }
            let dec_owned: Vec<u8>;
            let dec: &[u8] = if *compressed_size == 0 {
                &buf[..]
            } else {
                dec_owned =
                    zstd::decode_all(&buf[..]).map_err(|e| format!("投影解压 chunk 失败: {e}"))?;
                &dec_owned[..]
            };
            let code_base = n * id_w;
            let fac_base = n * id_w * 2;
            for i in 0..n {
                date_ids.push(u32::from_le_bytes(
                    dec[i * id_w..i * id_w + 4].try_into().unwrap(),
                ));
                code_ids.push(u32::from_le_bytes(
                    dec[code_base + i * id_w..code_base + i * id_w + 4]
                        .try_into()
                        .unwrap(),
                ));
            }
            // 转置因子值：chunk 内列优先 [因子0的n值][因子1的n值]...
            for f in 0..factor_count {
                let f_off = fac_base + f * n * F32_BYTES;
                for i in 0..n {
                    factor_values[f].push(f32::from_le_bytes(
                        dec[f_off + i * F32_BYTES..f_off + i * F32_BYTES + F32_BYTES]
                            .try_into()
                            .unwrap(),
                    ));
                }
            }
            if (ci + 1) % 500 == 0 {
                eprintln!(
                    "🏗️ 读取进度: {}/{} chunks ({}%)",
                    ci + 1,
                    n_chunks,
                    (ci + 1) * 100 / n_chunks
                );
            }
        }
        eprintln!("🏗️ 投影②：写共享行序段 + rayon 并行压缩每因子 value 列...");

        // ---- 步骤②：定位 chunk 区末尾（用 chunk_index 算，崩溃安全），truncate 掉旧/残缺投影 ----
        let chunk_area_end = self
            .chunk_index
            .last()
            .map(|(off, csz, n)| {
                let body = (id_w * 2 + factor_count * F32_BYTES) as u64;
                let dsize = if *csz == 0 { *n as u64 * body } else { *csz as u64 };
                off + dsize
            })
            .unwrap_or(COLBLK_HEADER_SIZE as u64);
        self.colblk_file
            .set_len(chunk_area_end)
            .map_err(|e| format!("truncate 旧投影失败: {e}"))?;
        let project_start = chunk_area_end;

        // ---- 步骤③：占位写投影头（保留空间，最后回填）----
        // 头布局：[proj_format_version u32][factor_count u32][total_rows u64]
        //         [row_order_offset u64][row_order_csz u64][factor_count × (val_off u64, val_csz u64)]
        let header_size = 32 + factor_count * 16;
        self.colblk_file
            .seek(SeekFrom::Start(project_start))
            .map_err(|e| format!("seek 投影起点失败: {e}"))?;
        self.colblk_file
            .write_all(&vec![0u8; header_size])
            .map_err(|e| format!("写投影头占位失败: {e}"))?;

        // ---- 步骤④：写共享行序段（[d_id,c_id] × total_rows，zstd 压缩）----
        let mut ro_bytes = Vec::with_capacity(total_rows * id_w * 2);
        for i in 0..total_rows {
            ro_bytes.extend_from_slice(&date_ids[i].to_le_bytes());
            ro_bytes.extend_from_slice(&code_ids[i].to_le_bytes());
        }
        let ro_compressed =
            zstd::encode_all(&ro_bytes[..], ZSTD_LEVEL).map_err(|e| format!("行序段压缩失败: {e}"))?;
        let row_order_offset = project_start + header_size as u64;
        let row_order_csz = ro_compressed.len() as u64;
        self.colblk_file
            .write_all(&ro_compressed)
            .map_err(|e| format!("写共享行序段失败: {e}"))?;

        // ---- 步骤⑤：rayon 窗口化并行压缩每因子 value 列，顺序追加写 ----
        // 窗口=256 限制同时驻留的压缩段内存（~256 × total_rows×4 压缩后）。
        let fv = std::sync::Arc::new(factor_values);
        let mut val_offsets: Vec<(u64, u64)> = Vec::with_capacity(factor_count);
        let mut cur = row_order_offset + row_order_csz;
        let window = 256usize;
        let mut f0 = 0usize;
        while f0 < factor_count {
            let f1 = (f0 + window).min(factor_count);
            let compressed: Vec<Vec<u8>> = (f0..f1)
                .into_par_iter()
                .map(|fi| -> Result<Vec<u8>, String> {
                    let vals = &fv[fi];
                    let mut seg = Vec::with_capacity(total_rows * F32_BYTES);
                    for &v in vals.iter() {
                        seg.extend_from_slice(&v.to_le_bytes());
                    }
                    zstd::encode_all(&seg[..], ZSTD_LEVEL)
                        .map_err(|e| format!("因子 {fi} value 段压缩失败: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for c in compressed {
                let csz = c.len() as u64;
                self.colblk_file
                    .write_all(&c)
                    .map_err(|e| format!("写 value 段失败: {e}"))?;
                val_offsets.push((cur, csz));
                cur += csz;
            }
            f0 = f1;
            if f0 % 5120 == 0 {
                eprintln!("🏗️ 投影进度: {}/{} 因子", f0, factor_count);
            }
        }
        self.colblk_file
            .flush()
            .map_err(|e| format!("flush 投影区失败: {e}"))?;
        drop(fv); // 释放转置大数组

        // ---- 步骤⑥：回填投影头 ----
        let mut hdr = Vec::with_capacity(header_size);
        hdr.extend_from_slice(&PROJ_FORMAT_VERSION.to_le_bytes());
        hdr.extend_from_slice(&(factor_count as u32).to_le_bytes());
        hdr.extend_from_slice(&(total_rows as u64).to_le_bytes());
        hdr.extend_from_slice(&row_order_offset.to_le_bytes());
        hdr.extend_from_slice(&row_order_csz.to_le_bytes());
        for (off, csz) in &val_offsets {
            hdr.extend_from_slice(&off.to_le_bytes());
            hdr.extend_from_slice(&csz.to_le_bytes());
        }
        self.colblk_file
            .seek(SeekFrom::Start(project_start))
            .map_err(|e| format!("seek 回填投影头失败: {e}"))?;
        self.colblk_file
            .write_all(&hdr)
            .map_err(|e| format!("回填投影头失败: {e}"))?;

        // ---- 步骤⑦：更新 colblk header（projected_offset + proj_format_version=2）+ idx ----
        self.projected_offset = project_start;
        self.proj_format_version = PROJ_FORMAT_VERSION;
        write_colblk_header_fields(
            &mut self.colblk_file,
            self.record_count,
            self.factor_count as u32,
            self.chunk_count,
            self.dict.dates.len() as u32,
            self.dict.codes.len() as u32,
            self.projected_offset,
            self.proj_format_version,
        )?;
        self.dict.write_idx(&self.idx_path, 1)?;
        eprintln!("✅ 投影完成（格式 v{}）", PROJ_FORMAT_VERSION);
        Ok(())
    }

    /// 返回已写入记录数
    pub fn record_count(&self) -> u64 {
        self.record_count
    }

    /// 是否已投影
    pub fn is_projected(&self) -> bool {
        self.projected_offset > 0
    }

    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }
}

// ============================ Reader ============================

/// 单个存储目录（单分片或扁平结构）的底层读取器：一个 colblk + 一个 idx。
/// 持有 mmap、header、字典、投影区索引。所有读路径都在此实现。
/// 投影区元信息（proj_format_version=2 新格式）。
struct ProjMeta {
    row_order_offset: u64,
    row_order_csz: u64,
    /// 每因子 value 段 (offset, compressed_size)
    val_index: Vec<(u64, u64)>,
}

struct SingleStoreReader {
    mmap: Mmap,
    hdr: ColblkHeader,
    dict: FactorDict,
    /// 投影区元信息（若有）
    proj_index: Option<ProjMeta>,
    /// 共享行序缓存（首次读投影时填充，全因子复用）
    row_order: std::sync::OnceLock<(Vec<u32>, Vec<u32>)>,
}

impl SingleStoreReader {
    /// 打开单个存储目录（必须含 factors.colblk + factors.idx）。
    fn open_dir(store_dir: &Path) -> Result<Self, String> {
        let colblk_path = store_dir.join("factors.colblk");
        let idx_path = store_dir.join("factors.idx");
        if !colblk_path.exists() || !idx_path.exists() {
            return Err(format!("存储目录不完整: 缺少 colblk 或 idx: {store_dir:?}"));
        }
        let file = File::open(&colblk_path).map_err(|e| format!("打开 colblk 失败: {e}"))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap colblk 失败: {e}"))? };
        let hdr = parse_colblk_header(&mmap[..COLBLK_HEADER_SIZE])?;
        let (dict, _flag) = FactorDict::read_idx(&idx_path)?;

        // 解析投影索引（若 projected_offset > 0）。新格式 proj_format_version=2。
        let proj_index = if hdr.projected_offset > 0
            && (hdr.projected_offset as usize) + 32 <= mmap.len()
        {
            if hdr.proj_format_version != PROJ_FORMAT_VERSION {
                return Err(format!(
                    "投影区为旧格式 proj_format_version={}（期望 {}），请用新代码重新计算/投影此 store",
                    hdr.proj_format_version, PROJ_FORMAT_VERSION
                ));
            }
            let base = hdr.projected_offset as usize;
            let fc = u32::from_le_bytes(mmap[base + 4..base + 8].try_into().unwrap()) as usize;
            let row_order_offset =
                u64::from_le_bytes(mmap[base + 16..base + 24].try_into().unwrap());
            let row_order_csz =
                u64::from_le_bytes(mmap[base + 24..base + 32].try_into().unwrap());
            let idx_start = base + 32;
            let mut val_index = Vec::with_capacity(fc);
            for f in 0..fc {
                let off_pos = idx_start + f * 16;
                if off_pos + 16 > mmap.len() {
                    break;
                }
                let off = u64::from_le_bytes(mmap[off_pos..off_pos + 8].try_into().unwrap());
                let csz = u64::from_le_bytes(mmap[off_pos + 8..off_pos + 16].try_into().unwrap());
                val_index.push((off, csz));
            }
            Some(ProjMeta {
                row_order_offset,
                row_order_csz,
                val_index,
            })
        } else {
            None
        };

        Ok(Self {
            mmap,
            hdr,
            dict,
            proj_index,
            row_order: std::sync::OnceLock::new(),
        })
    }

    /// 返回已投影状态
    fn is_projected(&self) -> bool {
        self.hdr.projected_offset > 0 && self.proj_index.is_some()
    }

    /// 解压共享行序段 → (date_ids, code_ids)。由 OnceLock 首次访问时调用，全因子复用。
    fn load_row_order(&self, proj: &ProjMeta) -> (Vec<u32>, Vec<u32>) {
        let s = proj.row_order_offset as usize;
        let e = s + proj.row_order_csz as usize;
        let dec = zstd::decode_all(&self.mmap[s..e]).expect("解压共享行序段失败");
        let n = dec.len() / 8;
        let mut d = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        for i in 0..n {
            d.push(u32::from_le_bytes(dec[i * 8..i * 8 + 4].try_into().unwrap()));
            c.push(u32::from_le_bytes(dec[i * 8 + 4..i * 8 + 8].try_into().unwrap()));
        }
        (d, c)
    }

    /// 把第 col_idx 个因子的有效单元格填进 output（NaN 位不动，已有非 NaN 值不覆盖）。
    /// 分片间按 date 互斥，同一单元格不会被两个分片写入，可安全跨分片累加。
    /// 优先走投影区（单次顺序读）；无投影则回退跨 chunk 扫描。
    fn read_factor_into(
        &self,
        col_idx: usize,
        date_pos: &HashMap<i64, usize>,
        stock_pos: &HashMap<String, usize>,
        output: &mut ndarray::Array2<f32>,
    ) {
        // ---- 优先投影区（新格式：缓存行序 + 解压 value 段）----
        if let Some(proj) = &self.proj_index {
            if col_idx < proj.val_index.len() {
                let (date_ids, code_ids) = self
                    .row_order
                    .get_or_init(|| self.load_row_order(proj));
                let (off, csz) = proj.val_index[col_idx];
                let start = off as usize;
                let end = start + csz as usize;
                if end <= self.mmap.len() {
                    let vals = match zstd::decode_all(&self.mmap[start..end]) {
                        Ok(v) => v,
                        Err(_) => return,
                    };
                    let n_rows = vals.len() / F32_BYTES;
                    for r in 0..n_rows {
                        let v = f32::from_le_bytes(vals[r * 4..r * 4 + 4].try_into().unwrap());
                        let d_id = date_ids[r] as usize;
                        let c_id = code_ids[r] as usize;
                        let date = self.dict.dates.get(d_id).copied();
                        let code = self.dict.codes.get(c_id);
                        if let (Some(date), Some(code)) = (date, code) {
                            if let Some(&dp) = date_pos.get(&date) {
                                if let Some(&sp) = stock_pos.get(code) {
                                    output[[dp, sp]] = if v.is_finite() { v } else { f32::NAN };
                                }
                            }
                        }
                    }
                    return;
                }
            }
        }

        // ---- 回退：跨 chunk 扫描（无投影区）----
        // 扫描所有 chunk，每 chunk 内因子 col_idx 的值在 fac_base + col_idx*n*4 + i*4
        let mut offset = COLBLK_HEADER_SIZE as u64;
        let scan_end = if self.hdr.projected_offset > 0 {
            self.hdr.projected_offset
        } else {
            self.mmap.len() as u64
        };
        while offset + 8 <= scan_end {
            let s = offset as usize;
            let compressed_size =
                u32::from_le_bytes(self.mmap[s..s + 4].try_into().unwrap()) as usize;
            let n_in_batch =
                u32::from_le_bytes(self.mmap[s + 4..s + 8].try_into().unwrap()) as usize;
            let data_start = s + 8;
            let body_row_size = ID_BYTES * 2 + self.hdr.factor_count * F32_BYTES;
            let data_end = if compressed_size == 0 {
                data_start + n_in_batch * body_row_size
            } else {
                data_start + compressed_size
            };
            if data_end > self.mmap.len() {
                break;
            }
            let decompressed = if compressed_size == 0 {
                self.mmap[data_start..data_end].to_vec()
            } else {
                match zstd::decode_all(&self.mmap[data_start..data_end]) {
                    Ok(d) => d,
                    Err(_) => break,
                }
            };
            let n = n_in_batch;
            let id_w = ID_BYTES;
            let fac_base = n * id_w * 2;
            for i in 0..n {
                let d_id =
                    u32::from_le_bytes(decompressed[i * id_w..i * id_w + 4].try_into().unwrap())
                        as usize;
                let c_id = u32::from_le_bytes(
                    decompressed[n * id_w + i * id_w..n * id_w + i * id_w + 4]
                        .try_into()
                        .unwrap(),
                ) as usize;
                let off = fac_base + (col_idx * n + i) * F32_BYTES;
                if off + F32_BYTES > decompressed.len() {
                    break;
                }
                let v = f32::from_le_bytes(decompressed[off..off + F32_BYTES].try_into().unwrap());
                let date = self.dict.dates.get(d_id).copied();
                let code = self.dict.codes.get(c_id);
                if let (Some(date), Some(code)) = (date, code) {
                    if let Some(&dp) = date_pos.get(&date) {
                        if let Some(&sp) = stock_pos.get(code) {
                            output[[dp, sp]] = if v.is_finite() { v } else { f32::NAN };
                        }
                    }
                }
            }
            offset = data_end as u64;
        }
    }
}

/// 列式因子存储读取器（回测侧）。
///
/// 支持两种布局：
/// - **扁平**：`store_dir/{factors.colblk, factors.idx}`
/// - **分片**：`store_dir/shard_*/{factors.colblk, factors.idx}`。
///   `run_factor_pipeline` 默认按 `n_shards=8` 写分片，对应 8 盘 LVM 条带并行，
///   按 `date % n_shards` 路由——分片间 date 互斥，同一单元格不会落在两个分片。
///
/// 回测侧统一入口：先 `open`，再对每个原始因子调 `read_factor_to_matrix`，
/// 得到 `Array2<f32>`（n_dates × n_stocks）喂给 tail 的 process_task。
/// 分片模式下自动跨分片合并：所有分片共享相同的 factor_names / col_idx 语义。
pub struct FactorStoreReader {
    store_dir: PathBuf,
    /// 所有分片（扁平存储就是单个分片）
    stores: Vec<SingleStoreReader>,
}

impl FactorStoreReader {
    /// 打开存储。自动识别扁平 vs 分片布局：
    /// - 存在 `shard_0/factors.idx` → 分片模式，收集全部 `shard_i`
    /// - 否则 → 扁平模式，整目录当单分片读
    pub fn open(store_dir: &str) -> Result<Self, String> {
        let root = PathBuf::from(store_dir);
        // 分片布局：检测 shard_0 子目录（写入端固定从 shard_0 起）
        if root.join("shard_0").join("factors.idx").exists() {
            let mut shard_dirs: Vec<(usize, PathBuf)> = Vec::new();
            let entries = std::fs::read_dir(&root).map_err(|e| format!("读取存储目录失败: {e}"))?;
            for entry in entries {
                let entry = entry.map_err(|e| format!("遍历目录项失败: {e}"))?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(num) = name.strip_prefix("shard_") {
                    if let Ok(i) = num.parse::<usize>() {
                        let p = entry.path();
                        if p.join("factors.colblk").exists() && p.join("factors.idx").exists() {
                            shard_dirs.push((i, p));
                        }
                    }
                }
            }
            if shard_dirs.is_empty() {
                return Err(format!("存储目录无有效分片: {store_dir:?}"));
            }
            // 按 shard 编号升序，保证跨分片读取顺序稳定
            shard_dirs.sort_by_key(|(i, _)| *i);
            let mut stores = Vec::with_capacity(shard_dirs.len());
            for (_, p) in shard_dirs {
                stores.push(SingleStoreReader::open_dir(&p)?);
            }
            return Ok(Self {
                store_dir: root,
                stores,
            });
        }
        // 扁平布局
        let store = SingleStoreReader::open_dir(&root)?;
        Ok(Self {
            store_dir: root,
            stores: vec![store],
        })
    }

    /// 返回因子名列表（所有分片一致，取第 0 个）
    pub fn factor_names(&self) -> &[String] {
        &self.stores[0].dict.factor_names
    }

    /// 返回已投影状态（所有分片都已投影才算完成）
    pub fn is_projected(&self) -> bool {
        !self.stores.is_empty() && self.stores.iter().all(|s| s.is_projected())
    }

    /// 读取第 col_idx 个因子，pivot 成 (n_dates × n_stocks) 矩阵。
    /// template_dates / template_stocks 决定输出矩阵的行列轴。
    /// 分片模式下跨分片合并：初始化为 NaN，逐分片填有效单元格
    /// （分片间 date 互斥，同一单元格不会被两个分片写入）。
    pub fn read_factor_to_matrix(
        &self,
        col_idx: usize,
        template_dates: &[i32],
        template_stocks: &[String],
    ) -> Result<ndarray::Array2<f32>, String> {
        let mut output = ndarray::Array2::<f32>::from_elem(
            (template_dates.len(), template_stocks.len()),
            f32::NAN,
        );

        // 构建 date / code → 模板位置映射
        let date_pos: HashMap<i64, usize> = template_dates
            .iter()
            .enumerate()
            .map(|(i, &d)| (d as i64, i))
            .collect();
        // template_stocks 可能带 .SH/.SZ 后缀，字典里的是裸代码
        let stock_pos: HashMap<String, usize> = template_stocks
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let bare = s.split('.').next().unwrap_or(s).to_string();
                (bare, i)
            })
            .collect();

        for store in &self.stores {
            store.read_factor_into(col_idx, &date_pos, &stock_pos, &mut output);
        }
        Ok(output)
    }

    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }

    /// 返回记录总数（所有分片 record_count 之和）
    pub fn record_count(&self) -> u64 {
        self.stores.iter().map(|s| s.hdr.record_count).sum()
    }

    /// 返回模板轴：跨分片去重排序后的 dates（i32）和 stocks（裸代码，无交易所后缀）。
    /// 供 Python 侧推断回测模板，替代 v4 的 pandas 读 parquet。
    pub fn template_axes(&self) -> (Vec<i32>, Vec<String>) {
        let mut dates: Vec<i32> = Vec::new();
        let mut stocks: Vec<String> = Vec::new();
        for store in &self.stores {
            dates.extend(store.dict.dates.iter().map(|&d| d as i32));
            stocks.extend(store.dict.codes.iter().cloned());
        }
        dates.sort_unstable();
        dates.dedup();
        stocks.sort();
        stocks.dedup();
        (dates, stocks)
    }
}

// ============================ PyO3 导出 ============================

/// 创建/打开列式因子存储写入器（Python 侧主要供测试用；正式流程由 run_factor_pipeline 内部调用）
#[pyfunction]
#[pyo3(signature = (store_dir, factor_names))]
pub fn factor_store_v5_open(store_dir: String, factor_names: Vec<String>) -> PyResult<()> {
    let mut writer = FactorStoreWriter::open(&store_dir, &factor_names).map_err(pyerr)?;
    // 仅打开，不写数据。返回前 flush 确保文件落盘
    writer.dict.write_idx(&writer.idx_path, 0).map_err(pyerr)?;
    Ok(())
}

/// 查询列式存储的元信息（因子名、记录数、是否已投影）
#[pyfunction]
pub fn factor_store_v5_info(store_dir: String) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    Python::with_gil(|py| {
        let reader = FactorStoreReader::open(&store_dir).map_err(pyerr)?;
        let dict = PyDict::new(py);
        dict.set_item("record_count", reader.record_count())?;
        dict.set_item("factor_count", reader.factor_names().len())?;
        dict.set_item("is_projected", reader.is_projected())?;
        dict.set_item("factor_names", reader.factor_names().to_vec())?;
        Ok(dict.into())
    })
}

/// 把压缩 chunk 的 colblk 转换为不压缩格式（就地转换，原地重写文件）。
/// 压缩比仅 1.15× 但解压 55min/遍，不压缩后 Reader 可直接 mmap 随机读。
/// 转换过程：遍历所有压缩 chunk → 解压 → 重新写为 compressed_size=0 的不压缩 chunk。
#[pyfunction]
pub fn factor_store_v5_decompress_inplace(store_dir: String) -> PyResult<()> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let colblk_path = std::path::Path::new(&store_dir).join("factors.colblk");
    let tmp_path = std::path::Path::new(&store_dir).join("factors.colblk.new");

    // 读 header
    let mut src = File::open(&colblk_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut hdr = [0u8; 64];
    src.read_exact(&mut hdr)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let header = parse_colblk_header(&hdr).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

    // 创建新文件，复制 header（保持 record_count/factor_count 等，chunk_count 不变）
    let mut dst = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    dst.write_all(&hdr)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    // 遍历所有 chunk，解压后写为不压缩
    let mut offset = 64u64;
    let scan_end = if header.projected_offset > 0 {
        header.projected_offset
    } else {
        std::fs::metadata(&colblk_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
            .len()
    };
    let mut processed = 0u64;
    while offset + 8 <= scan_end {
        src.seek(SeekFrom::Start(offset))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut chunk_hdr = [0u8; 8];
        if src.read_exact(&mut chunk_hdr).is_err() {
            break;
        }
        let csz = u32::from_le_bytes(chunk_hdr[0..4].try_into().unwrap());
        let nib = u32::from_le_bytes(chunk_hdr[4..8].try_into().unwrap());

        if csz == 0 {
            // 已不压缩，直接复制
            let body_size = nib as usize * (36 + header.factor_count * 4);
            let mut body = vec![0u8; body_size];
            src.read_exact(&mut body)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            dst.write_all(&0u32.to_le_bytes())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            dst.write_all(&nib.to_le_bytes())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            dst.write_all(&body)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            offset += 8 + body_size as u64;
        } else {
            // 压缩，解压后写不压缩
            let mut cdata = vec![0u8; csz as usize];
            src.read_exact(&mut cdata)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let body = zstd::decode_all(&cdata[..])
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("解压失败: {e}")))?;
            dst.write_all(&0u32.to_le_bytes())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            dst.write_all(&nib.to_le_bytes())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            dst.write_all(&body)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            offset += 8 + csz as u64;
        }
        processed += 1;
        if processed % 10000 == 0 {
            println!("已转换 {} chunks...", processed);
        }
    }
    dst.flush()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    drop(dst);
    drop(src);

    // 替换原文件
    let backup = colblk_path.with_extension("bak");
    std::fs::rename(&colblk_path, &backup)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    std::fs::rename(&tmp_path, &colblk_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let _ = std::fs::remove_file(&backup);
    println!("✅ 转换完成，{} chunks 已转为不压缩格式", processed);
    Ok(())
}

/// 对已有 colblk 存储触发投影（不重新计算）。
/// 用于计算已完成但投影失败/中断的场景。流式投影，内存可控。
#[pyfunction]
#[pyo3(signature = (store_dir, n_jobs=0))]
pub fn factor_store_v5_project_only(store_dir: String, n_jobs: usize) -> PyResult<()> {
    let info_path = std::path::Path::new(&store_dir).join("factors.idx");
    // 先读 idx 获取 factor_names（Writer::open 需要校验因子数）
    let (dict, _) = FactorDict::read_idx(&info_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("读取 idx 失败: {}", e)))?;
    let mut writer = FactorStoreWriter::open(&store_dir, &dict.factor_names)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("打开存储失败: {}", e)))?;
    if writer.is_projected() {
        return Ok(());
    }
    writer
        .finish_and_project(n_jobs)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("投影失败: {}", e)))?;
    Ok(())
}

/// 读取第 col_idx 个因子为扁平数组（date_id[], code_id[], factor[]），供测试/调试
#[pyfunction]
pub fn factor_store_v5_read_factor(store_dir: String, col_idx: usize) -> PyResult<PyObject> {
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    Python::with_gil(|py| {
        let reader = FactorStoreReader::open(&store_dir).map_err(pyerr)?;
        if col_idx >= reader.factor_names().len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "col_idx {col_idx} 超出范围（共 {} 个因子）",
                reader.factor_names().len()
            )));
        }
        // 用空模板走投影区读取原始三元组
        // 这里直接读投影段得到 (date_id, code_id, value)
        // 支持未投影：走 read_factor_to_matrix 的回退路径
        let tmpl = reader.template_axes();
        let dates_i32: Vec<i32> = tmpl.0;
        let stocks: Vec<String> = tmpl.1;
        // 用 read_factor_to_matrix 读取（支持投影和未投影回退）
        let matrix = reader
            .read_factor_to_matrix(col_idx, &dates_i32, &stocks)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        // 展平矩阵为 (date_id, code_id, factor) 三元组
        let mut dates = Vec::new();
        let mut codes = Vec::new();
        let mut factors = Vec::new();
        for (di, _) in dates_i32.iter().enumerate() {
            for (ci, _) in stocks.iter().enumerate() {
                let v = matrix[[di, ci]];
                if v.is_finite() {
                    dates.push(di as u32);
                    codes.push(ci as u32);
                    factors.push(v);
                }
            }
        }
        let dict = PyDict::new(py);
        dict.set_item("date_id", PyArray1::from_vec(py, dates))?;
        dict.set_item("code_id", PyArray1::from_vec(py, codes))?;
        dict.set_item("factor", PyArray1::from_vec(py, factors))?;
        Ok(dict.into())
    })
}

/// 从 colblk 存储推断回测模板轴：返回去重排序的 dates（int 数组）和 stocks（带 .SZ/.SH 后缀）。
/// 供 design_whatever 的 tail_v5 替代 v4 的 pandas 读 parquet 推断模板。
#[pyfunction]
pub fn factor_store_v5_template(store_dir: String) -> PyResult<PyObject> {
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    Python::with_gil(|py| {
        let reader = FactorStoreReader::open(&store_dir).map_err(pyerr)?;
        let (dates, stocks_bare) = reader.template_axes();
        // 给股票代码加交易所后缀（与 v4 parquet 导出的 add_exchange_suffix 一致）
        let stocks: Vec<String> = stocks_bare.iter().map(|c| add_exchange_suffix(c)).collect();
        let dict = PyDict::new(py);
        dict.set_item("dates", PyArray1::from_vec(py, dates))?;
        dict.set_item("stocks", stocks)?;
        Ok(dict.into())
    })
}

/// 给股票代码加交易所后缀（6/9 开头→.SH，0/3 开头→.SZ，8/4 开头→.BJ）。
/// 与 backup_column_cache::add_exchange_suffix 保持一致。
fn add_exchange_suffix(code: &str) -> String {
    if code.is_empty() {
        return code.to_string();
    }
    match code.chars().next() {
        Some('6') | Some('9') => format!("{code}.SH"),
        Some('8') | Some('4') => format!("{code}.BJ"),
        _ => format!("{code}.SZ"),
    }
}

/// 把 colblk 存储中指定因子（按因子名）批量导出为 parquet 文件。
/// 每个因子一个 parquet，schema = date(Int64) + 每个股票一列(Float64)，
/// 与现有 export_backup_to_parquet_rust 输出格式完全一致，保证 postprocess 兼容。
///
/// 供 design_whatever tail_v5 在候选筛选后导出入选因子给 fulltest/materialize。
#[pyfunction]
#[pyo3(signature = (store_dir, output_dir, factor_names, n_jobs=0))]
pub fn factor_store_v5_export_factors_parquet(
    store_dir: String,
    output_dir: String,
    factor_names: Vec<String>,
    n_jobs: usize,
) -> PyResult<usize> {
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use rayon::prelude::*;
    use std::sync::Arc as ArrowArc;

    // 控制 rayon 线程数
    if n_jobs > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs)
            .build_global();
    }

    let reader = FactorStoreReader::open(&store_dir).map_err(pyerr)?;

    // 建立因子名 → col_idx 映射
    let name_to_idx: std::collections::HashMap<&str, usize> = reader
        .factor_names()
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    // 模板轴（dates/stocks），用带后缀的股票代码作为列名
    let (dates, stocks_bare) = reader.template_axes();
    let dates_i64: Vec<i64> = dates.iter().map(|&d| d as i64).collect();
    let stock_cols: Vec<String> = stocks_bare.iter().map(|c| add_exchange_suffix(c)).collect();

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("创建输出目录失败: {e}")))?;

    // 并行导出每个因子
    let total = factor_names.len();
    let names_arc = ArrowArc::new(factor_names);
    let reader_data: Vec<u8> = Vec::new(); // 占位，实际通过 mmap 读
    let _ = reader_data;

    // 收集 (name, col_idx) 对
    let targets: Vec<(String, usize)> = names_arc
        .iter()
        .filter_map(|name| {
            name_to_idx
                .get(name.as_str())
                .map(|&idx| (name.clone(), idx))
        })
        .collect();

    let exported = targets
        .par_iter()
        .map(|(name, col_idx)| -> Result<bool, String> {
            // 读因子矩阵（投影区顺序读）
            let matrix =
                read_factor_matrix_for_export(&store_dir, *col_idx, &dates_i64, &stocks_bare)?;
            // 构造 arrow schema + RecordBatch
            let mut fields = vec![Field::new("date", DataType::Int64, false)];
            for col_name in &stock_cols {
                fields.push(Field::new(col_name, DataType::Float64, true));
            }
            let schema = ArrowArc::new(Schema::new(fields));
            let n_rows = dates_i64.len();
            let date_arr = Int64Array::from(dates_i64.clone());
            let mut columns: Vec<ArrowArc<dyn arrow::array::Array>> = vec![ArrowArc::new(date_arr)];
            for s_idx in 0..stock_cols.len() {
                let col_data: Vec<f64> = (0..n_rows)
                    .map(|r| {
                        let v = matrix[[r, s_idx]];
                        if v.is_finite() {
                            v as f64
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                columns.push(ArrowArc::new(Float64Array::from(col_data)));
            }
            let batch = RecordBatch::try_new(schema.clone(), columns)
                .map_err(|e| format!("构造 RecordBatch 失败: {e}"))?;
            // 写 parquet
            let out_path = std::path::Path::new(&output_dir).join(format!("{name}.parquet"));
            let file = std::fs::File::create(&out_path)
                .map_err(|e| format!("创建 parquet 失败 {out_path:?}: {e}"))?;
            let mut writer = ArrowWriter::try_new(file, schema.clone(), None)
                .map_err(|e| format!("创建 ArrowWriter 失败: {e}"))?;
            writer
                .write(&batch)
                .map_err(|e| format!("写 parquet 失败: {e}"))?;
            writer
                .close()
                .map_err(|e| format!("关闭 parquet 失败: {e}"))?;
            Ok(true)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(pyerr)?;

    Ok(exported.len())
}

/// 导出用的矩阵读取（重新打开 Reader，避免跨线程共享 mmap 问题）
fn read_factor_matrix_for_export(
    store_dir: &str,
    col_idx: usize,
    dates: &[i64],
    stocks_bare: &[String],
) -> Result<ndarray::Array2<f32>, String> {
    let reader = FactorStoreReader::open(store_dir)?;
    let dates_i32: Vec<i32> = dates.iter().map(|&d| d as i32).collect();
    reader.read_factor_to_matrix(col_idx, &dates_i32, stocks_bare)
}

// ============================ BackupSink：统一写入封装 ============================
// 让 factor_pipeline 的 collector 线程用同一接口写"旧 bin"或"新 colblk"，
// 避免改动 collector 的线程结构。线程安全：colblk 分支用 Mutex 包裹（内部有 File）。

use std::sync::{Arc, Mutex};

/// 统一备份写入后端。
/// - `Bin`：旧 RPBACKUP v4 行式格式（`save_results_to_backup`）
/// - `Colblk`：新 RPFBINV5 列式格式（`FactorStoreWriter`）
#[derive(Clone)]
pub enum BackupSink {
    Bin {
        backup_file: String,
        expected_len: usize,
    },
    Colblk {
        writer: Arc<Mutex<FactorStoreWriter>>,
    },
}

impl BackupSink {
    /// 创建旧 bin 后端
    pub fn new_bin(backup_file: String, expected_len: usize) -> Self {
        BackupSink::Bin {
            backup_file,
            expected_len,
        }
    }

    /// 创建新 colblk 后端（打开或断点续算）
    pub fn new_colblk(store_dir: &str, factor_names: &[String]) -> Result<Self, String> {
        let writer = FactorStoreWriter::open(store_dir, factor_names)?;
        Ok(BackupSink::Colblk {
            writer: Arc::new(Mutex::new(writer)),
        })
    }

    /// 追加一批结果。collector 线程调用，线程安全。
    pub fn append_batch(&self, results: &[TaskResult]) -> Result<(), String> {
        match self {
            BackupSink::Bin {
                backup_file,
                expected_len,
            } => crate::backup_writer::save_results_to_backup(results, backup_file, *expected_len)
                .map_err(|e| e.to_string()),
            BackupSink::Colblk { writer } => {
                let mut w = writer.lock().map_err(|e| format!("锁 writer 失败: {e}"))?;
                w.append_batch(results)
            }
        }
    }

    /// 克隆（仅 Colblk 分支增加 Arc 引用计数，开销极低；Bin 分支 clone 字符串）
    pub fn clone_handle(&self) -> Self {
        match self {
            BackupSink::Bin {
                backup_file,
                expected_len,
            } => BackupSink::Bin {
                backup_file: backup_file.clone(),
                expected_len: *expected_len,
            },
            BackupSink::Colblk { writer } => BackupSink::Colblk {
                writer: writer.clone(),
            },
        }
    }

    /// 计算结束后收尾：
    /// - Bin：无操作（旧流程靠后续 export_backup_to_parquet_py）
    /// - Colblk：执行投影 finish_and_project(n_jobs)
    pub fn finish_and_project(&self, n_jobs: usize) -> Result<(), String> {
        match self {
            BackupSink::Bin { .. } => Ok(()),
            BackupSink::Colblk { writer } => {
                let mut w = writer.lock().map_err(|e| format!("锁 writer 失败: {e}"))?;
                w.finish_and_project(n_jobs)
            }
        }
    }

    /// 返回已写入记录的 (date, code) 集合（断点续算过滤用）
    pub fn check_completed(&self) -> Result<HashSet<(i64, String)>, String> {
        match self {
            BackupSink::Bin {
                backup_file,
                expected_len: _,
            } => {
                // 复用旧逻辑：通过 read_existing_backup_with_filter
                let dates: Option<std::collections::HashSet<i64>> = None;
                let _ = dates;
                // 旧 bin 的断点续算：读全部记录的 (date, code)
                crate::backup_reader::read_existing_backup_with_filter(backup_file, None)
                    .map_err(|e| e.to_string())
            }
            BackupSink::Colblk { writer } => {
                let w = writer.lock().map_err(|e| format!("锁 writer 失败: {e}"))?;
                w.check_completed()
            }
        }
    }

    /// 是否为 colblk 后端
    pub fn is_colblk(&self) -> bool {
        matches!(self, BackupSink::Colblk { .. })
    }

    /// 返回存储目录（colblk 返回 store_dir 字符串，bin 返回 None）
    pub fn store_dir(&self) -> Option<String> {
        match self {
            BackupSink::Colblk { writer } => {
                let w = writer.lock().ok()?;
                Some(w.store_dir().to_string_lossy().to_string())
            }
            BackupSink::Bin { .. } => None,
        }
    }
}

// ============================ ShardedBackupSink：多分片并行写 ============================

/// 多分片备份写入后端。N 个独立 Writer 分片，按 date % N 路由，无锁并行写。
/// 解决单 collector 写盘瓶颈：200 worker 生产速度远超单流 HDD 写盘。
#[derive(Clone)]
pub struct ShardedBackupSink {
    shards: Vec<BackupSink>,
    n_shards: usize,
}

impl ShardedBackupSink {
    pub fn new_colblk_sharded(
        store_dir: &str,
        factor_names: &[String],
        n_shards: usize,
    ) -> Result<Self, String> {
        let mut shards = Vec::with_capacity(n_shards);
        for i in 0..n_shards {
            let shard_dir = format!("{store_dir}/shard_{i}");
            shards.push(BackupSink::new_colblk(&shard_dir, factor_names)?);
        }
        Ok(ShardedBackupSink { shards, n_shards })
    }

    /// 按 date % n_shards 路由 + rayon 并行写各 shard（不同 shard 的 Mutex 不竞争）。
    pub fn append_batch(&self, results: &[TaskResult]) -> Result<(), String> {
        let mut buckets: Vec<Vec<TaskResult>> = (0..self.n_shards).map(|_| Vec::new()).collect();
        for r in results {
            let idx = (r.date as usize) % self.n_shards;
            buckets[idx].push(r.clone());
        }
        // rayon 并行写各 shard（每 shard 独立 Mutex，无竞争）
        let results: Vec<Result<(), String>> = buckets
            .into_par_iter()
            .enumerate()
            .filter(|(_, b)| !b.is_empty())
            .map(|(i, batch)| self.shards[i].append_batch(&batch))
            .collect();
        for r in results {
            r?;
        }
        Ok(())
    }

    pub fn clone_handle(&self) -> Self {
        ShardedBackupSink {
            shards: self.shards.iter().map(|s| s.clone_handle()).collect(),
            n_shards: self.n_shards,
        }
    }

    pub fn finish_and_project(&self, n_jobs: usize) -> Result<(), String> {
        let n_shards = self.shards.len();
        for (i, shard) in self.shards.iter().enumerate() {
            eprintln!("🏗️ 投影 shard {}/{}", i + 1, n_shards);
            shard.finish_and_project(n_jobs)?;
        }
        Ok(())
    }

    pub fn check_completed(&self) -> Result<HashSet<(i64, String)>, String> {
        let mut all = HashSet::new();
        let n_shards = self.shards.len();
        for (i, shard) in self.shards.iter().enumerate() {
            eprintln!("📂 读取已有数据: 分片 {}/{}", i + 1, n_shards);
            all.extend(shard.check_completed()?);
        }
        eprintln!("📂 已读取 {} 条 (date, code) 记录，准备断点续算", all.len());
        Ok(all)
    }

    pub fn n_shards(&self) -> usize {
        self.n_shards
    }
}

fn pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyIOError::new_err(e.to_string())
}

// ============================ 单元测试 ============================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup_reader::TaskResult;
    use ndarray::Array2;

    /// 构造测试数据：dates × codes 个任务，每个任务 F 个因子值。
    /// 因子 f 在 (date_idx, code_idx) 的值 = date_idx * 1000 + code_idx * 10 + f（便于校验）
    fn make_test_results(dates: &[i64], codes: &[&str], factor_count: usize) -> Vec<TaskResult> {
        let mut results = Vec::new();
        for (di, &date) in dates.iter().enumerate() {
            for (ci, code) in codes.iter().enumerate() {
                let facs: Vec<f32> = (0..factor_count)
                    .map(|f| (di as f32) * 1000.0 + (ci as f32) * 10.0 + f as f32)
                    .collect();
                results.push(TaskResult {
                    date,
                    code: code.to_string(),
                    timestamp: 0,
                    facs,
                });
            }
        }
        results
    }

    #[test]
    fn test_write_project_read_consistency() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().to_str().unwrap().to_string();

        let dates: Vec<i64> = vec![20230101, 20230102, 20230103, 20230104, 20230105];
        let codes: Vec<&str> = vec!["000001", "600519", "000858"];
        let factor_count = 4;
        let factor_names: Vec<String> = (0..factor_count).map(|i| format!("f{i}")).collect();
        let results = make_test_results(&dates, &codes, factor_count);

        // 写入（分两批，模拟增量追加）
        let mut writer = FactorStoreWriter::open(&store_dir, &factor_names).unwrap();
        let mid = results.len() / 2;
        writer.append_batch(&results[..mid]).unwrap();
        writer.append_batch(&results[mid..]).unwrap();
        assert_eq!(writer.record_count(), results.len() as u64);

        // 投影
        writer.finish_and_project(2).unwrap();

        // 读取校验：每个因子的矩阵应正确
        let reader = FactorStoreReader::open(&store_dir).unwrap();
        assert!(reader.is_projected());
        assert_eq!(reader.factor_names(), factor_names);

        // template_dates 用 i32（tail 的约定），template_stocks 带后缀测试去后缀逻辑
        let template_dates: Vec<i32> = dates.iter().map(|&d| d as i32).collect();
        let template_stocks: Vec<String> = codes.iter().map(|c| format!("{c}.SZ")).collect();

        for f_idx in 0..factor_count {
            let matrix: Array2<f32> = reader
                .read_factor_to_matrix(f_idx, &template_dates, &template_stocks)
                .unwrap();
            assert_eq!(matrix.shape(), [dates.len(), codes.len()]);
            // 校验每个值（注意 f32 截断）
            for (di, _) in dates.iter().enumerate() {
                for (ci, _) in codes.iter().enumerate() {
                    let expected = (di as f32) * 1000.0 + (ci as f32) * 10.0 + f_idx as f32;
                    let actual = matrix[[di, ci]];
                    assert!(
                        (actual - expected).abs() < 0.01,
                        "因子 {f_idx} 在 (date_idx={di}, code_idx={ci}) 值不符: 期望 {expected}, 实际 {actual}"
                    );
                }
            }
        }
        println!("✅ test_write_project_read_consistency 通过");
    }

    #[test]
    fn test_checkpoint_resume() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().to_str().unwrap().to_string();

        let dates: Vec<i64> = vec![20230101, 20230102, 20230103];
        let codes: Vec<&str> = vec!["000001", "600519"];
        let factor_count = 2;
        let factor_names: Vec<String> = (0..factor_count).map(|i| format!("f{i}")).collect();
        let results = make_test_results(&dates, &codes, factor_count);

        // 第一批写入
        let mut writer = FactorStoreWriter::open(&store_dir, &factor_names).unwrap();
        writer.append_batch(&results[..3]).unwrap(); // 写 3 条
        let first_count = writer.record_count();
        assert_eq!(first_count, 3);
        drop(writer); // 模拟进程退出

        // 断点续算：重新打开，检查已完成记录
        let mut writer2 = FactorStoreWriter::open(&store_dir, &factor_names).unwrap();
        assert_eq!(writer2.record_count(), 3);
        let completed = writer2.check_completed().unwrap();
        // 前 3 条应标记为已完成
        assert!(completed.contains(&(dates[0], codes[0].to_string())));
        assert!(completed.contains(&(dates[1], codes[1].to_string())));
        // 第 4 条（dates[1], codes[1]）= index 3
        assert_eq!(completed.len(), 3);

        // 继续追加剩余
        writer2.append_batch(&results[3..]).unwrap();
        assert_eq!(writer2.record_count(), 6);
        writer2.finish_and_project(2).unwrap();

        // 最终读取校验
        let reader = FactorStoreReader::open(&store_dir).unwrap();
        let template_dates: Vec<i32> = dates.iter().map(|&d| d as i32).collect();
        let template_stocks: Vec<String> = codes.iter().map(|c| c.to_string()).collect();
        for f_idx in 0..factor_count {
            let matrix = reader
                .read_factor_to_matrix(f_idx, &template_dates, &template_stocks)
                .unwrap();
            for (di, _) in dates.iter().enumerate() {
                for (ci, _) in codes.iter().enumerate() {
                    let expected = (di as f32) * 1000.0 + (ci as f32) * 10.0 + f_idx as f32;
                    assert!((matrix[[di, ci]] - expected).abs() < 0.01);
                }
            }
        }
        println!("✅ test_checkpoint_resume 通过");
    }

    #[test]
    fn test_dictionary_growth_across_batches() {
        // 验证字典跨批次增长：不同批次引入新 date/code，字典 id 应稳定递增
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().to_str().unwrap().to_string();
        let factor_names = vec!["x".to_string()];

        let mut writer = FactorStoreWriter::open(&store_dir, &factor_names).unwrap();
        // 批1：2 个日期 2 个股票
        writer
            .append_batch(&[
                TaskResult {
                    date: 20230101,
                    code: "000001".into(),
                    timestamp: 0,
                    facs: vec![1.0],
                },
                TaskResult {
                    date: 20230101,
                    code: "600519".into(),
                    timestamp: 0,
                    facs: vec![2.0],
                },
            ])
            .unwrap();
        // 批2：新增 1 个日期 1 个股票
        writer
            .append_batch(&[TaskResult {
                date: 20230102,
                code: "000858".into(),
                timestamp: 0,
                facs: vec![3.0],
            }])
            .unwrap();
        writer.finish_and_project(1).unwrap();

        let reader = FactorStoreReader::open(&store_dir).unwrap();
        // 字典应有 2 日期 + 3 股票
        assert_eq!(reader.stores[0].dict.dates.len(), 2);
        assert_eq!(reader.stores[0].dict.codes.len(), 3);

        // 读取唯一因子，校验 3 个值都正确
        let template_dates: Vec<i32> = vec![20230101, 20230102];
        let template_stocks: Vec<String> = vec![
            "000001".to_string(),
            "600519".to_string(),
            "000858".to_string(),
        ];
        let m = reader
            .read_factor_to_matrix(0, &template_dates, &template_stocks)
            .unwrap();
        assert!((m[[0, 0]] - 1.0).abs() < 0.01); // (20230101, 000001) = 1.0
        assert!((m[[0, 1]] - 2.0).abs() < 0.01); // (20230101, 600519) = 2.0
        assert!((m[[1, 2]] - 3.0).abs() < 0.01); // (20230102, 000858) = 3.0
        assert!(m[[0, 2]].is_nan()); // (20230101, 000858) 不存在 = NaN
        assert!(m[[1, 0]].is_nan()); // (20230102, 000001) 不存在 = NaN
        println!("✅ test_dictionary_growth_across_batches 通过");
    }
}
