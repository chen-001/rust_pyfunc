//! tail_shared_cache: 引擎「与因子无关的固定冷启动预计算」的跨 run 磁盘缓存 (P2)。
//!
//! 背景
//! ----
//! `tail_backtest_engine` 每次调用都要重算一批**只依赖 (dates, stocks, style 文件,
//! restrict.npy, industry 矩阵)** 的量，与因子完全无关：
//!
//! 1. `IOOptimizedStyleData::load_from_parquet_io_optimized` —— 840MB parquet 解析 (≈18.9s)
//! 2. `neutralize_std_precompute` —— 10 张 (T,N) f64 barra + rank + 每日 Cholesky (≈12.6s)
//! 3. `build_bt_precomputed` —— 每日期全行 radix 排序 (T×N u32)
//!
//! 本模块把这三块按「键摘要」落盘，命中时直接读回，**逐位一致**（详见下方「一致性」）。
//!
//! 缓存键
//! ------
//! `SHA-256(dates ‖ stocks ‖ industry 字节 ‖ style 文件 sha256 ‖ restrict.npy sha256
//!          ‖ ret_sum_gap1.npy sha256 ‖ ret_sum_gap5.npy sha256)` 取前 16 字节十六进制。
//! 键变化即失效（旧键文件不会被读取；是否清理由调用方决定）。
//!
//! 落盘位置
//! --------
//! `dirname(restrict.npy)/_engine_shared_cache/`（= shared 输入目录下，与 cache_root 无关，
//! 因此 `force_restart` 清 cache_root 不会连带清掉缓存）。文件名为
//! `<kind>_<key16>.bin`，原子写（同目录 tmp + rename），并发安全。
//!
//! 一致性
//! ------
//! 写盘写入的就是内存中的原始数值（`f64`/`u32`/`i32`/`usize` 原始字节，小端），读回
//! 不经过任何浮点运算，因此逐位相同。唯一需要「重建」的是两处 Cholesky：
//! `chols` 由落盘的 X'X 调 `Cholesky::new` 重建（与 precompute 内部同一矩阵、同一
//! 确定性算法）；`chols_style` 由落盘的 `xtx_style`（写盘时用与 precompute **同一累积
//! 顺序**复算得到）重建。`industry` 与 `zeros` 不落盘：前者就是传入的输入矩阵本身
//! （`clone`），后者是 `Array2::zeros`，两者都是零风险恒等重建。
//!
//! 环境变量
//! --------
//! `TAIL_SHARED_CACHE=0`（或 `false`/`off`）关闭本缓存，用于 A/B 对账。

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::{Cholesky, DMatrix};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::factor_neutralization_io_optimized::{IOOptimizedStyleData, IOOptimizedStyleDayData};
use crate::factor_neutralize_std::NeutralizeStdShared;
use crate::tail_v5_pipeline::BtPrecomputed;

const CACHE_DIR_NAME: &str = "_engine_shared_cache";
/// 文件头 magic（含格式版本号，格式一变即失效）。
const MAGIC: &[u8; 8] = b"T5SHC001";
const KIND_STYLE: u8 = 1;
const KIND_NEUTRAL: u8 = 2;
const KIND_BT: u8 = 3;
const IO_BUF: usize = 16 << 20;

// ============================================================================
// SHA-256（自实现，避免新增依赖 / 改动 Cargo.toml）
// ============================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub(crate) struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
}

impl Sha256 {
    pub(crate) fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total: 0,
        }
    }

    /// 吞入数据（不更新 total，供 finish 的 padding 使用）。
    fn feed(&mut self, mut data: &[u8]) {
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub(crate) fn update(&mut self, data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        self.feed(data);
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let x = w[i - 15];
            let y = w[i - 2];
            let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
            let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut hh = self.h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(hh);
    }

    pub(crate) fn finish(mut self) -> [u8; 32] {
        let bits = self.total.wrapping_mul(8);
        let rem = (self.total % 64) as usize;
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        let pad_len = if rem < 56 { 56 - rem } else { 120 - rem };
        self.feed(&pad[..pad_len]);
        self.feed(&bits.to_be_bytes());
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[i * 4..i * 4 + 4].copy_from_slice(&self.h[i].to_be_bytes());
        }
        out
    }
}

pub(crate) fn sha256_file(path: &Path) -> Result<[u8; 32], String> {
    let mut f = File::open(path)
        .map_err(|e| format!("打开待摘要文件失败 {}: {}", path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("读取待摘要文件失败 {}: {}", path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finish())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

// ============================================================================
// 字节视图（小端原生布局；文件头写 le/usize 标记，跨端或 32 位平台读回判为 miss）
// ============================================================================

#[inline]
fn bytes_of_f64(d: &[f64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}
#[inline]
fn bytes_of_f64_mut(d: &mut [f64]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u8, std::mem::size_of_val(d))
    }
}
#[inline]
fn bytes_of_usize(d: &[usize]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}
#[inline]
fn bytes_of_usize_mut(d: &mut [usize]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u8, std::mem::size_of_val(d))
    }
}
#[inline]
fn bytes_of_u32(d: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}
#[inline]
fn bytes_of_u32_mut(d: &mut [u32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u8, std::mem::size_of_val(d))
    }
}
#[inline]
fn bytes_of_i32(d: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}
#[inline]
fn bytes_of_i32_mut(d: &mut [i32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u8, std::mem::size_of_val(d))
    }
}

// ============================================================================
// 小端标量读写
// ============================================================================

fn io_err(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

fn wr_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn wr_i64<W: Write>(w: &mut W, v: i64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn wr_f64s<W: Write>(w: &mut W, d: &[f64]) -> std::io::Result<()> {
    w.write_all(bytes_of_f64(d))
}
fn wr_usizes<W: Write>(w: &mut W, d: &[usize]) -> std::io::Result<()> {
    w.write_all(bytes_of_usize(d))
}
fn wr_u32s<W: Write>(w: &mut W, d: &[u32]) -> std::io::Result<()> {
    w.write_all(bytes_of_u32(d))
}
fn wr_i32s<W: Write>(w: &mut W, d: &[i32]) -> std::io::Result<()> {
    w.write_all(bytes_of_i32(d))
}

fn rd_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn rd_i64<R: Read>(r: &mut R) -> std::io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}
fn rd_f64s<R: Read>(r: &mut R, n: usize) -> std::io::Result<Vec<f64>> {
    let mut v = vec![0f64; n];
    r.read_exact(bytes_of_f64_mut(&mut v))?;
    Ok(v)
}
fn rd_usizes<R: Read>(r: &mut R, n: usize) -> std::io::Result<Vec<usize>> {
    let mut v = vec![0usize; n];
    r.read_exact(bytes_of_usize_mut(&mut v))?;
    Ok(v)
}
fn rd_u32s<R: Read>(r: &mut R, n: usize) -> std::io::Result<Vec<u32>> {
    let mut v = vec![0u32; n];
    r.read_exact(bytes_of_u32_mut(&mut v))?;
    Ok(v)
}
fn rd_i32s<R: Read>(r: &mut R, n: usize) -> std::io::Result<Vec<i32>> {
    let mut v = vec![0i32; n];
    r.read_exact(bytes_of_i32_mut(&mut v))?;
    Ok(v)
}
fn rd_utf8<R: Read>(r: &mut R, n: usize) -> std::io::Result<String> {
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|_| io_err("缓存中的字符串不是合法 UTF-8"))
}

/// 读文件头；返回 false 表示不是本格式 / 端序不符 / 长度不够（调用方按 miss 处理）。
fn read_header<R: Read>(r: &mut R, kind: u8) -> std::io::Result<bool> {
    let mut hdr = [0u8; 12];
    if r.read_exact(&mut hdr).is_err() {
        return Ok(false);
    }
    if &hdr[..8] != &MAGIC[..] || hdr[8] != kind {
        return Ok(false);
    }
    let le = cfg!(target_endian = "little");
    if hdr[9] != le as u8 || hdr[10] != std::mem::size_of::<usize>() as u8 {
        return Ok(false);
    }
    Ok(true)
}

fn write_header<W: Write>(w: &mut W, kind: u8) -> std::io::Result<()> {
    let mut hdr = [0u8; 12];
    hdr[..8].copy_from_slice(MAGIC);
    hdr[8] = kind;
    hdr[9] = cfg!(target_endian = "little") as u8;
    hdr[10] = std::mem::size_of::<usize>() as u8;
    w.write_all(&hdr)
}

// ============================================================================
// 文件摘要 sidecar：避免每 run 重算 840MB style parquet 的 sha256
// ============================================================================

#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct DigestEntry {
    size: u64,
    mtime_ns: i64,
    ctime_ns: i64,
    sha256: String,
}

#[derive(Serialize, Deserialize, Default)]
struct DigestTable {
    entries: HashMap<String, DigestEntry>,
}

struct DigestCache {
    path: PathBuf,
    table: DigestTable,
    dirty: bool,
}

fn stat_key(md: &fs::Metadata) -> (u64, i64, i64) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (md.len(), md.mtime(), md.mtime_nsec())
    }
    #[cfg(not(unix))]
    {
        let mtime = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        (md.len(), mtime, 0)
    }
}

fn stat_key_full(md: &fs::Metadata) -> (u64, i64, i64) {
    let (size, mtime, mtime_ns) = stat_key(md);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (size, mtime * 1_000_000_000 + mtime_ns, md.ctime() * 1_000_000_000 + md.ctime_nsec())
    }
    #[cfg(not(unix))]
    {
        (size, mtime, 0)
    }
}

impl DigestCache {
    fn load(path: &Path) -> Self {
        let table = fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<DigestTable>(&s).ok())
            .unwrap_or_default();
        Self {
            path: path.to_path_buf(),
            table,
            dirty: false,
        }
    }

    /// 文件内容 sha256（十六进制）。size+mtime+ctime 未变则复用 sidecar 中的旧摘要。
    fn digest_of_file(&mut self, path: &Path) -> Result<String, String> {
        let md = fs::metadata(path)
            .map_err(|e| format!("stat 失败 {}: {}", path.display(), e))?;
        let (size, mtime, ctime) = stat_key_full(&md);
        let key = path.to_string_lossy().to_string();
        if let Some(e) = self.table.entries.get(&key) {
            if e.size == size && e.mtime_ns == mtime && e.ctime_ns == ctime {
                return Ok(e.sha256.clone());
            }
        }
        let sha = hex_lower(&sha256_file(path)?);
        self.table.entries.insert(
            key,
            DigestEntry {
                size,
                mtime_ns: mtime,
                ctime_ns: ctime,
                sha256: sha.clone(),
            },
        );
        self.dirty = true;
        Ok(sha)
    }

    fn save(&self) {
        if !self.dirty {
            return;
        }
        let Ok(text) = serde_json::to_string(&self.table) else {
            return;
        };
        let tmp = self.path.with_extension(format!("json.tmp.{}", std::process::id()));
        if fs::write(&tmp, text).is_ok() && fs::rename(&tmp, &self.path).is_err() {
            let _ = fs::remove_file(&tmp);
        }
    }
}

// ============================================================================
// 缓存句柄
// ============================================================================

pub(crate) fn cache_enabled() -> bool {
    match std::env::var("TAIL_SHARED_CACHE") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no"),
        Err(_) => true,
    }
}

pub(crate) struct SharedCache {
    root: PathBuf,
    key: String,
}

impl SharedCache {
    /// 计算键并准备缓存目录。任何失败都返回 Err（调用方应退化为「不缓存」）。
    pub(crate) fn open(
        restrict_path: &str,
        style_path: &str,
        ret_sum_gap1_path: &str,
        ret_sum_gap5_path: &str,
        dates: &[i32],
        stocks: &[String],
        industry: Option<&Array2<f64>>,
    ) -> Result<Self, String> {
        let restrict_p = Path::new(restrict_path);
        let root = restrict_p
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .join(CACHE_DIR_NAME);
        fs::create_dir_all(&root)
            .map_err(|e| format!("创建缓存目录失败 {}: {}", root.display(), e))?;

        let mut digests = DigestCache::load(&root.join("digests.json"));
        let style_sha = digests.digest_of_file(Path::new(style_path))?;
        let restrict_sha = digests.digest_of_file(restrict_p)?;
        let rs1_sha = digests.digest_of_file(Path::new(ret_sum_gap1_path))?;
        let rs5_sha = digests.digest_of_file(Path::new(ret_sum_gap5_path))?;
        digests.save();

        let key = compute_key(
            dates,
            stocks,
            industry,
            &style_sha,
            &restrict_sha,
            &rs1_sha,
            &rs5_sha,
        );
        Ok(Self { root, key })
    }

    pub(crate) fn key(&self) -> &str {
        &self.key
    }

    pub(crate) fn dir_display(&self) -> String {
        self.root.display().to_string()
    }

    fn path(&self, kind: &str) -> PathBuf {
        self.root.join(format!("{}_{}.bin", kind, self.key))
    }

    fn exists(&self, kind: &str) -> bool {
        self.path(kind).exists()
    }

    fn atomic_write<F>(&self, kind: &str, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
    {
        let final_path = self.path(kind);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = self.root.join(format!(
            "{}_{}.bin.tmp.{}.{}",
            kind,
            self.key,
            std::process::id(),
            nanos
        ));
        let file = File::create(&tmp)
            .map_err(|e| format!("创建缓存临时文件失败 {}: {}", tmp.display(), e))?;
        let mut w = BufWriter::with_capacity(IO_BUF, file);
        let res = f(&mut w)
            .and_then(|_| w.flush())
            .map_err(|e| format!("写缓存失败 {}: {}", tmp.display(), e));
        drop(w);
        if let Err(e) = res {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        fs::rename(&tmp, &final_path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("缓存原子改名失败 {}: {}", final_path.display(), e)
        })
    }

    // ---------------- style ----------------

    pub(crate) fn load_style(&self) -> Option<IOOptimizedStyleData> {
        if !self.exists("style") {
            return None;
        }
        self.try_load_style().map_err(|e| {
            println!("⚠️ [shared-cache] style 缓存读取失败，回退 parquet 解析: {}", e);
            e
        }).ok()
    }

    fn try_load_style(&self) -> Result<IOOptimizedStyleData, String> {
        let f = File::open(self.path("style")).map_err(|e| e.to_string())?;
        let mut r = BufReader::with_capacity(IO_BUF, f);
        if !read_header(&mut r, KIND_STYLE).map_err(|e| e.to_string())? {
            return Err("style 缓存文件头不匹配".to_string());
        }
        let n_dates = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
        let mut data_by_date = HashMap::with_capacity(n_dates);
        for _ in 0..n_dates {
            let date = rd_i64(&mut r).map_err(|e| e.to_string())?;
            let n = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
            let mut stocks = Vec::with_capacity(n);
            for _ in 0..n {
                let len = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
                stocks.push(rd_utf8(&mut r, len).map_err(|e| e.to_string())?);
            }
            let flat = rd_f64s(&mut r, n * 41).map_err(|e| e.to_string())?;
            let style_matrix = DMatrix::from_row_slice(n, 41, &flat);
            // 与原 load 路径同一算法、同一输入，重建两个回归矩阵。
            let regression_matrix = compute_regression_matrix(&style_matrix)?;
            let style_only_matrix = style_matrix.columns(0, 10).into_owned();
            let regression_matrix_style_only = compute_regression_matrix(&style_only_matrix)?;
            let mut stock_index_map = HashMap::with_capacity(n);
            for (i, s) in stocks.iter().enumerate() {
                stock_index_map.insert(s.clone(), i);
            }
            data_by_date.insert(
                date,
                IOOptimizedStyleDayData {
                    stocks,
                    style_matrix,
                    regression_matrix: Some(Arc::new(regression_matrix)),
                    regression_matrix_style_only: Some(Arc::new(regression_matrix_style_only)),
                    stock_index_map,
                },
            );
        }
        Ok(IOOptimizedStyleData {
            data_by_date,
            file_cache: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub(crate) fn store_style(&self, data: &IOOptimizedStyleData) -> Result<(), String> {
        let mut dates: Vec<i64> = data.data_by_date.keys().copied().collect();
        dates.sort_unstable();
        self.atomic_write("style", |w| {
            write_header(w, KIND_STYLE)?;
            wr_u64(w, dates.len() as u64)?;
            for date in &dates {
                let Some(day) = data.data_by_date.get(date) else {
                    return Err(io_err(&format!("style 数据缺少日期 {}", date)));
                };
                wr_i64(w, *date)?;
                wr_u64(w, day.stocks.len() as u64)?;
                for s in &day.stocks {
                    wr_u64(w, s.len() as u64)?;
                    w.write_all(s.as_bytes())?;
                }
                let (n, cols) = day.style_matrix.shape();
                if cols != 41 {
                    return Err(io_err(&format!("style_matrix 列数不是 41: {}", cols)));
                }
                // DMatrix 为列主序；按列写出，读回时用 from_row_slice 还原。
                let flat = day.style_matrix.as_slice();
                let mut row_major = vec![0f64; n * 41];
                for i in 0..n {
                    for j in 0..41 {
                        row_major[i * 41 + j] = flat[j * n + i];
                    }
                }
                wr_f64s(w, &row_major)?;
            }
            Ok(())
        })
    }

    // ---------------- neutralize ----------------

    pub(crate) fn load_neutral(&self, industry: &Array2<f64>) -> Option<NeutralizeStdShared> {
        if !self.exists("neutral") {
            return None;
        }
        match self.try_load_neutral(industry) {
            Ok(v) => Some(v),
            Err(e) => {
                println!("⚠️ [shared-cache] neutral 缓存读取失败，回退重算: {}", e);
                None
            }
        }
    }

    fn try_load_neutral(
        &self,
        industry: &Array2<f64>,
    ) -> Result<NeutralizeStdShared, String> {
        let f = File::open(self.path("neutral")).map_err(|e| e.to_string())?;
        let mut r = BufReader::with_capacity(IO_BUF, f);
        if !read_header(&mut r, KIND_NEUTRAL).map_err(|e| e.to_string())? {
            return Err("neutral 缓存文件头不匹配".to_string());
        }
        let t = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
        let n = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
        if industry.dim() != (t, n) {
            return Err(format!(
                "neutral 缓存轴 ({},{}) 与 industry ({},{}) 不一致",
                t,
                n,
                industry.nrows(),
                industry.ncols()
            ));
        }
        let cells = t * n;
        let read_arr = |r: &mut BufReader<File>| -> Result<Array2<f64>, String> {
            let v = rd_f64s(r, cells).map_err(|e| e.to_string())?;
            Array2::from_shape_vec((t, n), v).map_err(|e| e.to_string())
        };
        let mut barra_ranked = Vec::with_capacity(10);
        for _ in 0..10 {
            barra_ranked.push(read_arr(&mut r)?);
        }
        let size_ranked = read_arr(&mut r)?;
        let restrict_f64 = read_arr(&mut r)?;
        let ind1 = read_arr(&mut r)?;
        let ind2 = read_arr(&mut r)?;
        let ind1_mask = read_arr(&mut r)?;

        let mut orders = Vec::with_capacity(5);
        for _ in 0..5 {
            let v = rd_usizes(&mut r, cells).map_err(|e| e.to_string())?;
            orders.push(Array2::from_shape_vec((t, n), v).map_err(|e| e.to_string())?);
        }

        let mut per_date = Vec::with_capacity(t);
        let mut chols = Vec::with_capacity(t);
        let mut xdays = Vec::with_capacity(t);
        for _ in 0..t {
            let p = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
            let nv = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
            let valid_idx = rd_u32s(&mut r, nv).map_err(|e| e.to_string())?;
            let valid_cols = rd_i32s(&mut r, nv).map_err(|e| e.to_string())?;
            let xtx_len = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
            let xtx = rd_f64s(&mut r, xtx_len).map_err(|e| e.to_string())?;
            if p == 0 {
                chols.push(None);
            } else {
                if xtx_len != p * p {
                    return Err(format!("X'X 长度 {} 与 p={} 不匹配", xtx_len, p));
                }
                chols.push(Cholesky::new(DMatrix::from_row_slice(p, p, &xtx)));
            }
            per_date.push((p, valid_idx, valid_cols, xtx));
        }
        for _ in 0..t {
            let len = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
            xdays.push(rd_f64s(&mut r, len).map_err(|e| e.to_string())?);
        }
        let mut chols_style = Vec::with_capacity(t);
        for i in 0..t {
            let v = rd_f64s(&mut r, 121).map_err(|e| e.to_string())?;
            if per_date[i].0 == 0 {
                chols_style.push(None);
            } else {
                chols_style.push(Cholesky::new(DMatrix::from_row_slice(11, 11, &v)));
            }
        }

        Ok(NeutralizeStdShared {
            // 输入矩阵本身：与 precompute 内部的 `industry.clone()` 逐位相同。
            industry: industry.to_owned(),
            restrict_f64,
            ind1,
            ind2,
            // precompute 内部即 `Array2::zeros((T,N))`。
            zeros: Array2::zeros((t, n)),
            ind1_mask,
            barra_ranked,
            size_ranked,
            orders,
            per_date,
            chols,
            xdays,
            chols_style,
        })
    }

    pub(crate) fn store_neutral(&self, ns: &NeutralizeStdShared) -> Result<(), String> {
        let (t, n) = ns.industry.dim();
        if ns.barra_ranked.len() != 10 || ns.orders.len() != 5 {
            return Err(format!(
                "NeutralizeStdShared 形状异常: barra={} orders={}",
                ns.barra_ranked.len(),
                ns.orders.len()
            ));
        }
        self.atomic_write("neutral", |w| {
            write_header(w, KIND_NEUTRAL)?;
            wr_u64(w, t as u64)?;
            wr_u64(w, n as u64)?;
            let wr_arr = |w: &mut BufWriter<File>, a: &Array2<f64>| -> std::io::Result<()> {
                if a.dim() != (t, n) {
                    return Err(io_err(&format!("(T,N) 矩阵形状不一致: {:?}", a.dim())));
                }
                let s = a.as_slice().ok_or_else(|| io_err("(T,N) 矩阵非标准行主序"))?;
                wr_f64s(w, s)?;
                Ok(())
            };
            for b in &ns.barra_ranked {
                wr_arr(w, b)?;
            }
            wr_arr(w, &ns.size_ranked)?;
            wr_arr(w, &ns.restrict_f64)?;
            wr_arr(w, &ns.ind1)?;
            wr_arr(w, &ns.ind2)?;
            wr_arr(w, &ns.ind1_mask)?;
            for o in &ns.orders {
                if o.dim() != (t, n) {
                    return Err(io_err(&format!("orders 形状不一致: {:?}", o.dim())));
                }
                let s = o
                    .as_slice()
                    .ok_or_else(|| io_err("orders 非标准行主序"))?;
                wr_usizes(w, s)?;
            }
            for (p, vi, vc, xtx) in &ns.per_date {
                wr_u64(w, *p as u64)?;
                wr_u64(w, vi.len() as u64)?;
                wr_u32s(w, vi)?;
                wr_i32s(w, vc)?;
                wr_u64(w, xtx.len() as u64)?;
                wr_f64s(w, xtx)?;
            }
            for xd in &ns.xdays {
                wr_u64(w, xd.len() as u64)?;
                wr_f64s(w, xd)?;
            }
            for i in 0..t {
                let v = xtx_style_from_xdays(&ns.xdays[i]);
                wr_f64s(w, &v)?;
            }
            Ok(())
        })
    }

    // ---------------- bt_precomputed ----------------

    pub(crate) fn load_bt(&self) -> Option<BtPrecomputed> {
        if !self.exists("bt") {
            return None;
        }
        match self.try_load_bt() {
            Ok(v) => Some(v),
            Err(e) => {
                println!("⚠️ [shared-cache] bt 缓存读取失败，回退重算: {}", e);
                None
            }
        }
    }

    fn try_load_bt(&self) -> Result<BtPrecomputed, String> {
        let f = File::open(self.path("bt")).map_err(|e| e.to_string())?;
        let mut r = BufReader::with_capacity(IO_BUF, f);
        if !read_header(&mut r, KIND_BT).map_err(|e| e.to_string())? {
            return Err("bt 缓存文件头不匹配".to_string());
        }
        let t = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
        let n = rd_u64(&mut r).map_err(|e| e.to_string())? as usize;
        let read_orders = |r: &mut BufReader<File>| -> Result<Vec<Vec<u32>>, String> {
            let mut out = Vec::with_capacity(t);
            for _ in 0..t {
                out.push(rd_u32s(r, n).map_err(|e| e.to_string())?);
            }
            Ok(out)
        };
        let orders_g1 = read_orders(&mut r)?;
        let orders_g5 = read_orders(&mut r)?;
        Ok(BtPrecomputed {
            orders_g1,
            orders_g5,
        })
    }

    pub(crate) fn store_bt(&self, bt: &BtPrecomputed) -> Result<(), String> {
        let t = bt.orders_g1.len();
        if bt.orders_g5.len() != t {
            return Err("BtPrecomputed gap1/gap5 日数不一致".to_string());
        }
        let n = if t > 0 { bt.orders_g1[0].len() } else { 0 };
        self.atomic_write("bt", |w| {
            write_header(w, KIND_BT)?;
            wr_u64(w, t as u64)?;
            wr_u64(w, n as u64)?;
            for orders in [&bt.orders_g1, &bt.orders_g5] {
                for row in orders.iter() {
                    if row.len() != n {
                        return Err(io_err(&format!("orders 行长 {} != {}", row.len(), n)));
                    }
                    wr_u32s(w, row)?;
                }
            }
            Ok(())
        })
    }
}

// ============================================================================
// 与 factor_neutralize_std 内部同序复算的辅助量
// ============================================================================

/// 复算每日纯风格 `[1, b0..b9]` 的 X'X。
///
/// **必须与 `factor_neutralize_std::neutralize_std_precompute` 内 986-1004 行的
/// 累积顺序逐位一致**：按 valid 升序逐行、每行 `[1.0, b0..b9]` 外积累加。
/// `xdays[date]` 即该日 valid 行的 `[b0..b9]` 连续内存（行主序），因此按 pos 升序
/// 遍历 xdays 与遍历 `valid_idx` 完全等价。
fn xtx_style_from_xdays(xd: &[f64]) -> [f64; 121] {
    let mut xtx_s = [0.0f64; 121];
    let nv = xd.len() / 10;
    for pos in 0..nv {
        let b = &xd[pos * 10..pos * 10 + 10];
        xtx_s[0] += 1.0;
        for c in 0..10 {
            xtx_s[c + 1] += b[c];
            xtx_s[(c + 1) * 11] += b[c];
        }
        for c1 in 0..10 {
            xtx_s[(c1 + 1) * 11 + (c1 + 1)] += b[c1] * b[c1];
            for c2 in (c1 + 1)..10 {
                let v = b[c1] * b[c2];
                xtx_s[(c1 + 1) * 11 + (c2 + 1)] += v;
                xtx_s[(c2 + 1) * 11 + (c1 + 1)] += v;
            }
        }
    }
    xtx_s
}

/// 与 `factor_neutralization_io_optimized::compute_regression_matrix_io_optimized`
/// 同一算式（`(X'X)^-1 X'`）。style 缓存只落盘 41 列 style_matrix，回归矩阵由它
/// 确定性重建（同一输入 → 同一 LU 分解 → 逐位相同）。
fn compute_regression_matrix(style_matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let xt = style_matrix.transpose();
    let xtx = &xt * style_matrix;
    let xtx_inv = xtx
        .try_inverse()
        .ok_or_else(|| "风格因子矩阵不可逆，可能存在多重共线性".to_string())?;
    Ok(xtx_inv * xt)
}

// ============================================================================
// 键
// ============================================================================

fn compute_key(
    dates: &[i32],
    stocks: &[String],
    industry: Option<&Array2<f64>>,
    style_sha: &str,
    restrict_sha: &str,
    rs1_sha: &str,
    rs5_sha: &str,
) -> String {
    let mut h = Sha256::new();
    h.update(b"tail_engine_shared_cache_v1\0");
    h.update(&(dates.len() as u64).to_le_bytes());
    for d in dates {
        h.update(&d.to_le_bytes());
    }
    h.update(&(stocks.len() as u64).to_le_bytes());
    for s in stocks {
        h.update(&(s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    }
    match industry {
        Some(a) => {
            h.update(&[1u8]);
            h.update(&(a.nrows() as u64).to_le_bytes());
            h.update(&(a.ncols() as u64).to_le_bytes());
            match a.as_slice() {
                Some(s) => h.update(bytes_of_f64(s)),
                None => {
                    for v in a.iter() {
                        h.update(&v.to_le_bytes());
                    }
                }
            }
        }
        None => h.update(&[0u8]),
    }
    for part in [style_sha, restrict_sha, rs1_sha, rs5_sha] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    hex_lower(&h.finish()[..16])
}
