use super::kernels::{eval_50faces_from_tl_u16, rank_raw_mat_into_u16, SF_SCRATCH, U16_NAN};
use super::pool;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

const IDX_MAGIC: [u8; 8] = *b"RPFBIDX6";
const IDX_HEADER_SIZE: usize = 64;
const CODE_SLOT_BYTES: usize = 16;
const PROJ_DELTA_PREFIX: &str = "factors.proj.delta.";

static IO_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn io_pool() -> &'static rayon::ThreadPool {
    IO_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .expect("failed to build 12-thread IO_POOL")
    })
}

/// Cross-platform exact positional read (`pread`) that compiles cleanly on Linux, macOS, and Windows.
fn pread_exact_at(file: &File, buf: &mut [u8], offset: u64, ctx: &Path) -> Result<(), String> {
    let mut done = 0usize;
    while done < buf.len() {
        let off = offset
            .checked_add(done as u64)
            .ok_or_else(|| format!("pread 偏移溢出 {:?} (off={offset})", ctx))?;
        #[cfg(unix)]
        let res = {
            use std::os::unix::fs::FileExt;
            file.read_at(&mut buf[done..], off)
        };
        #[cfg(not(unix))]
        let res = {
            use std::os::windows::fs::FileExt;
            file.seek_read(&mut buf[done..], off)
        };
        match res {
            Ok(0) => {
                return Err(format!(
                    "pread 提前到达文件尾 {:?} (off={offset}, len={}, read={done})",
                    ctx,
                    buf.len()
                ))
            }
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(format!(
                    "pread 失败 {:?} (off={offset}, len={}): {e}",
                    ctx,
                    buf.len()
                ))
            }
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct GroupEntryJson {
    name: String,
    dir: String,
}

#[derive(Deserialize)]
struct GroupManifestJson {
    version: u32,
    groups: Vec<GroupEntryJson>,
}

#[derive(Deserialize)]
struct DirectPartJson {
    file: String,
    rows: u64,
}

#[derive(Deserialize)]
struct DirectManifestJson {
    version: u32,
    factor_names: Vec<String>,
    dates: Vec<i64>,
    codes: Vec<String>,
    parts: Vec<DirectPartJson>,
}

#[derive(Deserialize)]
struct DeltaPartJson {
    start: u64,
    end: u64,
    file: String,
}

#[derive(Deserialize)]
struct DeltaManifestJson {
    version: u32,
    base_end: u64,
    parts: Vec<DeltaPartJson>,
}

struct ProjSegFast {
    path: PathBuf,
    file: File,
    ver: u32,
    rows: usize,
    val_index: Vec<(u64, u64)>,
    target_idx: Vec<u32>,
}

struct GroupFast {
    factor_offset: usize,
    factor_count: usize,
    segments: Vec<ProjSegFast>,
}

pub struct DirectStoreHandle {
    t_len: usize,
    n_stocks: usize,
    total_factors: usize,
    groups: Vec<GroupFast>,
}

#[inline]
fn decode_col_bytes<'a>(
    seg_ver: u32,
    col_bytes: &'a [u8],
    expected_rows: usize,
    shuf_scratch: &mut Vec<u8>,
    dec_scratch: &'a mut Vec<u8>,
) -> Result<&'a [u8], String> {
    let raw_len = expected_rows
        .checked_mul(4)
        .ok_or_else(|| "列字节长度溢出".to_string())?;
    if seg_ver <= 3 || col_bytes.len() == raw_len {
        if col_bytes.len() != raw_len {
            return Err(format!(
                "列字节长度不匹配: got={}, expected={raw_len}",
                col_bytes.len()
            ));
        }
        return Ok(col_bytes);
    }
    if seg_ver != 4 || col_bytes.is_empty() || col_bytes.len() > raw_len || col_bytes[0] != 1u8 {
        return Err(format!(
            "无效压缩列头: ver={seg_ver}, len={}, raw={raw_len}",
            col_bytes.len()
        ));
    }
    if shuf_scratch.len() < raw_len {
        shuf_scratch.resize(raw_len, 0u8);
    }
    let dec_len = zstd::bulk::decompress_to_buffer(&col_bytes[1..], &mut shuf_scratch[..raw_len])
        .map_err(|e| format!("zstd 解压列失败: {e}"))?;
    if dec_len != raw_len {
        return Err(format!("zstd 解压长度不匹配: got={dec_len}, expected={raw_len}"));
    }
    if dec_scratch.len() < raw_len {
        dec_scratch.resize(raw_len, 0u8);
    }
    let n = expected_rows;
    let shuf = &shuf_scratch[..raw_len];
    let (b0, rest) = shuf.split_at(n);
    let (b1, rest) = rest.split_at(n);
    let (b2, b3) = rest.split_at(n);
    for (i, dst) in dec_scratch[..raw_len].chunks_exact_mut(4).enumerate() {
        dst[0] = b0[i];
        dst[1] = b1[i];
        dst[2] = b2[i];
        dst[3] = b3[i];
    }
    Ok(&dec_scratch[..raw_len])
}

fn open_proj_seg_fast(
    path: &Path,
    dict_dates: &[i64],
    dict_codes: &[String],
    date_pos: &HashMap<i64, usize>,
    stock_pos: &HashMap<String, usize>,
    t_len: usize,
    n_stocks: usize,
) -> Result<Option<ProjSegFast>, String> {
    let pf = File::open(path).map_err(|e| format!("打开投影段 {path:?}: {e}"))?;
    let mut h = [0u8; 32];
    pread_exact_at(&pf, &mut h, 0, path)?;
    let ver = u32::from_le_bytes(h[0..4].try_into().unwrap());
    if ver != 3 && ver != 4 {
        return Err(format!("不支持的投影段版本 {ver}: {path:?}"));
    }
    let fc = u32::from_le_bytes(h[4..8].try_into().unwrap()) as usize;
    let rows = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
    if rows == 0 {
        return Ok(None);
    }
    let row_order_offset = u64::from_le_bytes(h[16..24].try_into().unwrap());
    let row_order_csz = u64::from_le_bytes(h[24..32].try_into().unwrap()) as usize;

    let mut ix = vec![0u8; fc * 16];
    pread_exact_at(&pf, &mut ix, 32, path)?;
    let mut val_index = Vec::with_capacity(fc);
    for v in ix.chunks_exact(16) {
        let off = u64::from_le_bytes(v[0..8].try_into().unwrap());
        let len = u64::from_le_bytes(v[8..16].try_into().unwrap());
        val_index.push((off, len));
    }

    let mut ro_raw = vec![0u8; row_order_csz];
    pread_exact_at(&pf, &mut ro_raw, row_order_offset, path)?;
    let expected_ro = rows * 8;
    let decoder = zstd::stream::read::Decoder::new(&ro_raw[..])
        .map_err(|e| format!("解压行序失败 {path:?}: {e}"))?;
    let mut ro_dec = Vec::with_capacity(expected_ro);
    decoder
        .take((expected_ro + 1) as u64)
        .read_to_end(&mut ro_dec)
        .map_err(|e| format!("读取行序流失败 {path:?}: {e}"))?;
    if ro_dec.len() != expected_ro {
        return Err(format!("行序解压字节数不匹配 {path:?}"));
    }

    let dm: Vec<usize> = dict_dates
        .iter()
        .map(|d| date_pos.get(d).copied().unwrap_or(usize::MAX))
        .collect();
    let cm: Vec<usize> = dict_codes
        .iter()
        .map(|c| {
            let bare = c.split('.').next().unwrap_or(c);
            stock_pos.get(bare).copied().unwrap_or(usize::MAX)
        })
        .collect();

    let mut target_idx = Vec::with_capacity(rows);
    for v in ro_dec.chunks_exact(8) {
        let di = u32::from_le_bytes(v[0..4].try_into().unwrap()) as usize;
        let ci = u32::from_le_bytes(v[4..8].try_into().unwrap()) as usize;
        let r = dm.get(di).copied().unwrap_or(usize::MAX);
        let c = cm.get(ci).copied().unwrap_or(usize::MAX);
        if r < t_len && c < n_stocks {
            target_idx.push((r * n_stocks + c) as u32);
        } else {
            target_idx.push(u32::MAX);
        }
    }

    Ok(Some(ProjSegFast {
        path: path.to_path_buf(),
        file: pf,
        ver,
        rows,
        val_index,
        target_idx,
    }))
}

fn read_colblk_shard_dict_and_segments(
    shard_dir: &Path,
    date_pos: &HashMap<i64, usize>,
    stock_pos: &HashMap<String, usize>,
    t_len: usize,
    n_stocks: usize,
) -> Result<(usize, Vec<ProjSegFast>), String> {
    let idx_bytes = std::fs::read(shard_dir.join("factors.idx"))
        .map_err(|e| format!("读取 factors.idx 失败 {shard_dir:?}: {e}"))?;
    if idx_bytes.len() < IDX_HEADER_SIZE || idx_bytes[0..8] != IDX_MAGIC {
        return Err(format!("无效 factors.idx 头: {shard_dir:?}"));
    }
    let factor_count = u32::from_le_bytes(idx_bytes[12..16].try_into().unwrap()) as usize;

    let dates_bytes = std::fs::read(shard_dir.join("dates.bin"))
        .map_err(|e| format!("读取 dates.bin 失败 {shard_dir:?}: {e}"))?;
    let mut dict_dates = Vec::new();
    let mut seen_d = HashSet::new();
    for ch in dates_bytes.chunks_exact(8) {
        let d = i64::from_le_bytes(ch.try_into().unwrap());
        if seen_d.insert(d) {
            dict_dates.push(d);
        }
    }

    let codes_bytes = std::fs::read(shard_dir.join("codes.bin"))
        .map_err(|e| format!("读取 codes.bin 失败 {shard_dir:?}: {e}"))?;
    let mut dict_codes = Vec::new();
    let mut seen_c = HashSet::new();
    for slot in codes_bytes.chunks_exact(CODE_SLOT_BYTES) {
        let end = slot.iter().position(|&b| b == 0).unwrap_or(CODE_SLOT_BYTES);
        let s = String::from_utf8(slot[..end].to_vec())
            .map_err(|e| format!("codes.bin utf8 错误: {e}"))?;
        if seen_c.insert(s.clone()) {
            dict_codes.push(s);
        }
    }

    let mut seg_paths = Vec::new();
    let base_proj = shard_dir.join("factors.proj");
    if base_proj.exists() {
        seg_paths.push(base_proj);
    }
    let p_man = shard_dir.join("projection.manifest");
    if p_man.exists() {
        let m_bytes = std::fs::read(&p_man).map_err(|e| e.to_string())?;
        let dm: DeltaManifestJson = serde_json::from_slice(&m_bytes).map_err(|e| e.to_string())?;
        if dm.version != 1 {
            return Err(format!("不支持的 projection.manifest 版本: {shard_dir:?}"));
        }
        let _ = dm.base_end;
        for p in dm.parts {
            let _ = (p.start, p.end);
            seg_paths.push(shard_dir.join(p.file));
        }
    } else {
        let mut deltas: Vec<(u64, PathBuf)> = Vec::new();
        for ent in std::fs::read_dir(shard_dir).map_err(|e| e.to_string())? {
            let ent = ent.map_err(|e| e.to_string())?;
            let nm = ent.file_name();
            let nm_s = nm.to_string_lossy();
            if let Some(rest) = nm_s.strip_prefix(PROJ_DELTA_PREFIX) {
                if let Ok(end_off) = rest.parse::<u64>() {
                    deltas.push((end_off, ent.path()));
                }
            }
        }
        deltas.sort_by_key(|(e, _)| *e);
        if let Some((_, newest)) = deltas.last() {
            seg_paths.push(newest.clone());
        }
    }

    let mut segs = Vec::new();
    for sp in seg_paths {
        if let Some(seg) =
            open_proj_seg_fast(&sp, &dict_dates, &dict_codes, date_pos, stock_pos, t_len, n_stocks)?
        {
            segs.push(seg);
        }
    }
    Ok((factor_count, segs))
}

fn open_group_fast(
    group_dir: &Path,
    date_pos: &HashMap<i64, usize>,
    stock_pos: &HashMap<String, usize>,
    t_len: usize,
    n_stocks: usize,
) -> Result<(usize, Vec<ProjSegFast>), String> {
    let direct_man_path = group_dir.join("segments.manifest.json");
    if direct_man_path.exists() {
        let m_bytes = std::fs::read(&direct_man_path)
            .map_err(|e| format!("读取 segments.manifest.json 失败: {e}"))?;
        let dm: DirectManifestJson = serde_json::from_slice(&m_bytes)
            .map_err(|e| format!("解析 segments.manifest.json 失败: {e}"))?;
        if dm.version != 1 {
            return Err("不支持的 segments.manifest.json 版本".into());
        }
        let fc = dm.factor_names.len();
        let mut segs = Vec::with_capacity(dm.parts.len());
        for p in dm.parts {
            if p.rows == 0 {
                continue;
            }
            if let Some(seg) = open_proj_seg_fast(
                &group_dir.join(&p.file),
                &dm.dates,
                &dm.codes,
                date_pos,
                stock_pos,
                t_len,
                n_stocks,
            )? {
                segs.push(seg);
            }
        }
        return Ok((fc, segs));
    }

    let mut shard_dirs: Vec<(usize, PathBuf)> = Vec::new();
    for ent in std::fs::read_dir(group_dir).map_err(|e| format!("读取目录失败 {group_dir:?}: {e}"))? {
        let ent = ent.map_err(|e| e.to_string())?;
        let nm = ent.file_name();
        let nm_s = nm.to_string_lossy();
        if let Some(num_s) = nm_s.strip_prefix("shard_") {
            if let Ok(idx) = num_s.parse::<usize>() {
                shard_dirs.push((idx, ent.path()));
            }
        }
    }

    if !shard_dirs.is_empty() {
        shard_dirs.sort_by_key(|(i, _)| *i);
        let mut all_segs = Vec::new();
        let mut fc_expected = 0usize;
        for (si, (_, sdir)) in shard_dirs.into_iter().enumerate() {
            let (fc, mut segs) =
                read_colblk_shard_dict_and_segments(&sdir, date_pos, stock_pos, t_len, n_stocks)?;
            if si == 0 {
                fc_expected = fc;
            } else if fc != fc_expected {
                return Err(format!("分片因子数不一致: {sdir:?}"));
            }
            all_segs.append(&mut segs);
        }
        Ok((fc_expected, all_segs))
    } else {
        read_colblk_shard_dict_and_segments(group_dir, date_pos, stock_pos, t_len, n_stocks)
    }
}

impl DirectStoreHandle {
    fn open(store_dir: &str, dates: &[i32], bare_stocks: &[String]) -> Result<Self, String> {
        let root = PathBuf::from(store_dir);
        let t_len = dates.len();
        let n_stocks = bare_stocks.len();

        let date_pos: HashMap<i64, usize> = dates
            .iter()
            .enumerate()
            .map(|(i, &d)| (d as i64, i))
            .collect();
        let stock_pos: HashMap<String, usize> = bare_stocks
            .iter()
            .enumerate()
            .map(|(i, s)| (s.split('.').next().unwrap_or(s).to_string(), i))
            .collect();

        let fg_path = root.join("factor_groups.json");
        let group_entries = if fg_path.exists() {
            let bytes = std::fs::read(&fg_path).map_err(|e| e.to_string())?;
            let gm: GroupManifestJson = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            if gm.version != 1 {
                return Err("不支持的 factor_groups.json 版本".into());
            }
            gm.groups
        } else {
            vec![GroupEntryJson {
                name: "base".to_string(),
                dir: ".".to_string(),
            }]
        };

        let mut groups = Vec::with_capacity(group_entries.len());
        let mut offset = 0usize;
        for ge in group_entries {
            let _ = &ge.name;
            let gdir = if ge.dir == "." { root.clone() } else { root.join(&ge.dir) };
            let (fc, segs) = open_group_fast(&gdir, &date_pos, &stock_pos, t_len, n_stocks)?;
            if segs.is_empty() {
                return Err(format!("存储未投影或无有效段: {gdir:?}"));
            }
            groups.push(GroupFast {
                factor_offset: offset,
                factor_count: fc,
                segments: segs,
            });
            offset += fc;
        }

        Ok(Self {
            t_len,
            n_stocks,
            total_factors: offset,
            groups,
        })
    }

    fn resolve_col(&self, global_col: usize) -> Option<(usize, usize)> {
        for (gi, g) in self.groups.iter().enumerate() {
            if global_col >= g.factor_offset && global_col < g.factor_offset + g.factor_count {
                return Some((gi, global_col - g.factor_offset));
            }
        }
        None
    }
}

struct SpanSpec {
    group_idx: usize,
    local_start: usize,
    local_end: usize,
    items: Vec<(usize, usize)>,
}

struct PrefetchedSegmentBytes {
    seg_idx: usize,
    first_off: u64,
    bytes: Vec<u8>,
}

struct PrefetchedSpan {
    segs: Vec<PrefetchedSegmentBytes>,
    read_secs: f64,
    total_bytes: usize,
}

fn alloc_reusable_pref_span(handle: &DirectStoreHandle, spans: &[SpanSpec]) -> PrefetchedSpan {
    let max_segs = handle.groups.iter().map(|g| g.segments.len()).max().unwrap_or(0);
    let mut max_bytes_per_seg = vec![0usize; max_segs];
    for sp in spans {
        let grp = &handle.groups[sp.group_idx];
        for (si, seg) in grp.segments.iter().enumerate() {
            let first_off = seg.val_index[sp.local_start].0;
            let last = seg.val_index[sp.local_end - 1];
            let len = (last.0 + last.1 - first_off) as usize;
            if len > max_bytes_per_seg[si] {
                max_bytes_per_seg[si] = len;
            }
        }
    }
    let segs: Vec<PrefetchedSegmentBytes> = pool().install(|| {
        max_bytes_per_seg
            .into_par_iter()
            .enumerate()
            .map(|(seg_idx, cap)| {
                let mut v = Vec::with_capacity(cap + 4096);
                unsafe {
                    v.set_len(cap);
                }
                let mut p = 0usize;
                while p < cap {
                    v[p] = 0;
                    p += 4096;
                }
                v.clear();
                PrefetchedSegmentBytes {
                    seg_idx,
                    first_off: 0,
                    bytes: v,
                }
            })
            .collect()
    });
    PrefetchedSpan {
        segs,
        read_secs: 0.0,
        total_bytes: 0,
    }
}

fn prefetch_span_into(
    group: &GroupFast,
    local_start: usize,
    local_end: usize,
    dst: &mut PrefetchedSpan,
) -> Result<(), String> {
    let t0 = Instant::now();
    let n_segs = group.segments.len();
    let res: Result<Vec<usize>, String> = io_pool().install(|| {
        dst.segs[..n_segs]
            .par_iter_mut()
            .enumerate()
            .map(|(seg_idx, ps)| {
                let seg = &group.segments[seg_idx];
                let first_off = seg.val_index[local_start].0;
                let last = seg.val_index[local_end - 1];
                let total_len = (last.0 + last.1 - first_off) as usize;
                if ps.bytes.capacity() < total_len {
                    ps.bytes.reserve(total_len - ps.bytes.capacity());
                }
                unsafe {
                    ps.bytes.set_len(total_len);
                }
                ps.seg_idx = seg_idx;
                ps.first_off = first_off;
                pread_exact_at(&seg.file, &mut ps.bytes[..total_len], first_off, &seg.path)?;
                Ok(total_len)
            })
            .collect()
    });
    let lens = res?;
    dst.total_bytes = lens.into_iter().sum();
    dst.read_secs = t0.elapsed().as_secs_f64();
    Ok(())
}

fn scatter_col_from_prefetched(
    group: &GroupFast,
    pref: &PrefetchedSpan,
    local_col: usize,
    raw_mat_out: &mut [f32],
    shuf_scratch: &mut Vec<u8>,
    dec_scratch: &mut Vec<u8>,
) -> Result<(), String> {
    raw_mat_out.fill(f32::NAN);
    for ps in &pref.segs {
        let seg = &group.segments[ps.seg_idx];
        let (off, len) = seg.val_index[local_col];
        let rel = (off - ps.first_off) as usize;
        let col_slice = &ps.bytes[rel..rel + (len as usize)];
        let raw_bytes = decode_col_bytes(seg.ver, col_slice, seg.rows, shuf_scratch, dec_scratch)?;
        for (&t_idx, ch) in seg.target_idx.iter().zip(raw_bytes.chunks_exact(4)) {
            if t_idx != u32::MAX {
                let x = f32::from_le_bytes(ch.try_into().unwrap());
                raw_mat_out[t_idx as usize] = if x.is_finite() { x } else { f32::NAN };
            }
        }
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn sf_open_direct_store(
    store_dir_cstr: *const c_char,
    dates_ptr: *const i32,
    t_len: usize,
    stocks_flat_ptr: *const u8,
    stock_stride: usize,
    n_stocks: usize,
) -> *mut DirectStoreHandle {
    let store_dir = match CStr::from_ptr(store_dir_cstr).to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[sf_open_direct_store] invalid utf8 path: {e}");
            return std::ptr::null_mut();
        }
    };
    let dates = std::slice::from_raw_parts(dates_ptr, t_len);
    let stocks_flat = std::slice::from_raw_parts(stocks_flat_ptr, n_stocks * stock_stride);
    let mut bare_stocks = Vec::with_capacity(n_stocks);
    for i in 0..n_stocks {
        let sl = &stocks_flat[i * stock_stride..(i + 1) * stock_stride];
        let end = sl.iter().position(|&b| b == 0).unwrap_or(stock_stride);
        let s = String::from_utf8_lossy(&sl[..end]).into_owned();
        bare_stocks.push(s);
    }

    match DirectStoreHandle::open(store_dir, dates, &bare_stocks) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            eprintln!("[sf_open_direct_store] ERROR opening {store_dir}: {e}");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sf_close_direct_store(handle_ptr: *mut DirectStoreHandle) {
    if !handle_ptr.is_null() {
        drop(Box::from_raw(handle_ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn sf_load_and_rank_store_cols_u16(
    handle_ptr: *const DirectStoreHandle,
    store_cols_ptr: *const i32,
    n_cols: usize,
    out_raw_u16_ptr: *mut u16,
    out_fold_u16_ptr: *mut u16,
) -> i32 {
    if handle_ptr.is_null() {
        return -1;
    }
    if n_cols == 0 {
        return 0;
    }
    let handle = &*handle_ptr;
    let t_len = handle.t_len;
    let n_stocks = handle.n_stocks;
    let tn = t_len * n_stocks;
    let store_cols = std::slice::from_raw_parts(store_cols_ptr, n_cols);
    let out_raw_u16 = std::slice::from_raw_parts_mut(out_raw_u16_ptr, n_cols * tn);
    let out_fold_u16 = std::slice::from_raw_parts_mut(out_fold_u16_ptr, n_cols * tn);

    let mut raw_mats = vec![f32::NAN; n_cols * tn];
    let p_raw_mats = raw_mats.as_mut_ptr() as usize;

    let mut by_group: Vec<Vec<(usize, usize)>> = vec![Vec::new(); handle.groups.len()];
    for (ci, &gc_i32) in store_cols.iter().enumerate() {
        let gcol = gc_i32 as usize;
        match handle.resolve_col(gcol) {
            Some((gi, lcol)) => by_group[gi].push((lcol, ci)),
            None => {
                eprintln!("[sf_load_and_rank_store_cols_u16] ERROR: global_col {gcol} out of range");
                return -2;
            }
        }
    }
    for items in &mut by_group {
        items.sort_by_key(|&(lcol, _)| lcol);
    }

    for (gi, items) in by_group.iter().enumerate() {
        if items.is_empty() {
            continue;
        }
        let group = &handle.groups[gi];
        let rd_ok = io_pool().install(|| {
            group
                .segments
                .par_iter()
                .map(|seg| -> Result<(), String> {
                    let mut col_buf = Vec::new();
                    let mut shuf_scratch = Vec::new();
                    let mut dec_scratch = Vec::new();
                    for &(local_col, ci) in items {
                        let (off, len) = seg.val_index[local_col];
                        col_buf.resize(len as usize, 0u8);
                        pread_exact_at(&seg.file, &mut col_buf, off, &seg.path)?;
                        let raw_bytes =
                            decode_col_bytes(seg.ver, &col_buf, seg.rows, &mut shuf_scratch, &mut dec_scratch)?;
                        let dst = std::slice::from_raw_parts_mut((p_raw_mats as *mut f32).add(ci * tn), tn);
                        for (&t_idx, ch) in seg.target_idx.iter().zip(raw_bytes.chunks_exact(4)) {
                            if t_idx != u32::MAX {
                                let x = f32::from_le_bytes(ch.try_into().unwrap());
                                dst[t_idx as usize] = if x.is_finite() { x } else { f32::NAN };
                            }
                        }
                    }
                    Ok(())
                })
                .collect::<Result<Vec<()>, String>>()
        });
        if let Err(e) = rd_ok {
            eprintln!("[sf_load_and_rank_store_cols_u16] ERROR: {e}");
            return -2;
        }
    }

    pool().install(|| {
        out_raw_u16
            .par_chunks_mut(tn)
            .zip(out_fold_u16.par_chunks_mut(tn))
            .zip(raw_mats.par_chunks(tn))
            .for_each(|((r_u16, f_u16), raw_col)| {
                SF_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    sc.ensure(n_stocks, 14, 0);
                    let sc_ref = &mut *sc;
                    rank_raw_mat_into_u16(
                        raw_col,
                        t_len,
                        n_stocks,
                        &mut sc_ref.k_u32,
                        &mut sc_ref.s_u16,
                        &mut sc_ref.tmp_k,
                        &mut sc_ref.tmp_i,
                        &mut sc_ref.fold_vals[..n_stocks],
                        r_u16,
                        f_u16,
                    );
                });
            });
    });

    0
}

#[inline]
fn score_raw_factor_50faces_g5(ic5_50: &[f32], vd5_50: &[u16], n_g5: usize) -> f64 {
    let min_vd = ((0.60 * (n_g5 as f64)) as u16).max(20);
    let ir_sqrt = (50.0f64).sqrt();
    let mut max_ir = 0.0f64;
    let mut max_ic = 0.0f64;
    let mut max_bal = 0.0f64;
    for f in 0..50 {
        if vd5_50[f] < min_vd {
            continue;
        }
        let row = &ic5_50[f * n_g5..(f + 1) * n_g5];
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        let mut cnt = 0usize;
        for &v in row {
            if v.is_finite() {
                let vf = v as f64;
                sum += vf;
                sq += vf * vf;
                cnt += 1;
            }
        }
        if cnt < 20 {
            continue;
        }
        let mu = sum / (cnt as f64);
        let mua = mu.abs();
        if mua < 0.005 {
            continue;
        }
        let var = (sq / (cnt as f64) - mu * mu).max(0.0);
        if var <= 1e-12 {
            continue;
        }
        let sd = var.sqrt().max(1e-6);
        let ir = (mua / sd) * ir_sqrt;
        let bal = (mua * ir).sqrt();
        if ir > max_ir {
            max_ir = ir;
        }
        if mua > max_ic {
            max_ic = mua;
        }
        if bal > max_bal {
            max_bal = bal;
        }
    }
    max_ir + 200.0 * max_ic + 15.0 * max_bal
}

#[no_mangle]
pub unsafe extern "C" fn sf_scan_store_double_buffered_top_u16(
    handle_ptr: *const DirectStoreHandle,
    series_label_cstr: *const c_char,
    valid_store_cols_ptr: *const i32,
    n_valid_raw: usize,
    span_max_cols: usize,
    win_len_turn_ptr: *const u8,
    n_b10: usize,
    n_c2: usize,
    need_g5_c2: i32,
    g1_rows_ptr: *const i32,
    n_g1: usize,
    ev_g1_ptr: *const u8,
    q14_g1_ptr: *const f32,
    ret1_ord_ptr: *const f32,
    g5_from_g1_ptr: *const i32,
    n_g5: usize,
    ev_g5_ptr: *const u8,
    q14_g5_ptr: *const f32,
    ret5_ord_ptr: *const f32,
    fp_g1_map_ptr: *const i32,
    fp_stock_idx_ptr: *const i32,
    n_fp_dates: usize,
    fp_k_per_day: usize,
    out_ic5_b10_ptr: *mut f32,
    out_ic5_c2_ptr: *mut f32,
    out_ic1_b10_ptr: *mut f32,
    out_fp_b10_i8_ptr: *mut i8,
    out_fp_b10_sc_ptr: *mut f32,
    out_res_b10_sc_ptr: *mut f32,
    out_fp_c2_i8_ptr: *mut i8,
    out_fp_c2_sc_ptr: *mut f32,
    out_res_c2_sc_ptr: *mut f32,
    out_vdays_g5_ptr: *mut u16,
    out_vdays_g1_ptr: *mut u16,
    out_raw_u16_slab_ptr: *mut u16,
    out_fold_u16_slab_ptr: *mut u16,
    slab_cap: usize,
    must_keep_raw_u8_ptr: *const u8,
    out_slab_raw_ids_ptr: *mut i32,
    out_slab_prios_ptr: *mut f64,
) -> i32 {
    if handle_ptr.is_null() || n_valid_raw == 0 {
        return -1;
    }
    let handle = &*handle_ptr;
    let label = CStr::from_ptr(series_label_cstr).to_string_lossy();
    let t_len = handle.t_len;
    let n_stocks = handle.n_stocks;
    let tn = t_len * n_stocks;
    let d_fp = n_fp_dates * fp_k_per_day;
    let has_c2 = need_g5_c2 != 0;

    let valid_store_cols = std::slice::from_raw_parts(valid_store_cols_ptr, n_valid_raw);
    let win_len_turn = std::slice::from_raw_parts(win_len_turn_ptr, 3 * tn);
    let g1_rows = std::slice::from_raw_parts(g1_rows_ptr, n_g1);
    let ev_g1 = std::slice::from_raw_parts(ev_g1_ptr, n_g1 * n_stocks);
    let q14_g1 = std::slice::from_raw_parts(q14_g1_ptr, n_g1 * n_c2 * n_stocks);
    let ret1_ord = std::slice::from_raw_parts(ret1_ord_ptr, n_g1 * n_stocks);
    let g5_from_g1 = std::slice::from_raw_parts(g5_from_g1_ptr, n_g1);
    let ev_g5 = std::slice::from_raw_parts(ev_g5_ptr, n_g5 * n_stocks);
    let q14_g5 = std::slice::from_raw_parts(q14_g5_ptr, n_g5 * n_c2 * n_stocks);
    let ret5_ord = std::slice::from_raw_parts(ret5_ord_ptr, n_g5 * n_stocks);
    let fp_g1_map = std::slice::from_raw_parts(fp_g1_map_ptr, n_g1);
    let fp_stock_idx = std::slice::from_raw_parts(fp_stock_idx_ptr, d_fp);
    let must_keep = if !must_keep_raw_u8_ptr.is_null() {
        Some(std::slice::from_raw_parts(must_keep_raw_u8_ptr, n_valid_raw))
    } else {
        None
    };

    let max_span = span_max_cols.max(400);
    let mut spans: Vec<SpanSpec> = Vec::new();
    for (out_idx, &gcol_i32) in valid_store_cols.iter().enumerate() {
        let gcol = gcol_i32 as usize;
        if gcol >= handle.total_factors {
            eprintln!("[sf_scan_store_double_buffered_top_u16] col {gcol} out of range {}", handle.total_factors);
            return -2;
        }
        let (gi, lcol) = handle.resolve_col(gcol).unwrap();
        let append_cur = if let Some(last) = spans.last() {
            last.group_idx == gi && lcol >= last.local_start && (lcol - last.local_start) < max_span
        } else {
            false
        };
        if append_cur {
            let last = spans.last_mut().unwrap();
            last.local_end = last.local_end.max(lcol + 1);
            last.items.push((out_idx, lcol));
        } else {
            spans.push(SpanSpec {
                group_idx: gi,
                local_start: lcol,
                local_end: lcol + 1,
                items: vec![(out_idx, lcol)],
            });
        }
    }

    let p_ic5_b10 = out_ic5_b10_ptr as usize;
    let p_ic5_c2 = out_ic5_c2_ptr as usize;
    let p_ic1_b10 = out_ic1_b10_ptr as usize;
    let p_fp10_i8 = out_fp_b10_i8_ptr as usize;
    let p_fp10_sc = out_fp_b10_sc_ptr as usize;
    let p_rs10_sc = out_res_b10_sc_ptr as usize;
    let p_fpc2_i8 = out_fp_c2_i8_ptr as usize;
    let p_fpc2_sc = out_fp_c2_sc_ptr as usize;
    let p_rsc2_sc = out_res_c2_sc_ptr as usize;
    let p_vd5 = out_vdays_g5_ptr as usize;
    let p_vd1 = out_vdays_g1_ptr as usize;
    let p_ru_slab = out_raw_u16_slab_ptr as usize;
    let p_fu_slab = out_fold_u16_slab_ptr as usize;

    let use_slab = slab_cap > 0 && p_ru_slab != 0 && p_fu_slab != 0;
    let direct_fill_all = use_slab && n_valid_raw <= slab_cap;
    let max_items_in_span = spans.iter().map(|s| s.items.len()).max().unwrap_or(0);

    if use_slab {
        let touch_slots = if direct_fill_all { n_valid_raw } else { slab_cap };
        let ru_all = std::slice::from_raw_parts_mut(out_raw_u16_slab_ptr, touch_slots * tn);
        let fu_all = std::slice::from_raw_parts_mut(out_fold_u16_slab_ptr, touch_slots * tn);
        pool().install(|| {
            ru_all
                .par_chunks_mut(tn)
                .zip(fu_all.par_chunks_mut(tn))
                .for_each(|(r_s, f_s)| {
                    let mut i = 0usize;
                    while i < tn {
                        r_s[i] = U16_NAN;
                        f_s[i] = U16_NAN;
                        i += 2048;
                    }
                });
        });
    }

    let mut span_ru_buf: Vec<u16> = if use_slab && !direct_fill_all {
        let len = max_items_in_span * tn;
        let mut v = Vec::with_capacity(len);
        unsafe {
            v.set_len(len);
        }
        pool().install(|| {
            v.par_chunks_mut(tn).for_each(|s| {
                let mut i = 0usize;
                while i < tn {
                    s[i] = U16_NAN;
                    i += 2048;
                }
            });
        });
        v
    } else {
        Vec::new()
    };
    let mut span_fu_buf: Vec<u16> = if use_slab && !direct_fill_all {
        let len = max_items_in_span * tn;
        let mut v = Vec::with_capacity(len);
        unsafe {
            v.set_len(len);
        }
        pool().install(|| {
            v.par_chunks_mut(tn).for_each(|s| {
                let mut i = 0usize;
                while i < tn {
                    s[i] = U16_NAN;
                    i += 2048;
                }
            });
        });
        v
    } else {
        Vec::new()
    };
    let p_span_ru = span_ru_buf.as_mut_ptr() as usize;
    let p_span_fu = span_fu_buf.as_mut_ptr() as usize;

    let mut used_slots = 0usize;
    let mut slot_raw_id = vec![-1i32; slab_cap];
    let mut slot_prio = vec![-1.0e18f64; slab_cap];
    let mut cur_min_slot = 0usize;
    let mut cur_min_prio = f64::INFINITY;

    let mut cur_pref = alloc_reusable_pref_span(handle, &spans);
    let mut next_pref = if spans.len() > 1 {
        alloc_reusable_pref_span(handle, &spans)
    } else {
        PrefetchedSpan {
            segs: Vec::new(),
            read_secs: 0.0,
            total_bytes: 0,
        }
    };

    let t_series = Instant::now();
    let n_spans = spans.len();

    if let Err(e) = prefetch_span_into(
        &handle.groups[spans[0].group_idx],
        spans[0].local_start,
        spans[0].local_end,
        &mut cur_pref,
    ) {
        eprintln!("[sf_scan_store_double_buffered_top_u16] Prefetch span 0 error: {e}");
        return -3;
    }
    let mut io_wait_secs = cur_pref.read_secs;

    for si in 0..n_spans {
        let cur_span = &spans[si];
        let next_span_opt = spans.get(si + 1);
        let cur_group = &handle.groups[cur_span.group_idx];

        let mut next_res: Option<Result<(), String>> = None;
        let mut comp_secs = 0.0f64;

        let cur_wait_reported = io_wait_secs;
        let mut next_io_wait_secs = 0.0f64;
        let next_pref_ref = &mut next_pref;

        std::thread::scope(|s| {
            let bg_handle = next_span_opt.map(|n_sp| {
                let n_grp = &handle.groups[n_sp.group_idx];
                let ls = n_sp.local_start;
                let le = n_sp.local_end;
                s.spawn(move || prefetch_span_into(n_grp, ls, le, next_pref_ref))
            });

            let t_comp = Instant::now();
            pool().install(|| {
                cur_span
                    .items
                    .par_iter()
                    .enumerate()
                    .with_max_len(1)
                    .for_each(|(item_idx, &(out_bi, local_col))| {
                        let ic5_b10_c = std::slice::from_raw_parts_mut(
                            (p_ic5_b10 as *mut f32).add(out_bi * 50 * n_g5),
                            50 * n_g5,
                        );
                        let ic5_c2_opt = if has_c2 && p_ic5_c2 != 0 {
                            Some(std::slice::from_raw_parts_mut(
                                (p_ic5_c2 as *mut f32).add(out_bi * 50 * n_g5),
                                50 * n_g5,
                            ))
                        } else {
                            None
                        };
                        let ic1_b10_opt = if p_ic1_b10 != 0 {
                            Some(std::slice::from_raw_parts_mut(
                                (p_ic1_b10 as *mut f32).add(out_bi * 50 * n_g1),
                                50 * n_g1,
                            ))
                        } else {
                            None
                        };
                        let fp10_i8_c = std::slice::from_raw_parts_mut(
                            (p_fp10_i8 as *mut i8).add(out_bi * 50 * d_fp),
                            50 * d_fp,
                        );
                        let fp10_sc_c = std::slice::from_raw_parts_mut(
                            (p_fp10_sc as *mut f32).add(out_bi * 50),
                            50,
                        );
                        let rs10_sc_c = std::slice::from_raw_parts_mut(
                            (p_rs10_sc as *mut f32).add(out_bi * 50),
                            50,
                        );
                        let fpc2_opt = if has_c2 && p_fpc2_i8 != 0 && p_fpc2_sc != 0 && p_rsc2_sc != 0 {
                            Some((
                                std::slice::from_raw_parts_mut(
                                    (p_fpc2_i8 as *mut i8).add(out_bi * 50 * d_fp),
                                    50 * d_fp,
                                ),
                                std::slice::from_raw_parts_mut(
                                    (p_fpc2_sc as *mut f32).add(out_bi * 50),
                                    50,
                                ),
                                std::slice::from_raw_parts_mut(
                                    (p_rsc2_sc as *mut f32).add(out_bi * 50),
                                    50,
                                ),
                            ))
                        } else {
                            None
                        };
                        let vd5_c = std::slice::from_raw_parts_mut(
                            (p_vd5 as *mut u16).add(out_bi * 50),
                            50,
                        );
                        let mut vd1_dummy = [0u16; 50];
                        let vd1_c = if p_vd1 != 0 {
                            std::slice::from_raw_parts_mut((p_vd1 as *mut u16).add(out_bi * 50), 50)
                        } else {
                            &mut vd1_dummy[..]
                        };

                        SF_SCRATCH.with(|cell| {
                            let mut sc = cell.borrow_mut();
                            sc.ensure(n_stocks, n_c2, d_fp);
                            sc.ensure_matrix_tl(tn);
                            {
                                let sc_ref = &mut *sc;
                                scatter_col_from_prefetched(
                                    cur_group,
                                    &cur_pref,
                                    local_col,
                                    &mut sc_ref.raw_mat_tl[..tn],
                                    &mut sc_ref.shuf_scratch,
                                    &mut sc_ref.dec_scratch,
                                )
                                .expect("scatter_col_from_prefetched failed");

                                rank_raw_mat_into_u16(
                                    &sc_ref.raw_mat_tl[..tn],
                                    t_len,
                                    n_stocks,
                                    &mut sc_ref.k_u32,
                                    &mut sc_ref.s_u16,
                                    &mut sc_ref.tmp_k,
                                    &mut sc_ref.tmp_i,
                                    &mut sc_ref.fold_vals[..n_stocks],
                                    &mut sc_ref.raw_u16_tl[..tn],
                                    &mut sc_ref.fold_u16_tl[..tn],
                                );
                                if direct_fill_all {
                                    let dst_r = std::slice::from_raw_parts_mut(
                                        (p_ru_slab as *mut u16).add(out_bi * tn),
                                        tn,
                                    );
                                    let dst_f = std::slice::from_raw_parts_mut(
                                        (p_fu_slab as *mut u16).add(out_bi * tn),
                                        tn,
                                    );
                                    dst_r.copy_from_slice(&sc_ref.raw_u16_tl[..tn]);
                                    dst_f.copy_from_slice(&sc_ref.fold_u16_tl[..tn]);
                                } else if use_slab {
                                    let dst_r = std::slice::from_raw_parts_mut(
                                        (p_span_ru as *mut u16).add(item_idx * tn),
                                        tn,
                                    );
                                    let dst_f = std::slice::from_raw_parts_mut(
                                        (p_span_fu as *mut u16).add(item_idx * tn),
                                        tn,
                                    );
                                    dst_r.copy_from_slice(&sc_ref.raw_u16_tl[..tn]);
                                    dst_f.copy_from_slice(&sc_ref.fold_u16_tl[..tn]);
                                }
                            }
                            eval_50faces_from_tl_u16(
                                &mut sc,
                                t_len,
                                n_stocks,
                                win_len_turn,
                                n_b10,
                                n_c2,
                                has_c2,
                                g1_rows,
                                n_g1,
                                ev_g1,
                                q14_g1,
                                ret1_ord,
                                g5_from_g1,
                                n_g5,
                                ev_g5,
                                q14_g5,
                                ret5_ord,
                                fp_g1_map,
                                fp_stock_idx,
                                n_fp_dates,
                                fp_k_per_day,
                                ic5_b10_c,
                                ic5_c2_opt,
                                ic1_b10_opt,
                                fp10_i8_c,
                                fp10_sc_c,
                                rs10_sc_c,
                                fpc2_opt,
                                vd5_c,
                                vd1_c,
                            );
                        });
                    });

                if use_slab && !direct_fill_all {
                    let span_prios: Vec<f64> = cur_span
                        .items
                        .par_iter()
                        .map(|&(out_bi, _)| {
                            if let Some(mk) = must_keep {
                                if mk[out_bi] != 0 {
                                    return 1.0e12 + ((n_valid_raw - out_bi) as f64);
                                }
                            }
                            let ic5_c = std::slice::from_raw_parts(
                                (p_ic5_b10 as *const f32).add(out_bi * 50 * n_g5),
                                50 * n_g5,
                            );
                            let vd5_c = std::slice::from_raw_parts(
                                (p_vd5 as *const u16).add(out_bi * 50),
                                50,
                            );
                            score_raw_factor_50faces_g5(ic5_c, vd5_c, n_g5)
                        })
                        .collect();

                    let mut slot_owner = vec![-1i32; slab_cap];
                    for (item_idx, &(out_bi, _)) in cur_span.items.iter().enumerate() {
                        let prio = span_prios[item_idx];
                        if used_slots < slab_cap {
                            let sl = used_slots;
                            used_slots += 1;
                            slot_raw_id[sl] = out_bi as i32;
                            slot_prio[sl] = prio;
                            slot_owner[sl] = item_idx as i32;
                            if used_slots == slab_cap {
                                let mut m_sl = 0usize;
                                let mut m_pr = slot_prio[0];
                                for k in 1..slab_cap {
                                    if slot_prio[k] < m_pr {
                                        m_pr = slot_prio[k];
                                        m_sl = k;
                                    }
                                }
                                cur_min_slot = m_sl;
                                cur_min_prio = m_pr;
                            }
                        } else if prio > cur_min_prio {
                            let sl = cur_min_slot;
                            slot_raw_id[sl] = out_bi as i32;
                            slot_prio[sl] = prio;
                            slot_owner[sl] = item_idx as i32;
                            let mut m_sl = 0usize;
                            let mut m_pr = slot_prio[0];
                            for k in 1..slab_cap {
                                if slot_prio[k] < m_pr {
                                    m_pr = slot_prio[k];
                                    m_sl = k;
                                }
                            }
                            cur_min_slot = m_sl;
                            cur_min_prio = m_pr;
                        }
                    }

                    let copy_list: Vec<(usize, usize)> = slot_owner
                        .iter()
                        .enumerate()
                        .filter_map(|(sl, &own)| if own >= 0 { Some((sl, own as usize)) } else { None })
                        .collect();

                    copy_list.par_iter().for_each(|&(sl, item_idx)| {
                        let src_r = std::slice::from_raw_parts(
                            (p_span_ru as *const u16).add(item_idx * tn),
                            tn,
                        );
                        let src_f = std::slice::from_raw_parts(
                            (p_span_fu as *const u16).add(item_idx * tn),
                            tn,
                        );
                        let dst_r = std::slice::from_raw_parts_mut(
                            (p_ru_slab as *mut u16).add(sl * tn),
                            tn,
                        );
                        let dst_f = std::slice::from_raw_parts_mut(
                            (p_fu_slab as *mut u16).add(sl * tn),
                            tn,
                        );
                        dst_r.copy_from_slice(src_r);
                        dst_f.copy_from_slice(src_f);
                    });
                }
            });
            comp_secs = t_comp.elapsed().as_secs_f64();

            if let Some(jh) = bg_handle {
                let t_wait = Instant::now();
                let r = jh.join().expect("prefetch thread panicked");
                next_io_wait_secs = t_wait.elapsed().as_secs_f64();
                next_res = Some(r);
            }
        });

        let gb = (cur_pref.total_bytes as f64) / 1e9;
        eprintln!(
            "  [SF-SCAN {label}] 分块 {}/{n_spans} (有效原始={} 列, 派生={} 列, 顺序直读={gb:.1}GB): 预读等待={:.1}s (后台读={:.1}s) | 并行计算={comp_secs:.1}s | 累计={:.1}s",
            si + 1,
            cur_span.items.len(),
            cur_span.items.len() * 50,
            cur_wait_reported,
            cur_pref.read_secs,
            t_series.elapsed().as_secs_f64(),
        );

        if let Some(nr) = next_res {
            match nr {
                Ok(()) => {
                    io_wait_secs = next_io_wait_secs;
                    std::mem::swap(&mut cur_pref, &mut next_pref);
                }
                Err(e) => {
                    eprintln!(
                        "[sf_scan_store_double_buffered_top_u16] Prefetch span {} error: {e}",
                        si + 1
                    );
                    return -4;
                }
            }
        }
    }

    if direct_fill_all {
        used_slots = n_valid_raw;
        let prios: Vec<f64> = pool().install(|| {
            (0..n_valid_raw)
                .into_par_iter()
                .map(|out_bi| {
                    if let Some(mk) = must_keep {
                        if mk[out_bi] != 0 {
                            return 1.0e12 + ((n_valid_raw - out_bi) as f64);
                        }
                    }
                    let ic5_c = std::slice::from_raw_parts(
                        (p_ic5_b10 as *const f32).add(out_bi * 50 * n_g5),
                        50 * n_g5,
                    );
                    let vd5_c =
                        std::slice::from_raw_parts((p_vd5 as *const u16).add(out_bi * 50), 50);
                    score_raw_factor_50faces_g5(ic5_c, vd5_c, n_g5)
                })
                .collect()
        });
        for i in 0..n_valid_raw {
            slot_raw_id[i] = i as i32;
            slot_prio[i] = prios[i];
        }
    }

    if use_slab {
        if !out_slab_raw_ids_ptr.is_null() {
            let out_ids = std::slice::from_raw_parts_mut(out_slab_raw_ids_ptr, slab_cap);
            out_ids.copy_from_slice(&slot_raw_id);
        }
        if !out_slab_prios_ptr.is_null() {
            let out_prs = std::slice::from_raw_parts_mut(out_slab_prios_ptr, slab_cap);
            out_prs.copy_from_slice(&slot_prio);
        }
    }

    used_slots as i32
}
