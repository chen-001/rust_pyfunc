//! yupei_dist 备份读写层（从 sandbox_yupei_dist/src/matrix_store.rs 迁入主项目）。
//!
//! 备份目录结构（与 sandbox 完全同格式, 双向兼容）:
//!   <outdir>/<date>/codes.txt         代码列表（每行一个, 与矩阵行序一致）
//!   <outdir>/<date>/stats.bin         每股统计量（u64 n + n×14 个 f64, 小端）
//!   <outdir>/<date>/mats/<name>.bin   N×N f32 有向矩阵（行主序）
//!   <outdir>/<date>/factors.bin       N×F f32 因子值（行主序）
//!   <outdir>/<date>/names.txt         因子名（每行一个）
//!   <outdir>/<date>/meta.json         目录信息（代码/因子名/矩阵名清单）
//!   <outdir>/<date>/report.txt        报告
//!   <outdir>/<date>/industry.bin      行业码（u64 n + n×i16, 全市场 symbol_map 序）
//!
//! Python 入口:
//!   rp.yupei_dist_backup_matrices(outdir, date)              读逐笔→37矩阵→写备份
//!   rp.yupei_dist_backup_factors(outdir, date, prev_date)    读备份→2761因子→写盘
//!   rp.yupei_dist_backup_read_matrix(outdir, date, name)     读单个矩阵 (N,N) f32
//!   rp.yupei_dist_backup_read_stats(outdir, date)            (codes, N×14 统计量)
//!   rp.yupei_dist_backup_read_factors(outdir, date)          (codes, names, N×F)
//!   rp.yupei_dist_backup_verify(outdir, date)                缺失矩阵清单（空=完整）
//!   rp.yupei_dist_backup_matrix_names()                      37 个矩阵名
//!
//! 注: compute_yupei_dist_full_with_prev 的前一日矩阵加载已按本格式读取
//! （{BACKUP_DIR}/{prev_date}/mats/*.bin）, 本模块补齐写入侧与通用读取侧。

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use super::indicator_ctx::{MatrixSet, PrevMats};
use super::industry;
use super::matrix_stage::{StockStats, MATRIX_SPECS, N_MATRICES};

// ---------------------------------------------------------------------------
// 纯 Rust 读写层（sandbox 逐函数迁移）
// ---------------------------------------------------------------------------

pub fn write_codes(dir: &Path, codes: &[String]) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(dir.join("codes.txt"))?);
    for c in codes {
        writeln!(f, "{c}")?;
    }
    Ok(())
}

pub fn read_codes(dir: &Path) -> std::io::Result<Vec<String>> {
    let s = std::fs::read_to_string(dir.join("codes.txt"))?;
    Ok(s.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

/// stats.bin: u64 n + n×14 个 f64（小端; 14 字段序同 StockStats 声明序）
pub fn write_stats(dir: &Path, stats: &[StockStats]) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(dir.join("stats.bin"))?);
    f.write_all(&(stats.len() as u64).to_le_bytes())?;
    let mut buf = [0u8; 8];
    for s in stats {
        for v in [
            s.n_trades, s.amount, s.total_vol, s.imb, s.ret, s.vol30, s.vwap, s.q95u,
            s.sum_w_cnt, s.sum_w_vol, s.sum_w_logvol, s.sum_w_flow, s.sum_w_urg, s.sum_w_ext,
        ] {
            buf.copy_from_slice(&v.to_le_bytes());
            f.write_all(&buf)?;
        }
    }
    Ok(())
}

pub fn read_stats(dir: &Path) -> std::io::Result<Vec<StockStats>> {
    let mut data = Vec::new();
    File::open(dir.join("stats.bin"))?.read_to_end(&mut data)?;
    if data.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "stats.bin too short",
        ));
    }
    let n = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if data.len() < 8 + n * 14 * 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "stats.bin length mismatch",
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = 8 + i * 14 * 8;
        let mut vals = [0.0f64; 14];
        for (j, v) in vals.iter_mut().enumerate() {
            let off = base + j * 8;
            *v = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        }
        out.push(StockStats {
            n_trades: vals[0], amount: vals[1], total_vol: vals[2], imb: vals[3],
            ret: vals[4], vol30: vals[5], vwap: vals[6], q95u: vals[7],
            sum_w_cnt: vals[8], sum_w_vol: vals[9], sum_w_logvol: vals[10],
            sum_w_flow: vals[11], sum_w_urg: vals[12], sum_w_ext: vals[13],
        });
    }
    Ok(out)
}

/// 写单个矩阵（mats/<name>.bin, f32 行主序）
pub fn write_matrix(dir: &Path, name: &str, mat: &[f32]) -> std::io::Result<()> {
    let p = dir.join("mats");
    std::fs::create_dir_all(&p)?;
    let mut f = BufWriter::new(File::create(p.join(format!("{name}.bin")))?);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(mat.as_ptr() as *const u8, mat.len() * 4) };
    f.write_all(bytes)?;
    Ok(())
}

/// 读矩阵（整读入内存）
pub fn read_matrix(dir: &Path, name: &str) -> std::io::Result<Vec<f32>> {
    let mut data = Vec::new();
    File::open(dir.join("mats").join(format!("{name}.bin")))?.read_to_end(&mut data)?;
    debug_assert!(data.len() % 4 == 0);
    Ok(unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4).to_vec()
    })
}

/// mmap 读矩阵（零拷贝; 供 Rust 侧高频访问, Python 侧请用 read_matrix）
pub fn mmap_matrix(dir: &Path, name: &str) -> std::io::Result<memmap2::Mmap> {
    let f = File::open(dir.join("mats").join(format!("{name}.bin")))?;
    unsafe { memmap2::Mmap::map(&f) }
}

/// 备份目录中应存在的全部矩阵名
pub fn expected_matrix_names() -> Vec<&'static str> {
    MATRIX_SPECS.iter().map(|s| s.name).collect()
}

/// 写因子备份: factors.bin + names.txt + meta.json
pub fn write_factors(
    dir: &Path,
    codes: &[String],
    names: &[String],
    vals: &[f32],
) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(dir.join("factors.bin"))?);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
    f.write_all(bytes)?;
    let mut nf = BufWriter::new(File::create(dir.join("names.txt"))?);
    for nm in names {
        writeln!(nf, "{nm}")?;
    }
    write_meta_json(dir, codes, names, vals)
}

fn write_meta_json(
    dir: &Path,
    codes: &[String],
    names: &[String],
    vals: &[f32],
) -> std::io::Result<()> {
    let n = codes.len();
    let nf = if n > 0 { vals.len() / n } else { 0 };
    let meta = serde_json::json!({
        "n_stocks": n,
        "n_factors": nf,
        "codes": codes,
        "names": names,
        "n_matrices": N_MATRICES,
        "matrices": expected_matrix_names(),
    });
    std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap())?;
    Ok(())
}

pub fn read_factors(dir: &Path) -> std::io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    let codes = read_codes(dir)?;
    let names: Vec<String> = std::fs::read_to_string(dir.join("names.txt"))?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let mut data = Vec::new();
    File::open(dir.join("factors.bin"))?.read_to_end(&mut data)?;
    let vals =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4).to_vec() };
    Ok((codes, names, vals))
}

// ---------------------------------------------------------------------------
// 备份 → 正式库内存结构 的桥接
// ---------------------------------------------------------------------------

/// 从备份目录加载一个日期的完整数据为内存 MatrixSet（供 IndicatorCtx 使用）。
/// industry.bin 按全市场 symbol_map 顺序存储: 读入后按 codes 重排（缺失 -1）。
pub fn load_backup_set(outdir: &str, date: i64, load_mats: bool) -> std::io::Result<MatrixSet> {
    let d = Path::new(outdir).join(date.to_string());
    let codes = read_codes(&d)?;
    let stats = read_stats(&d)?;
    let mut mats: HashMap<String, Vec<f32>> = HashMap::new();
    if load_mats {
        for spec in MATRIX_SPECS.iter() {
            let p = d.join("mats").join(format!("{}.bin", spec.name));
            if p.exists() {
                mats.insert(spec.name.to_string(), read_matrix(&d, spec.name)?);
            }
        }
    }
    let n = codes.len();
    let industry: Option<Vec<i16>> = industry::load_industry_all(&d.join("industry.bin"))
        .ok()
        .map(|all: HashMap<String, i16>| {
            codes.iter().map(|c| all.get(c).copied().unwrap_or(-1)).collect()
        })
        .filter(|v: &Vec<i16>| v.len() == n);
    let _ = date; // MatrixSet 无日期字段; 日期由目录结构承载
    Ok(MatrixSet { n, codes, stats, mats, industry })
}

/// 从备份目录加载前一交易日矩阵（PrevMats; 只读 mats, 供 dyn_* 跨日因子）。
pub fn load_prev_mats(outdir: &str, date: i64) -> std::io::Result<PrevMats> {
    let d = Path::new(outdir).join(date.to_string());
    let codes = read_codes(&d)?;
    let mut mats: HashMap<String, Vec<f32>> = HashMap::new();
    for spec in MATRIX_SPECS.iter() {
        let p = d.join("mats").join(format!("{}.bin", spec.name));
        if p.exists() {
            mats.insert(spec.name.to_string(), read_matrix(&d, spec.name)?);
        }
    }
    Ok(PrevMats { n: codes.len(), codes, mats })
}

// ---------------------------------------------------------------------------
// Python 入口
// ---------------------------------------------------------------------------

/// 读全市场逐笔 → 37 矩阵 + 统计量 → 写入 <outdir>/<date>/（codes/stats/mats）。
/// 参数口径与生产一致（全市场 universe, MIN_TRADES=200）。
/// 返回 (n_stocks, n_matrices)。
#[pyfunction]
pub fn yupei_dist_backup_matrices(outdir: String, date: i64) -> PyResult<(usize, usize)> {
    let (codes, n, stats, hm, _industry) =
        super::compute::compute_yupei_dist_partial(date)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))?;
    let dir = Path::new(&outdir).join(date.to_string());
    std::fs::create_dir_all(&dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))?;
    write_codes(&dir, &codes).map_err(io_err)?;
    write_stats(&dir, &stats).map_err(io_err)?;
    for spec in MATRIX_SPECS.iter() {
        if let Some(m) = hm.get(spec.name) {
            write_matrix(&dir, spec.name, m).map_err(io_err)?;
        }
    }
    Ok((n, hm.len()))
}

/// 读备份 → 运行全部 28 个指标模块（2761 因子）→ 写 factors.bin/names.txt/meta.json/report.txt。
/// prev_date 给定时从同备份目录加载前一日矩阵（dyn_* 跨日因子有值）。
/// 返回 (n_stocks, n_factors)。
#[pyfunction]
pub fn yupei_dist_backup_factors(
    outdir: String,
    date: i64,
    prev_date: Option<i64>,
) -> PyResult<(usize, usize)> {
    let set = load_backup_set(&outdir, date, true).map_err(io_err)?;
    let prev = prev_date.map(|pd| {
        let pset = load_prev_mats(&outdir, pd).ok()?;
        Some(super::indicator_ctx::PrevDay::new(Box::leak(Box::new(pset))))
    });
    let ctx = super::indicator_ctx::IndicatorCtx::new(&set, prev.flatten());
    let n = set.n;
    let codes = set.codes.clone();
    let vals =
        super::compute::run_indicators(&ctx, n, &codes).map_err(io_err)?;
    let names = super::compute::yupei_dist_names();
    let dir = Path::new(&outdir).join(date.to_string());
    write_factors(&dir, &codes, &names, &vals).map_err(io_err)?;
    let report = format!(
        "date={date} n_stocks={n} n_factors={} source=backup prev_date={:?}\n",
        names.len(),
        prev_date
    );
    std::fs::write(dir.join("report.txt"), report).map_err(io_err)?;
    Ok((n, names.len()))
}

/// 读单个矩阵为 (N,N) f32 numpy 数组。
#[pyfunction]
pub fn yupei_dist_backup_read_matrix(
    py: Python,
    outdir: String,
    date: i64,
    name: String,
) -> PyResult<Py<PyArray2<f32>>> {
    let d = Path::new(&outdir).join(date.to_string());
    let codes = read_codes(&d).map_err(io_err)?;
    let data = read_matrix(&d, &name).map_err(io_err)?;
    let n = codes.len();
    let arr = ndarray::Array2::from_shape_vec((n, n), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:?}")))?;
    Ok(arr.into_pyarray(py).to_owned())
}

/// 读每股统计量: (codes, N×14 f64)。列序同 StockStats 声明序
/// （n_trades, amount, total_vol, imb, ret, vol30, vwap, q95u,
///   sum_w_cnt, sum_w_vol, sum_w_logvol, sum_w_flow, sum_w_urg, sum_w_ext）。
#[pyfunction]
pub fn yupei_dist_backup_read_stats(
    py: Python,
    outdir: String,
    date: i64,
) -> PyResult<(Vec<String>, Py<PyArray2<f64>>)> {
    let d = Path::new(&outdir).join(date.to_string());
    let codes = read_codes(&d).map_err(io_err)?;
    let stats = read_stats(&d).map_err(io_err)?;
    let n = stats.len();
    let mut flat = Vec::with_capacity(n * 14);
    for s in &stats {
        flat.extend([
            s.n_trades, s.amount, s.total_vol, s.imb, s.ret, s.vol30, s.vwap, s.q95u,
            s.sum_w_cnt, s.sum_w_vol, s.sum_w_logvol, s.sum_w_flow, s.sum_w_urg, s.sum_w_ext,
        ]);
    }
    let arr = ndarray::Array2::from_shape_vec((n, 14), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:?}")))?;
    Ok((codes, arr.into_pyarray(py).to_owned()))
}

/// 读因子备份: (codes, names, N×F f32)。
#[pyfunction]
pub fn yupei_dist_backup_read_factors(
    py: Python,
    outdir: String,
    date: i64,
) -> PyResult<(Vec<String>, Vec<String>, Py<PyArray2<f32>>)> {
    let d = Path::new(&outdir).join(date.to_string());
    let (codes, names, vals) = read_factors(&d).map_err(io_err)?;
    let nf = names.len();
    let n = codes.len();
    let arr = ndarray::Array2::from_shape_vec((n, nf), vals)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:?}")))?;
    Ok((codes, names, arr.into_pyarray(py).to_owned()))
}

/// 校验备份完整性: 返回缺失的矩阵名清单（空 = 完整）。
#[pyfunction]
pub fn yupei_dist_backup_verify(outdir: String, date: i64) -> Vec<String> {
    let d = Path::new(&outdir).join(date.to_string());
    expected_matrix_names()
        .into_iter()
        .filter(|name| !d.join("mats").join(format!("{name}.bin")).exists())
        .map(|s| s.to_string())
        .collect()
}

/// 37 个矩阵名（与 MATRIX_SPECS 顺序一致）。
#[pyfunction]
pub fn yupei_dist_backup_matrix_names() -> Vec<&'static str> {
    expected_matrix_names()
}

fn io_err(e: std::io::Error) -> PyErr {
    pyo3::exceptions::PyIOError::new_err(format!("{e:?}"))
}

// 注: 读写格式的正确性由 tests/test_sandbox_migration.py 用真实 sandbox 备份校验
// （cargo test --lib 因历史遗留测试代码编译失败, 不在本模块添加 cfg(test)）。
