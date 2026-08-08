//! 矩阵/统计量/因子的二进制备份读写（纯 Rust）。
//! 备份目录结构:
//!   <outdir>/<date>/codes.txt         代码列表（每行一个, 与矩阵行序一致）
//!   <outdir>/<date>/stats.bin         每股统计量（f64 数组）
//!   <outdir>/<date>/mats/<name>.bin   N×N f32 有向矩阵（行主序）
//!   <outdir>/<date>/factors.bin       N×F f32 因子值（行主序）
//!   <outdir>/<date>/names.txt         因子名（每行一个）
//!   <outdir>/<date>/report.txt        各阶段/各模块耗时

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use crate::matrix_stage::{StockStats, MATRIX_SPECS, N_MATRICES};

pub fn write_codes(dir: &Path, codes: &[String]) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(dir.join("codes.txt"))?);
    for c in codes {
        writeln!(f, "{c}")?;
    }
    Ok(())
}

pub fn write_stats(dir: &Path, stats: &[StockStats]) -> std::io::Result<()> {
    // 格式: u64 n, 然后 n × 15 个 f64
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
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "stats.bin too short"));
    }
    let n = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
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

pub fn write_matrix(dir: &Path, name: &str, mat: &[f32]) -> std::io::Result<()> {
    let p = dir.join("mats");
    std::fs::create_dir_all(&p)?;
    let mut f = BufWriter::new(File::create(p.join(format!("{name}.bin")))?);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(mat.as_ptr() as *const u8, mat.len() * 4)
    };
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

/// mmap 读矩阵（零拷贝, 供指标阶段）
pub fn mmap_matrix(dir: &Path, name: &str) -> std::io::Result<memmap2::Mmap> {
    let f = File::open(dir.join("mats").join(format!("{name}.bin")))?;
    unsafe { memmap2::Mmap::map(&f) }
}

pub fn read_codes(dir: &Path) -> std::io::Result<Vec<String>> {
    let s = std::fs::read_to_string(dir.join("codes.txt"))?;
    Ok(s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
}

/// 备份目录中应存在的全部矩阵名
pub fn expected_matrix_names() -> Vec<&'static str> {
    MATRIX_SPECS.iter().map(|s| s.name).collect()
}

pub fn write_factors(dir: &Path, codes: &[String], names: &[String], vals: &[f32]) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(dir.join("factors.bin"))?);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4)
    };
    f.write_all(bytes)?;
    let mut nf = BufWriter::new(File::create(dir.join("names.txt"))?);
    for nm in names {
        writeln!(nf, "{nm}")?;
    }
    // 目录信息
    write_meta_json(dir, codes, names, vals)
}

fn write_meta_json(dir: &Path, codes: &[String], names: &[String], vals: &[f32]) -> std::io::Result<()> {
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
        .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let mut data = Vec::new();
    File::open(dir.join("factors.bin"))?.read_to_end(&mut data)?;
    let vals = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4).to_vec() };
    Ok((codes, names, vals))
}
