//! 行业分类（申万一级, 从 industry_dump 提取的 CSV 读取, 纯 Rust）。
//! industry.bin 格式: u64 n + n × i16（与 codes 对齐, -1 = 未知）。
//! 生成: cargo run --release --features hdf5 --bin industry_dump -- <date> <out>

use std::io::{Read, Write};
use std::path::Path;

/// 从 CSV（code,ind）读入并重排为 codes 序
pub fn load_industry_from_csv(path: &Path, codes: &[String]) -> std::io::Result<Vec<i16>> {
    let s = std::fs::read_to_string(path)?;
    let mut map: std::collections::HashMap<String, i16> = std::collections::HashMap::new();
    for line in s.lines() {
        if line.is_empty() { continue; }
        let mut it = line.split(',');
        let code = it.next().unwrap_or("").trim();
        let ind = it.next().unwrap_or("").trim();
        if code.is_empty() || ind.is_empty() { continue; }
        if let Ok(v) = ind.parse::<i16>() {
            map.insert(code.to_string(), v);
        }
    }
    Ok(codes.iter().map(|c| map.get(c).copied().unwrap_or(-1)).collect())
}

/// 读 industry.bin（与 codes 对齐）
pub fn load_industry_bin(path: &Path, codes: &[String]) -> std::io::Result<Vec<i16>> {
    let mut data = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut data)?;
    if data.len() < 8 {
        return Ok(vec![-1; codes.len()]);
    }
    let n = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * 2;
        if off + 2 <= data.len() {
            out.push(i16::from_le_bytes(data[off..off + 2].try_into().unwrap()));
        } else {
            out.push(-1);
        }
    }
    Ok(out)
}

pub fn write_industry_bin(path: &Path, inds: &[i16]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(inds.len() as u64).to_le_bytes())?;
    for &v in inds {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
