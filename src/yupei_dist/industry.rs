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

/// 读 industry.bin 原始数组并转为 code→ind 映射（industry.bin 无 code 列,
/// 顺序对应 /ssd_data/data/basic_info/symbol_map.csv 的 pos 列; 这里返回 (code, ind)
/// 依赖调用方知道 symbol_map 顺序 —— 简化: 直接返回按序向量并附 code 列表）。
/// 本函数返回 HashMap: code -> ind（-1 未知）。
pub fn load_industry_all(path: &Path) -> std::io::Result<std::collections::HashMap<String, i16>> {
    let mut data = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut data)?;
    if data.len() < 8 {
        return Ok(std::collections::HashMap::new());
    }
    let n = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let vals: Vec<i16> = (0..n)
        .map(|i| {
            let off = 8 + i * 2;
            if off + 2 <= data.len() {
                i16::from_le_bytes(data[off..off + 2].try_into().unwrap())
            } else {
                -1
            }
        })
        .collect();
    // symbol_map.csv: pos → code
    let sym = std::fs::read_to_string("/ssd_data/data/basic_info/symbol_map.csv")?;
    let mut map = std::collections::HashMap::new();
    for (pos, line) in sym.lines().skip(1).enumerate() {
        let code = line.split(',').next().unwrap_or("").trim();
        if !code.is_empty() {
            map.insert(code.to_string(), vals.get(pos).copied().unwrap_or(-1));
        }
    }
    Ok(map)
}

pub fn write_industry_bin(path: &Path, inds: &[i16]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(inds.len() as u64).to_le_bytes())?;
    for &v in inds {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
