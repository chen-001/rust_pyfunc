//! 共享 CLI 工具: 代码枚举（按文件大小）、day_start_us 计算。

use std::collections::BTreeSet;

/// 枚举当日全市场代码（按文件大小降序, 用于 universe 截断; 同大小按代码序）
pub fn list_codes_by_size(date: i64) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/transaction");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut v: Vec<(u64, String)> = Vec::new();
            for e in entries.flatten() {
                if let Some(code) = e.file_name().to_str().and_then(|n| n.split('_').next()) {
                    if code.bytes().all(|b| b.is_ascii_digit()) {
                        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                        v.push((size, code.to_string()));
                    }
                }
            }
            v.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let codes: Vec<String> = v.into_iter().map(|(_, c)| c).collect();
            if !codes.is_empty() {
                return codes;
            }
        }
    }
    Vec::new()
}

/// 仅按代码排序（无 universe 截断时用）
pub fn list_codes(date: i64) -> Vec<String> {
    let mut set = BTreeSet::new();
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/transaction");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if let Some(code) = e.file_name().to_str().and_then(|n| n.split('_').next()) {
                    if code.bytes().all(|b| b.is_ascii_digit()) {
                        set.insert(code.to_string());
                    }
                }
            }
        }
    }
    set.into_iter().collect()
}

pub fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as i64;
    era * 146097 + doe - 719468
}

/// 日起点（UTC 零点 + 9.5h, 与 fast_csv_reader 的 +8h 偏移配合, 9:30 开盘 = 0）
pub fn day_start_us(date: i64) -> i64 {
    days_from_civil(date / 10000, (date / 100 % 100) as u64, (date % 100) as u64)
        * 86400 * 1_000_000
        + (9 * 3600 + 30 * 60) * 1_000_000
}
