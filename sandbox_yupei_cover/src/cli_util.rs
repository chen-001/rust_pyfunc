//! 共享小工具：日起点微秒、名单解析、股票范围过滤。
//!
//! 口径来源：experiments/yupei2/pair_spec.txt 第 1/3 节 + kernel_spec.txt 第 1 节。
//! 桶基准 = 当日 09:30 的微秒差（下午已由 fast_csv_reader 整体前移 90 分钟），
//! 与 sandbox_yupei_dist/src/cli_util.rs 的 day_start_us 完全一致。

pub const BUCKET_US: i64 = 10_000;
/// 参与计算的桶数上限：生产实现只算满块的 347*4096 = 1421312 个桶，丢掉末尾的 688 个。
pub const N_BUCKETS: usize = 1_421_312;

pub fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as i64;
    era * 146097 + doe - 719468
}

/// 日起点（UTC 零点 + 9.5h，与 fast_csv_reader 的 +8h 偏移配合，9:30 开盘 = 0）
pub fn day_start_us(date: i64) -> i64 {
    days_from_civil(date / 10000, (date / 100 % 100) as u64, (date % 100) as u64)
        * 86400 * 1_000_000
        + (9 * 3600 + 30 * 60) * 1_000_000
}

pub fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).map(|i| args[i + 1].clone())
}

/// 逐笔文件是否存在（与 fast_csv_reader::resolve_stock_path 同一套根目录规则）。
pub fn trade_file_exists(code: &str, date: i64) -> bool {
    let filename = format!("{}_{}_transaction.csv", code, date);
    if let Ok(env_path) = std::env::var("RUST_PYFUNC_LEVEL2_PATH") {
        return std::path::Path::new(&env_path)
            .join(date.to_string())
            .join("transaction")
            .join(&filename)
            .is_file();
    }
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        if std::path::Path::new(root)
            .join(date.to_string())
            .join("transaction")
            .join(&filename)
            .is_file()
        {
            return true;
        }
    }
    false
}

/// 股票白名单（symbol_map.csv 的 symbol 列）。
///
/// 原始 transaction 目录里同时可能有 ETF、LOF、债券、可转债，"六位数字文件名"筛不掉它们。
/// 白名单必须在读逐笔之前生效，候选配对两端与 codes.txt 都必须来自它。
/// 读表规则：按列名找 symbol 列，不按位置猜（表头是 symbol,pos，但换一版就可能变）。
/// 读不到文件、缺 symbol 列、或一行有效代码都没有，一律报错——不允许退回"全都算"。
pub struct StockWhitelist {
    codes: std::collections::HashSet<String>,
    pub path: String,
    pub n_codes: usize,
}

impl StockWhitelist {
    pub fn contains(&self, code: &str) -> bool {
        self.codes.contains(code)
    }
}

/// 读白名单。默认 /ssd_data/data/basic_info/symbol_map.csv；目录可用
/// RUST_PYFUNC_BASIC_INFO_DIR 覆盖（与项目其他模块约定一致），整条路径用 --symbol-map 指定。
pub fn load_stock_whitelist(path_override: Option<&str>) -> std::io::Result<StockWhitelist> {
    use std::io::{Error, ErrorKind};
    let path = match path_override {
        Some(p) => p.to_string(),
        None => {
            let dir = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR")
                .unwrap_or_else(|_| "/ssd_data/data/basic_info".to_string());
            format!("{}/symbol_map.csv", dir)
        }
    };
    let content = std::fs::read_to_string(&path).map_err(|e| {
        Error::new(
            ErrorKind::NotFound,
            format!("读不到股票白名单 {}: {}（不能用「全都算」兜底）", path, e),
        )
    })?;
    let mut lines = content.lines();
    let header = lines.next().ok_or_else(|| {
        Error::new(ErrorKind::InvalidData, format!("股票白名单 {} 是空文件", path))
    })?;
    let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    let sym_idx = cols.iter().position(|c| *c == "symbol").ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            format!("股票白名单 {} 缺少 symbol 列（表头是 {:?}）", path, cols),
        )
    })?;
    let mut codes = std::collections::HashSet::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let field = line.split(',').nth(sym_idx).unwrap_or("").trim();
        if field.is_empty() || field.len() > 6 || !field.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("股票白名单 {} 的 symbol 列含无效代码 {:?}", path, field),
            ));
        }
        codes.insert(format!("{:0>6}", field));
    }
    if codes.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("股票白名单 {} 解析后为空", path),
        ));
    }
    let n = codes.len();
    Ok(StockWhitelist { codes, path, n_codes: n })
}

/// 读备份名单 codes.txt（每行一个 6 位代码），只保留当日 transaction 目录下确实有文件的，
/// 保留原顺序（原下标用于回查备份矩阵的行列号，所以下标映射必须留住）。
/// whitelist 给定时，白名单外的代码在读逐笔之前就被跳过。
pub struct Universe {
    /// 过滤后的代码（计算用行序）
    pub codes: Vec<String>,
    /// 过滤后第 i 只股票在备份 codes.txt 里的原下标
    pub backup_idx: Vec<usize>,
    /// 备份名单里被丢掉的代码数（当日无逐笔文件 + 白名单外）
    pub dropped: usize,
    /// 其中因为不在股票白名单里被丢掉的
    pub dropped_not_stock: usize,
}

pub fn load_universe(
    codes_txt: &str,
    date: i64,
    need_file: bool,
    whitelist: Option<&StockWhitelist>,
) -> std::io::Result<Universe> {
    let content = std::fs::read_to_string(codes_txt)?;
    let mut codes = Vec::new();
    let mut backup_idx = Vec::new();
    let mut dropped = 0usize;
    let mut dropped_not_stock = 0usize;
    for (lineno, line) in content.lines().enumerate() {
        let c = line.trim();
        if c.is_empty() {
            continue;
        }
        if !c.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        if let Some(w) = whitelist {
            if !w.contains(c) {
                dropped += 1;
                dropped_not_stock += 1;
                continue;
            }
        }
        if need_file && !trade_file_exists(c, date) {
            dropped += 1;
            continue;
        }
        codes.push(c.to_string());
        // lineno 是文件行号；备份文件每行一个代码，行号即原下标
        backup_idx.push(lineno);
    }
    Ok(Universe { codes, backup_idx, dropped, dropped_not_stock })
}
