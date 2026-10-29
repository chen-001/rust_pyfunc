//! Read-only input layer: whitelist before opening any trade file.
use crate::model::{Cell, Config, Stock};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
};

pub fn bad(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Gregorian date to days since 1970-01-01 (integer arithmetic, no timezone cast).
fn epoch_day(date: i64) -> io::Result<i64> {
    let mut y = date / 10000;
    let m = (date / 100) % 100;
    let d = date % 100;
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let lengths = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1900..=2200).contains(&y)
        || !(1..=12).contains(&m)
        || d < 1
        || d > lengths[(m - 1) as usize]
    {
        return Err(bad("invalid Gregorian date"));
    }
    y -= i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    Ok(era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468)
}

#[derive(Clone, Debug, Serialize)]
pub struct InputManifest {
    pub whitelist_path: String,
    pub whitelist_sha256: String,
    pub whitelist_size: usize,
    pub stock_count: usize,
    pub excluded_nonstock_files: usize,
    pub empty_stocks: usize,
    pub industry_source: String,
    pub industry_date: i64,
    pub industry_symbol_sha256: String,
    pub trade_files_opened: usize,
    pub test_subset: bool,
}

/// Strict symbol,pos parser. Never fall back to the raw security directory.
pub fn symbol_table(path: &Path) -> io::Result<(BTreeMap<String, usize>, String)> {
    let raw =
        std::fs::read(path).map_err(|e| bad(format!("stock table {}: {e}", path.display())))?;
    let mut reader = csv::Reader::from_reader(raw.as_slice());
    let header = reader.headers().map_err(|e| bad(e.to_string()))?.clone();
    let si = header
        .iter()
        .position(|x| x == "symbol")
        .ok_or_else(|| bad("symbol_map missing symbol column"))?;
    let pi = header
        .iter()
        .position(|x| x == "pos")
        .ok_or_else(|| bad("symbol_map missing pos column"))?;
    let mut table = BTreeMap::new();
    let mut positions = BTreeSet::new();
    for row in reader.records() {
        let row = row.map_err(|e| bad(e.to_string()))?;
        let code = row.get(si).unwrap_or("").trim();
        if code.len() != 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad(format!("invalid stock symbol {code:?}")));
        }
        let pos: usize = row
            .get(pi)
            .unwrap_or("")
            .parse()
            .map_err(|_| bad("invalid stock pos"))?;
        if table.insert(code.to_owned(), pos).is_some() || !positions.insert(pos) {
            return Err(bad("duplicate symbol or pos"));
        }
    }
    if table.is_empty() {
        return Err(bad("stock whitelist is empty"));
    }
    Ok((table, sha256(&raw)))
}

pub fn calendar(path: &Path, header: bool) -> io::Result<Vec<i64>> {
    let content = std::fs::read_to_string(path)?;
    let dates: Vec<i64> = content
        .lines()
        .skip(usize::from(header))
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<i64>()
                .map_err(|_| bad(format!("invalid calendar date {s:?}")))
        })
        .collect::<io::Result<_>>()?;
    if dates.is_empty() || dates.windows(2).any(|w| w[0] >= w[1]) {
        return Err(bad("calendar must be nonempty, strictly increasing"));
    }
    Ok(dates)
}

/// The HDF5 column axis is its own symbol table, never inferred from row order.
fn industry(config: &Config) -> io::Result<(BTreeMap<String, i32>, i64, String)> {
    let root = config.vars_dir.join("SzBa");
    let (symbols, hash) = symbol_table(&root.join("symbol_map.csv"))?;
    let dates = calendar(&root.join("calendar_map.csv"), true)?;
    let row = dates.binary_search(&config.date).map_err(|_| {
        bad(format!(
            "industry calendar missing exact date {}",
            config.date
        ))
    })?;
    let file = hdf5::File::open(root.join("industry.h5")).map_err(|e| bad(e.to_string()))?;
    let data = file.dataset("data").map_err(|e| bad(e.to_string()))?;
    let shape = data.shape();
    if shape.len() != 2 || row >= shape[0] || symbols.values().any(|&p| p >= shape[1]) {
        return Err(bad("industry axes exceed dataset shape"));
    }
    let values = data
        .read_slice_1d::<f64, _>((row, ..))
        .map_err(|e| bad(e.to_string()))?;
    let result = symbols
        .into_iter()
        .map(|(s, p)| {
            let v = values[p];
            (
                s,
                if v.is_finite() && v >= 1.0 && v <= 31.0 {
                    v as i32
                } else {
                    -1
                },
            )
        })
        .collect();
    Ok((result, dates[row], hash))
}

pub fn trade_tasks(
    config: &Config,
    whitelist: &BTreeMap<String, usize>,
) -> io::Result<(Vec<(String, PathBuf)>, usize)> {
    let dir = config
        .level2_root
        .join(config.date.to_string())
        .join("transaction");
    if !dir.is_dir() {
        return Err(bad(format!("missing trade directory {}", dir.display())));
    }
    let selected: BTreeSet<String> = match &config.codes {
        Some(codes) => {
            let set: BTreeSet<_> = codes.iter().cloned().collect();
            if set.is_empty()
                || set.len() != codes.len()
                || set.iter().any(|c| !whitelist.contains_key(c))
            {
                return Err(bad("test codes must be unique whitelist stocks"));
            }
            set
        }
        None => whitelist.keys().cloned().collect(),
    };
    let mut tasks = Vec::new();
    let mut excluded = 0;
    // Enumerate filenames only. Non-stocks are rejected before file open/read.
    let suffix = format!("_{}_transaction.csv", config.date);
    for item in std::fs::read_dir(&dir)? {
        let item = item?;
        let name = item.file_name();
        let name = name.to_string_lossy();
        if let Some(code) = name.strip_suffix(&suffix) {
            if !whitelist.contains_key(code) {
                excluded += 1;
                continue;
            }
            if selected.contains(code) {
                tasks.push((code.to_string(), item.path()));
            }
        }
    }
    tasks.sort_by(|a, b| a.0.cmp(&b.0));
    if tasks.is_empty() {
        return Err(bad("no whitelist stock trade files"));
    }
    if config.codes.is_some() && tasks.len() != selected.len() {
        return Err(bad("requested test stock trade file missing"));
    }
    Ok((tasks, excluded))
}

pub fn read_stock(code: &str, path: &Path, industry: i32, date: i64) -> io::Result<Stock> {
    let content =
        std::fs::read_to_string(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let mut lines = content.lines();
    let header = lines.next().ok_or_else(|| bad("empty trade file"))?;
    let names: Vec<_> = header.trim_end_matches('\r').split(',').collect();
    for (i, name) in [
        (0, "symbol"),
        (2, "date"),
        (4, "exchtime"),
        (7, "price"),
        (9, "turnover"),
        (10, "flag"),
    ] {
        if names.get(i) != Some(&name) {
            return Err(bad(format!("{}: unsupported trade header", path.display())));
        }
    }
    let mut events = Vec::new();
    let mut n_buy = 0;
    let mut n_sell = 0;
    let mut amount = 0.0;
    let mut minute_buy = vec![0.0; 237];
    let mut minute_sell = vec![0.0; 237];
    let mut first = vec![f64::NAN; 237];
    let mut close = vec![f64::NAN; 237];
    let mut first_ts = vec![(i64::MAX, f64::INFINITY); 237];
    let mut last_ts = vec![(i64::MIN, f64::NEG_INFINITY); 237];
    let expected_day = epoch_day(date)?;
    let mut iter = lines.enumerate().peekable();
    let mut skipped_foreign_date = false;
    let mut any_row = false;
    while let Some((line_no, line)) = iter.next() {
        if line.is_empty() {
            continue;
        }
        let mut fields = [""; 11];
        let mut count = 0;
        for (i, field) in line.split(',').take(11).enumerate() {
            fields[i] = field;
            count += 1;
        }
        let fail = || {
            bad(format!(
                "{}:{} malformed trade",
                path.display(),
                line_no + 2
            ))
        };
        if count < 11 {
            if iter.peek().is_none() {
                break;
            }
            return Err(fail());
        }
        // 2025 文件把整型列写成 "66.0"/"83.0" 浮点，按 f64 解析同时兼容。
        let flag: f64 = fields[10].parse().map_err(|_| fail())?;
        if flag != 66.0 && flag != 83.0 {
            continue;
        }
        let flag = flag as i32;
        // symbol：旧文件写不带前导零的数字（2552 对应 002552），按数值比。
        if fields[0].parse::<i64>().map_err(|_| fail())?
            != code.parse::<i64>().map_err(|_| fail())?
        {
            return Err(bad(format!(
                "{}:{} symbol mismatch",
                path.display(),
                line_no + 2
            )));
        }
        // 行内 date：个别文件混入了其他交易日的整段记录（如 600476_20160405
        // 含 14752 行 date=20160401 且 exchtime 也是 4月1日），那些行属于别天，
        // 直接跳过；一行都不剩说明整个文件放错日期，才报整只票错。
        if fields[2].parse::<i64>().map_err(|_| fail())? != date {
            skipped_foreign_date = true;
            continue;
        }
        let ts: i64 = fields[4]
            .parse::<f64>()
            .map_err(|_| fail())? as i64;
        let local_ts = ts.checked_add(28_800_000_000).ok_or_else(fail)?;
        if local_ts.div_euclid(86_400_000_000) != expected_day {
            return Err(bad(format!(
                "{}:{} timestamp date mismatch",
                path.display(),
                line_no + 2
            )));
        }
        let clock = local_ts.rem_euclid(86_400_000_000);
        let offset = clock - 34_200_000_000;
        let minute = if (0..7_200_000_000).contains(&offset) {
            (offset / 60_000_000) as usize
        } else if (12_600_000_000..19_620_000_000).contains(&offset) {
            120 + ((offset - 12_600_000_000) / 60_000_000) as usize
        } else {
            continue;
        };
        let price: f64 = fields[7].parse().map_err(|_| fail())?;
        let value: f64 = fields[9].parse().map_err(|_| fail())?;
        if !price.is_finite() || price <= 0.0 || !value.is_finite() || value < 0.0 {
            return Err(fail());
        }
        let buy = flag == 66;
        events.push(((offset / 10_000) as u32, buy));
        amount += value;
        if buy {
            n_buy += 1;
            minute_buy[minute] += value;
        } else {
            n_sell += 1;
            minute_sell[minute] += value;
        }
        // 2015 等旧文件 index 是 "0.0" 浮点写法；f64 兼容两种格式且非负下序一致。
        let index: f64 = fields[6].parse().map_err(|_| fail())?;
        if !index.is_finite() {
            return Err(fail());
        }
        let key = (ts, index);
        if key < first_ts[minute] {
            first_ts[minute] = key;
            first[minute] = price;
        }
        if key > last_ts[minute] {
            last_ts[minute] = key;
            close[minute] = price;
        }
    }
    events.sort_unstable_by_key(|x| x.0);
    let mut cells: Vec<Cell> = Vec::new();
    for (slot, buy) in events {
        if cells.last().is_none_or(|c| c.slot != slot) {
            cells.push(Cell {
                slot,
                buy: 0.0,
                sell: 0.0,
            });
        }
        let c = cells.last_mut().unwrap();
        if buy {
            c.buy += 1.0;
        } else {
            c.sell += 1.0;
        }
    }
    let minute_ret = close
        .iter()
        .zip(&first)
        .map(|(&last, &start)| (last / start).ln())
        .collect();
    Ok(Stock {
        code: code.into(),
        industry,
        cells,
        n_buy,
        n_sell,
        amount,
        minute_buy,
        minute_sell,
        minute_ret,
        minute_close: close,
    })
}

pub fn load(config: &Config) -> io::Result<(Vec<Stock>, InputManifest, Vec<i64>)> {
    let whitepath = config.basic_info_dir.join("symbol_map.csv");
    let (whitelist, hash) = symbol_table(&whitepath)?;
    let dates = calendar(&config.basic_info_dir.join("calendar.csv"), false)?;
    if dates.binary_search(&config.date).is_err() {
        return Err(bad("requested date is not in trading calendar"));
    }
    let (tasks, excluded) = trade_tasks(config, &whitelist)?;
    let (ind, ind_date, indhash) = industry(config)?;
    let stocks: Vec<Stock> = tasks
        .par_iter()
        .map(|(c, p)| read_stock(c, p, *ind.get(c).unwrap_or(&-1), config.date))
        .collect::<io::Result<_>>()?;
    let manifest = InputManifest {
        whitelist_path: whitepath.display().to_string(),
        whitelist_sha256: hash,
        whitelist_size: whitelist.len(),
        stock_count: stocks.len(),
        excluded_nonstock_files: excluded,
        empty_stocks: stocks.iter().filter(|s| s.cells.is_empty()).count(),
        trade_files_opened: stocks.len(),
        industry_source: config
            .vars_dir
            .join("SzBa/industry.h5")
            .display()
            .to_string(),
        industry_date: ind_date,
        industry_symbol_sha256: indhash,
        test_subset: config.codes.is_some(),
    };
    Ok((stocks, manifest, dates))
}
