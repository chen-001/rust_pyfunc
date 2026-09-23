//! hm95 的 Level2 → 分钟字段还原。分钟边界和旧 H5 一致：逐笔、盘口均归入
//! [分钟起点, 下一分钟起点)，但剔除 09:30:00 与 13:00:00 的起点记录。

use rayon::prelude::*;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

pub const FIELDS: [&str; 22] = [
    "act_buy_amount_sum",
    "act_sell_amount_sum",
    "amount",
    "act_buy_count_sum",
    "act_sell_count_sum",
    "up_tick_count",
    "down_tick_count",
    "volume",
    "turnover",
    "high",
    "low",
    "close",
    "spread_over_tick_size_mean",
    "bid_vol1",
    "bid_vol5",
    "ask_vol1",
    "bid_size_10_mean",
    "ask_size_10_mean",
    "ask_vwap10",
    "bid_vwap10",
    "ask_prc1",
    "bid_prc1",
];
pub const BARS: usize = 237; // 原分钟读取每个交易日最后 3 根置 NaN
const DAYS: usize = 5;

pub struct Meta {
    pub codes: Vec<String>,
    pub dates: Vec<i64>,
}

static META: LazyLock<Mutex<HashMap<PathBuf, Arc<Meta>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn meta() -> io::Result<Arc<Meta>> {
    let dir = crate::data_paths::basic_info_path("");
    let mut cache = META.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(meta) = cache.get(&dir) {
        return Ok(meta.clone());
    }
    let mut codes = Vec::new();
    for line in std::fs::read_to_string(dir.join("symbol_map.csv"))?.lines() {
        if let Some((code, pos)) = line.trim().split_once(',') {
            if pos.parse::<usize>().is_ok() {
                codes.push(code.to_string());
            }
        }
    }
    let dates = std::fs::read_to_string(dir.join("calendar.csv"))?
        .lines()
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    let result = Arc::new(Meta { codes, dates });
    cache.insert(dir, result.clone());
    Ok(result)
}

// time 为 HHMMSSmmm。两段连续交易时段各 120 根，分钟起点的成交归下一根。
fn bar(time: i64) -> Option<usize> {
    let h = time / 10_000_000;
    let m = time / 100_000 % 100;
    let s = time / 1_000 % 100;
    let ms = time % 1_000;
    let day_ms = ((h * 60 + m) * 60 + s) * 1_000 + ms;
    for (start, offset) in [((9 * 60 + 30) * 60_000, 0), (13 * 60 * 60_000, 120)] {
        let d = day_ms - start;
        if d > 0 && d < 120 * 60_000 {
            return Some(offset + (d / 60_000) as usize);
        }
    }
    None
}

#[inline]
fn number(bytes: &[u8]) -> f64 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(f64::NAN)
}

#[inline]
fn integer(bytes: &[u8]) -> i64 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(0)
}

fn each_csv_row(path: &Path, mut process: impl FnMut(&[&[u8]])) -> io::Result<()> {
    let data = std::fs::read(path)?;
    for line in data.split(|&x| x == b'\n').skip(1) {
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&[u8]> = line.split(|&x| x == b',').collect();
        process(&fields);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Trade {
    time: i64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i64,
    order: usize,
}

#[derive(Clone, Copy)]
struct Quote {
    time: i64,
    ask_prc1: f64,
    bid_prc1: f64,
    ask_vol1: f64,
    bid_vol1: f64,
    bid_vol5: f64,
    ask_size10: f64,
    bid_size10: f64,
    ask_vwap10: f64,
    bid_vwap10: f64,
    order: usize,
}

const LAST_FIELDS: [usize; 7] = [13, 14, 15, 18, 19, 20, 21];

fn transaction_path(root: &Path, code: &str, date: i64) -> PathBuf {
    root.join(date.to_string())
        .join("transaction")
        .join(format!("{code}_{date}_transaction.csv"))
}

fn stock_day(
    root: &Path,
    code: &str,
    date: i64,
    carry: &mut [f64; 7],
    read_trades: bool,
) -> io::Result<[[f64; 22]; BARS]> {
    let mut out = [[f64::NAN; 22]; BARS];
    let trade_path = transaction_path(root, code, date);
    let market_path = root
        .join(date.to_string())
        .join("market_data")
        .join(format!("{code}_{date}_market_data.csv"));
    // 旧库的分钟 sum/count 在没有相应成交时填 0；OHLC 则仍为 NaN。
    for row in &mut out {
        for j in 0..=8 {
            row[j] = 0.0;
        }
    }

    if read_trades && trade_path.exists() {
        let mut trades = Vec::new();
        each_csv_row(&trade_path, |f| {
            if f.len() < 11 {
                return;
            }
            let flag = integer(f[10]);
            let time = integer(f[3]);
            if flag == 32 || bar(time).is_none() {
                return;
            }
            trades.push(Trade {
                time,
                price: number(f[7]),
                volume: number(f[8]),
                turnover: number(f[9]),
                flag,
                order: trades.len(),
            });
        })?;
        trades.sort_by_key(|r| (r.time, r.order));
        let mut prev = None;
        for r in trades {
            let Some(i) = bar(r.time) else {
                continue;
            };
            if i >= BARS {
                continue;
            }
            let x = &mut out[i];
            for j in [2, 7, 8] {
                if x[j].is_nan() {
                    x[j] = 0.0;
                }
            }
            x[2] += r.turnover;
            x[7] += r.volume;
            x[8] += r.turnover;
            if x[9].is_nan() || r.price > x[9] {
                x[9] = r.price;
            }
            if x[10].is_nan() || r.price < x[10] {
                x[10] = r.price;
            }
            x[11] = r.price;
            if r.flag == 66 {
                if x[0].is_nan() {
                    x[0] = 0.0;
                    x[3] = 0.0;
                }
                x[0] += r.turnover;
                x[3] += 1.0;
            } else if r.flag == 83 {
                if x[1].is_nan() {
                    x[1] = 0.0;
                    x[4] = 0.0;
                }
                x[1] += r.turnover;
                x[4] += 1.0;
            }
            if let Some(p) = prev {
                let j = if r.price > p {
                    Some(5)
                } else if r.price < p {
                    Some(6)
                } else {
                    None
                };
                if let Some(j) = j {
                    if x[j].is_nan() {
                        x[j] = 0.0;
                    }
                    x[j] += 1.0;
                }
            }
            prev = Some(r.price);
        }
    }

    if market_path.exists() {
        let mut quotes = Vec::new();
        each_csv_row(&market_path, |f| {
            if f.len() < 61 {
                return;
            }
            let time = integer(f[3]);
            if time < 91500000 || time > 150100000 {
                return;
            }
            let mut ask_size10 = 0.0;
            let mut bid_size10 = 0.0;
            let mut ask_value = 0.0;
            let mut bid_value = 0.0;
            for level in 0..10 {
                let base = 21 + 4 * level;
                let ap = number(f[base]);
                let av = number(f[base + 1]);
                let bp = number(f[base + 2]);
                let bv = number(f[base + 3]);
                ask_size10 += av;
                bid_size10 += bv;
                ask_value += ap * av;
                bid_value += bp * bv;
            }
            quotes.push(Quote {
                time,
                ask_prc1: number(f[21]),
                bid_prc1: number(f[23]),
                ask_vol1: number(f[22]),
                bid_vol1: number(f[24]),
                bid_vol5: number(f[40]),
                ask_size10,
                bid_size10,
                ask_vwap10: ask_value / ask_size10,
                bid_vwap10: bid_value / bid_size10,
                order: quotes.len(),
            });
        })?;
        quotes.sort_by_key(|r| (r.time, r.order));
        let mut count = [0usize; BARS];
        let mut spread_sum = [0.0f64; BARS];
        let mut spread_denominator = [0usize; BARS];
        let mut ask_size_sum = [0.0f64; BARS];
        let mut bid_size_sum = [0.0f64; BARS];
        let mut ask_vwap_seen = [false; BARS];
        let mut bid_vwap_seen = [false; BARS];
        let mut start_carry = *carry;
        for q in quotes {
            let i = bar(q.time);
            let midday_open = q.time == 130000000;
            if let Some(i) = i.filter(|&i| i < BARS) {
                count[i] += 1;
                ask_size_sum[i] += q.ask_size10;
                bid_size_sum[i] += q.bid_size10;
                // 历史 ask_vwap10 取最后一条原始快照，包括集合竞价的 crossed 快照。
                out[i][18] = q.ask_vwap10;
                ask_vwap_seen[i] = true;
                let difference = q.ask_prc1 - q.bid_prc1;
                if difference >= 0.0 {
                    spread_denominator[i] += 1;
                    if difference > 0.0 {
                        spread_sum[i] += (difference / 0.01).ln();
                    }
                }
            }
            if midday_open {
                out[120][18] = q.ask_vwap10;
                ask_vwap_seen[120] = true;
            }
            // 收盘集合竞价前会出现 ask==bid 或全零快照；旧库的均量包含它们，
            // 但价差与末笔盘口使用最后一条有效连续竞价快照。
            // 单边涨跌停盘口允许一侧价格为 0；双边同价（集合竞价过渡）不作为末笔盘口。
            if q.ask_prc1 != q.bid_prc1 {
                let values = [
                    q.bid_vol1,
                    q.bid_vol5,
                    q.ask_vol1,
                    q.ask_vwap10,
                    q.bid_vwap10,
                    q.ask_prc1,
                    q.bid_prc1,
                ];
                for k in 0..7 {
                    if values[k].is_finite() {
                        carry[k] = values[k];
                    }
                }
                if q.time < 93000000 {
                    start_carry = *carry;
                }
                if let Some(i) = i.filter(|&i| i < BARS) {
                    out[i][19] = q.bid_vwap10;
                    bid_vwap_seen[i] = true;
                    for k in 0..7 {
                        if values[k].is_finite() {
                            out[i][LAST_FIELDS[k]] = values[k];
                        }
                    }
                }
                if midday_open {
                    out[120][19] = q.bid_vwap10;
                    bid_vwap_seen[120] = true;
                    for k in 0..7 {
                        if values[k].is_finite() {
                            out[120][LAST_FIELDS[k]] = values[k];
                        }
                    }
                }
            }
        }
        // last 类字段跨空分钟、跨交易日沿用最后一条有效快照；均值类不前填。
        for (i, row) in out.iter_mut().enumerate() {
            for k in 0..7 {
                let col = LAST_FIELDS[k];
                if (col == 18 && ask_vwap_seen[i]) || (col == 19 && bid_vwap_seen[i]) {
                    start_carry[k] = row[col];
                    continue;
                }
                if row[col].is_finite() {
                    start_carry[k] = row[col];
                } else {
                    row[col] = start_carry[k];
                }
            }
        }
        let has_quote = count.iter().any(|&n| n > 0);
        for i in 0..BARS {
            if has_quote {
                out[i][12] = 0.0;
            }
            if count[i] > 0 {
                let denom = count[i] as f64;
                if spread_denominator[i] > 0 {
                    x_set(
                        &mut out[i],
                        12,
                        spread_sum[i] / spread_denominator[i] as f64,
                    );
                }
                // 历史 H5 中十档平均量的字段名与原始盘口方向相反。
                x_set(&mut out[i], 16, ask_size_sum[i] / denom);
                x_set(&mut out[i], 17, bid_size_sum[i] / denom);
            }
        }
    } else {
        for row in &mut out {
            for (k, &col) in LAST_FIELDS.iter().enumerate() {
                row[col] = carry[k];
            }
        }
    }
    Ok(out)
}

#[inline]
fn x_set(row: &mut [f64; 22], col: usize, value: f64) {
    row[col] = value;
}

pub fn load(date: i64, meta: &Meta) -> io::Result<(usize, usize, HashMap<&'static str, Vec<f64>>)> {
    let idx = meta.dates.binary_search(&date).map_err(|_| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("日期 {date} 不在交易日历中"),
        )
    })?;
    if idx + 1 < DAYS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "不足五个交易日",
        ));
    }
    let first_idx = idx + 1 - DAYS;
    let dates = &meta.dates[first_idx..=idx];
    let historical_start = meta.dates.binary_search(&20150105).unwrap_or(0);
    let n = meta.codes.len();
    let t = BARS * DAYS;
    let mut columns: Vec<Vec<f64>> = (0..FIELDS.len()).map(|_| vec![f64::NAN; t * n]).collect();
    let root = crate::data_paths::level2_root();
    // 分批并发读 CSV，限制临时内存；输出始终按 symbol_map.csv 顺序写入。
    for start in (0..n).step_by(32) {
        let end = (start + 32).min(n);
        let batch: Vec<io::Result<Vec<[[f64; 22]; BARS]>>> = meta.codes[start..end]
            .par_iter()
            .map(|code| {
                let mut carry = [f64::NAN; 7];
                let mut active = false;
                if first_idx > historical_start {
                    for &prior in meta.dates[historical_start..first_idx].iter().rev() {
                        if transaction_path(&root, code, prior).exists() {
                            active = true;
                            break;
                        }
                    }
                    if active {
                        for &prior in meta.dates[historical_start..first_idx].iter().rev() {
                            stock_day(&root, code, prior, &mut carry, false)?;
                            if carry[5].is_finite() {
                                break;
                            }
                        }
                    }
                }
                let mut result = Vec::with_capacity(DAYS);
                for &d in dates {
                    // ask_vwap10 在旧库只沿用当日上一条，不跨交易日沿用。
                    carry[3] = f64::NAN;
                    let trade_exists = transaction_path(&root, code, d).exists();
                    if trade_exists {
                        result.push(stock_day(&root, code, d, &mut carry, true)?);
                    } else {
                        result.push([[f64::NAN; 22]; BARS]);
                    }
                }
                Ok(result)
            })
            .collect();
        for (local, result) in batch.into_iter().enumerate() {
            let stock = start + local;
            for (day, rows) in result?.into_iter().enumerate() {
                for (bar, row) in rows.into_iter().enumerate() {
                    let dest = (day * BARS + bar) * n + stock;
                    for (field, &value) in row.iter().enumerate() {
                        columns[field][dest] = value;
                    }
                }
            }
        }
    }
    let raw = FIELDS.iter().copied().zip(columns).collect();
    Ok((t, n, raw))
}

#[cfg(test)]
mod tests {
    use super::bar;
    #[test]
    fn minute_boundaries() {
        assert_eq!(bar(93000000), None);
        assert_eq!(bar(93000050), Some(0));
        assert_eq!(bar(93100000), Some(1));
        assert_eq!(bar(130000000), None);
        assert_eq!(bar(130010000), Some(121));
    }
}
