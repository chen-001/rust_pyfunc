use csv::ReaderBuilder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::read_dir;
use std::path::{Path, PathBuf};

const DATA_ROOT: &str = "/ssd_data/stock";
const SELECTION_BURST_WINDOW: usize = 5;
const CODE_COARSE_SIZE: usize = 300;

const SCALAR_BASE_NAMES: [&str; 30] = [
    "cwalk_peer_stability",
    "cwalk_peer_cohesion",
    "cwalk_herd_peak",
    "cwalk_loneliness",
    "cwalk_leader_rate",
    "cwalk_follower_rate",
    "cwalk_false_start_rate",
    "cwalk_relay_length",
    "cwalk_theme_turnover",
    "cwalk_return_to_open_theme",
    "cwalk_latent_consensus",
    "cwalk_solo_surge",
    "cwalk_reverse_clearheaded",
    "cwalk_late_confirmation",
    "cwalk_market_heat_mean",
    "cwalk_market_heat_tail",
    "cwalk_peer_cohesion_std",
    "cwalk_lone_wolf_streak",
    "cwalk_crowd_sync_ratio",
    "cwalk_leader_conviction",
    "cwalk_follower_remorse",
    "cwalk_false_start_amplitude",
    "cwalk_relay_depth_max",
    "cwalk_theme_momentum",
    "cwalk_contradiction_rate",
    "cwalk_echo_lag",
    "cwalk_dominance_shift",
    "cwalk_group_volatility",
    "cwalk_cross_sectional_dispersion",
    "cwalk_tail_heat_asymmetry",
];

const VECTOR_BASE_NAMES: [&str; 20] = [
    "cwalk_peer_strength_vector",
    "cwalk_self_score_vector",
    "cwalk_group_score_vector",
    "cwalk_loneliness_score_vector",
    "cwalk_statement_score_vector",
    "cwalk_book_imbalance_rank_vector",
    "cwalk_return_gap_vector",
    "cwalk_volume_share_vector",
    "cwalk_peer_jaccard_vector",
    "cwalk_act_buy_advantage_vector",
    "cwalk_trade_burst_gap_vector",
    "cwalk_amplitude_ratio_vector",
    "cwalk_turnover_burst_rank_vector",
    "cwalk_jump_bias_rank_vector",
    "cwalk_self_heat_rank_vector",
    "cwalk_group_count_vector",
    "cwalk_lead_lag_score_vector",
    "cwalk_peer_return_std_vector",
    "cwalk_book_tilt_diff_vector",
    "cwalk_volume_weighted_peer_strength_vector",
];

#[derive(Clone, Copy)]
struct RoleCfg {
    top_k: usize,
    burst_window: usize,
    statement_quantile: f64,
    loneliness_threshold: f64,
    relay_horizon: usize,
}

#[derive(Clone, Copy)]
struct Experiment {
    tag: &'static str,
    role: RoleCfg,
}

const EXPERIMENTS: [Experiment; 4] = [
    Experiment {
        tag: "base",
        role: RoleCfg {
            top_k: 8,
            burst_window: 5,
            statement_quantile: 0.9,
            loneliness_threshold: 0.25,
            relay_horizon: 3,
        },
    },
    Experiment {
        tag: "sensitive",
        role: RoleCfg {
            top_k: 6,
            burst_window: 3,
            statement_quantile: 0.85,
            loneliness_threshold: 0.35,
            relay_horizon: 2,
        },
    },
    Experiment {
        tag: "smooth",
        role: RoleCfg {
            top_k: 12,
            burst_window: 8,
            statement_quantile: 0.95,
            loneliness_threshold: 0.2,
            relay_horizon: 5,
        },
    },
    Experiment {
        tag: "crowded",
        role: RoleCfg {
            top_k: 16,
            burst_window: 5,
            statement_quantile: 0.85,
            loneliness_threshold: 0.4,
            relay_horizon: 4,
        },
    },
];

#[derive(Clone, Default)]
struct RoleTradeAgg {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    amount: f64,
    volume: f64,
    num_trades: f64,
    act_buy_amount: f64,
    act_sell_amount: f64,
    up_jump: f64,
    down_jump: f64,
    first_seen: bool,
}

#[derive(Clone, Default)]
struct RoleMarketAgg {
    market_open: f64,
    market_high: f64,
    market_low: f64,
    market_close: f64,
    bid_size_1_sum: f64,
    ask_size_1_sum: f64,
    bid_size_10_sum: f64,
    ask_size_10_sum: f64,
    count: usize,
    first_seen: bool,
}

#[derive(Clone)]
struct SymbolData {
    symbol: String,
    role_trade: HashMap<i32, RoleTradeAgg>,
    role_market: HashMap<i32, RoleMarketAgg>,
}

#[derive(Clone)]
struct FilledRoleData {
    symbol: String,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    amount: Vec<f64>,
    volume: Vec<f64>,
    num_trades: Vec<f64>,
    act_buy_amount: Vec<f64>,
    act_sell_amount: Vec<f64>,
    up_jump: Vec<f64>,
    down_jump: Vec<f64>,
    bid_size_1: Vec<f64>,
    ask_size_1: Vec<f64>,
    bid_size_10: Vec<f64>,
    ask_size_10: Vec<f64>,
}

#[derive(Clone)]
struct SelectionPrepared {
    symbol: String,
    returns: Vec<f64>,
    volume: Vec<f64>,
    volume_burst: Vec<f64>,
    turnover_burst: Vec<f64>,
    act_buy_ratio: Vec<f64>,
    amplitude: Vec<f64>,
    book_tilt: Vec<f64>,
}

struct PrecomputedUniverse {
    filled: Vec<FilledRoleData>,
    anchor_idx: usize,
    n: usize,
    t_len: usize,
    r1: Vec<Vec<f64>>,
    amp: Vec<Vec<f64>>,
    act_buy_ratio: Vec<Vec<f64>>,
    jump_bias: Vec<Vec<f64>>,
    book_tilt_1: Vec<Vec<f64>>,
    book_tilt_10: Vec<Vec<f64>>,
    turn_burst_by_window: HashMap<usize, Vec<Vec<f64>>>,
    trade_burst_by_window: HashMap<usize, Vec<Vec<f64>>>,
}

fn scalar_feature_names() -> Vec<String> {
    EXPERIMENTS
        .iter()
        .flat_map(|exp| {
            SCALAR_BASE_NAMES
                .iter()
                .map(move |name| format!("{name}_{}", exp.tag))
        })
        .collect()
}

fn vector_feature_names() -> Vec<String> {
    EXPERIMENTS
        .iter()
        .flat_map(|exp| {
            VECTOR_BASE_NAMES
                .iter()
                .map(move |name| format!("{name}_{}", exp.tag))
        })
        .collect()
}

#[pyfunction]
pub fn personalized_meeting_feature_names() -> PyResult<(Vec<String>, Vec<String>)> {
    Ok((scalar_feature_names(), vector_feature_names()))
}

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.total_cmp(b)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_finite(values: &[f64]) -> f64 {
    let filtered: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    mean(&filtered)
}

fn std_sample(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let m = mean(values);
    let var = values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

fn std_finite(values: &[f64]) -> f64 {
    let filtered: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    std_sample(&filtered)
}

fn quantile_linear(values: &[f64], q: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(cmp_f64);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn winsorize(values: &[f64]) -> Vec<f64> {
    let lo = quantile_linear(values, 0.01);
    let hi = quantile_linear(values, 0.99);
    values
        .iter()
        .map(|v| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                0.5 * (lo + hi)
            }
        })
        .collect()
}

fn rank_pct_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut pairs: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| cmp_f64(&a.1, &b.1));
    let mut out = vec![0.0; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (pairs[j].1 - pairs[i].1).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = ((i + 1 + j) as f64) * 0.5;
        for k in i..j {
            out[pairs[k].0] = avg_rank / n as f64;
        }
        i = j;
    }
    out
}

fn rolling_mean(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    if window == 0 {
        return out;
    }
    let mut sum = 0.0;
    for i in 0..values.len() {
        sum += values[i];
        if i >= window {
            sum -= values[i - window];
        }
        let count = usize::min(i + 1, window);
        if count >= min_periods {
            out[i] = sum / count as f64;
        }
    }
    out
}

fn skewness(values: &[f64]) -> f64 {
    if values.len() <= 2 {
        return 0.0;
    }
    let m = mean(values);
    let s = std_sample(values);
    if s <= 1e-12 {
        return 0.0;
    }
    values.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>() / values.len() as f64
}

fn kurtosis_excess(values: &[f64]) -> f64 {
    if values.len() <= 3 {
        return 0.0;
    }
    let m = mean(values);
    let s = std_sample(values);
    if s <= 1e-12 {
        return 0.0;
    }
    values.iter().map(|v| ((v - m) / s).powi(4)).sum::<f64>() / values.len() as f64 - 3.0
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(x, y)| {
            if x.is_finite() && y.is_finite() {
                Some((*x, *y))
            } else {
                None
            }
        })
        .collect();
    if pairs.len() <= 1 {
        return 0.0;
    }
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pairs.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in pairs {
        cov += (x - mean_x) * (y - mean_y);
        var_x += (x - mean_x) * (x - mean_x);
        var_y += (y - mean_y) * (y - mean_y);
    }
    let denom = (var_x * var_y).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

fn lagged_correlation(current: &[f64], lagged: &[f64], lag: usize) -> f64 {
    if lag == 0 || lag >= current.len() || lag >= lagged.len() {
        return 0.0;
    }
    correlation(&current[lag..], &lagged[..(lagged.len() - lag)])
}

fn granger_like_score(a: &[f64], b: &[f64], max_lag: usize) -> f64 {
    (1..=max_lag)
        .map(|lag| {
            lagged_correlation(a, b, lag)
                .abs()
                .max(lagged_correlation(b, a, lag).abs())
        })
        .fold(0.0, f64::max)
}

fn co_movement_score(a: &[f64], b: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(x, y)| {
            if x.is_finite() && y.is_finite() {
                Some((*x, *y))
            } else {
                None
            }
        })
        .collect();
    if pairs.is_empty() {
        return 0.0;
    }
    let same = pairs
        .iter()
        .filter(|(x, y)| (*x >= 0.0 && *y >= 0.0) || (*x <= 0.0 && *y <= 0.0))
        .count() as f64;
    same / pairs.len() as f64
}

fn co_burst_score(a: &[f64], b: &[f64], threshold: f64) -> f64 {
    let n = usize::min(a.len(), b.len());
    if n == 0 {
        return 0.0;
    }
    let count = (0..n)
        .filter(|&i| a[i].is_finite() && b[i].is_finite() && a[i] >= threshold && b[i] >= threshold)
        .count() as f64;
    count / n as f64
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn mean_of_selected(values: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&idx| values[idx]).sum::<f64>() / indices.len() as f64
}

fn first_lag_above(values: &[f64], threshold: f64) -> f64 {
    for (idx, value) in values.iter().enumerate() {
        if *value >= threshold {
            return (idx + 1) as f64;
        }
    }
    0.0
}

fn longest_streak(flags: &[bool]) -> f64 {
    let mut best = 0usize;
    let mut cur = 0usize;
    for flag in flags {
        if *flag {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best as f64
}

fn minute_from_time_value(time_value: i64) -> i32 {
    let hour = (time_value / 10_000_000) as i32;
    let minute = ((time_value / 100_000) % 100) as i32;
    hour * 60 + minute
}

fn intraday_role_minutes() -> Vec<i32> {
    let mut out = Vec::with_capacity(239);
    for minute in 570..=690 {
        out.push(minute);
    }
    for minute in 780..=897 {
        out.push(minute);
    }
    out
}

fn role_minute_to_index(raw_minute: i32) -> Option<usize> {
    if (570..=690).contains(&raw_minute) {
        Some((raw_minute - 570) as usize)
    } else if (780..=897).contains(&raw_minute) {
        Some(121 + (raw_minute - 780) as usize)
    } else {
        None
    }
}

fn parse_i64(field: &[u8], field_name: &str, row_number: usize, path: &Path) -> PyResult<i64> {
    let raw = std::str::from_utf8(field)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "UTF-8解析失败: file={}, row={}, field={}, err={}",
                path.display(),
                row_number,
                field_name,
                err
            ))
        })?
        .trim();

    if let Ok(value) = raw.parse::<i64>() {
        return Ok(value);
    }

    let float_value = raw.parse::<f64>().map_err(|err| {
        PyValueError::new_err(format!(
            "整数解析失败: file={}, row={}, field={}, value={:?}, err={}",
            path.display(),
            row_number,
            field_name,
            raw,
            err
        ))
    })?;
    if !float_value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "整数列出现非有限值: file={}, row={}, field={}, value={:?}",
            path.display(),
            row_number,
            field_name,
            raw
        )));
    }
    let rounded = float_value.round();
    if (float_value - rounded).abs() > 1e-9 {
        return Err(PyValueError::new_err(format!(
            "整数列出现非整值: file={}, row={}, field={}, value={:?}",
            path.display(),
            row_number,
            field_name,
            raw
        )));
    }
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "整数列超出i64范围: file={}, row={}, field={}, value={:?}",
            path.display(),
            row_number,
            field_name,
            raw
        )));
    }
    Ok(rounded as i64)
}

fn parse_f64(field: &[u8], field_name: &str, row_number: usize, path: &Path) -> PyResult<f64> {
    let raw = std::str::from_utf8(field)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "UTF-8解析失败: file={}, row={}, field={}, err={}",
                path.display(),
                row_number,
                field_name,
                err
            ))
        })?
        .trim();
    raw.parse::<f64>().map_err(|err| {
        PyValueError::new_err(format!(
            "浮点解析失败: file={}, row={}, field={}, value={:?}, err={}",
            path.display(),
            row_number,
            field_name,
            raw,
            err
        ))
    })
}

fn csv_column_map(headers: &csv::ByteRecord, path: &Path) -> PyResult<HashMap<String, usize>> {
    headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let name = std::str::from_utf8(h).map_err(|err| {
                PyValueError::new_err(format!(
                    "CSV列名UTF-8解析失败: file={}, index={}, err={}",
                    path.display(),
                    i,
                    err
                ))
            })?;
            Ok((name.to_string(), i))
        })
        .collect()
}

fn column_index(cols: &HashMap<String, usize>, name: &str, path: &Path) -> PyResult<usize> {
    cols.get(name).copied().ok_or_else(|| {
        PyValueError::new_err(format!(
            "CSV缺少字段: file={}, field={}",
            path.display(),
            name
        ))
    })
}

fn parse_row_i64(
    record: &csv::ByteRecord,
    idx: usize,
    field_name: &str,
    row_number: usize,
    path: &Path,
) -> Option<i64> {
    let field = record.get(idx)?;
    parse_i64(field, field_name, row_number, path).ok()
}

fn parse_row_f64(
    record: &csv::ByteRecord,
    idx: usize,
    field_name: &str,
    row_number: usize,
    path: &Path,
) -> Option<f64> {
    let field = record.get(idx)?;
    parse_f64(field, field_name, row_number, path).ok()
}

fn available_symbols(date: i32) -> PyResult<Vec<String>> {
    let trade_dir = PathBuf::from(DATA_ROOT)
        .join(date.to_string())
        .join("transaction");
    let market_dir = PathBuf::from(DATA_ROOT)
        .join(date.to_string())
        .join("market_data");
    let trade_symbols: HashSet<String> = read_dir(&trade_dir)
        .map_err(|err| {
            PyRuntimeError::new_err(format!(
                "读取交易目录失败: dir={}, err={}",
                trade_dir.display(),
                err
            ))
        })?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str().map(|s| s.to_string()))
        .filter_map(|name| name.split('_').next().map(|s| s.to_string()))
        .collect();
    let mut market_symbols: Vec<String> = read_dir(&market_dir)
        .map_err(|err| {
            PyRuntimeError::new_err(format!(
                "读取盘口目录失败: dir={}, err={}",
                market_dir.display(),
                err
            ))
        })?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str().map(|s| s.to_string()))
        .filter_map(|name| name.split('_').next().map(|s| s.to_string()))
        .filter(|symbol| trade_symbols.contains(symbol))
        .collect();
    market_symbols.sort();
    Ok(market_symbols)
}

fn symbol_bucket(symbol: &str) -> String {
    if symbol.starts_with("68") {
        "68".to_string()
    } else if symbol.starts_with('6') {
        "6".to_string()
    } else if symbol.starts_with('3') {
        "3".to_string()
    } else if symbol.starts_with('0') {
        "0".to_string()
    } else {
        symbol
            .chars()
            .next()
            .map(|c| c.to_string())
            .unwrap_or_default()
    }
}

fn same_bucket_symbols(date: i32, anchor_symbol: &str) -> PyResult<Vec<String>> {
    let anchor_bucket = symbol_bucket(anchor_symbol);
    let mut symbols: Vec<String> = available_symbols(date)?
        .into_iter()
        .filter(|symbol| symbol_bucket(symbol) == anchor_bucket)
        .collect();
    symbols.sort();
    Ok(symbols)
}

fn select_code_neighbors(
    symbols: &[String],
    anchor_symbol: &str,
    size: usize,
) -> PyResult<Vec<String>> {
    if symbols.is_empty() {
        return Err(PyValueError::new_err("同组股票列表为空"));
    }
    let target = size.max(1).min(symbols.len());
    let anchor_idx = symbols
        .iter()
        .position(|item| item == anchor_symbol)
        .ok_or_else(|| {
            PyValueError::new_err(format!("未在候选股票中找到锚点股票: {}", anchor_symbol))
        })?;
    let mut start = anchor_idx.saturating_sub((target - 1) / 2);
    let end = usize::min(symbols.len(), start + target);
    if end - start < target {
        start = end.saturating_sub(target);
    }
    Ok(symbols[start..end].to_vec())
}

fn load_symbol_data(date: i32, symbol: &str) -> PyResult<SymbolData> {
    let trade_path = PathBuf::from(DATA_ROOT)
        .join(date.to_string())
        .join("transaction")
        .join(format!("{symbol}_{date}_transaction.csv"));
    let market_path = PathBuf::from(DATA_ROOT)
        .join(date.to_string())
        .join("market_data")
        .join(format!("{symbol}_{date}_market_data.csv"));

    let mut role_trade: HashMap<i32, RoleTradeAgg> = HashMap::new();
    let mut role_market: HashMap<i32, RoleMarketAgg> = HashMap::new();
    let mut prev_price: Option<f64> = None;

    {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(&trade_path)
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "打开交易CSV失败: file={}, err={}",
                    trade_path.display(),
                    err
                ))
            })?;
        let headers = rdr
            .byte_headers()
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "读取交易CSV表头失败: file={}, err={}",
                    trade_path.display(),
                    err
                ))
            })?
            .clone();
        let cols = csv_column_map(&headers, &trade_path)?;
        let time_idx = column_index(&cols, "time", &trade_path)?;
        let price_idx = column_index(&cols, "price", &trade_path)?;
        let volume_idx = column_index(&cols, "volume", &trade_path)?;
        let turnover_idx = column_index(&cols, "turnover", &trade_path)?;
        let flag_idx = column_index(&cols, "flag", &trade_path)?;

        for (row_idx, row) in rdr.byte_records().enumerate() {
            let row_number = row_idx + 2;
            let Ok(record) = row else {
                continue;
            };
            let Some(flag) = parse_row_i64(&record, flag_idx, "flag", row_number, &trade_path)
            else {
                continue;
            };
            let flag = flag as i32;
            if flag == 32 {
                continue;
            }
            let Some(time_value) =
                parse_row_i64(&record, time_idx, "time", row_number, &trade_path)
            else {
                continue;
            };
            let Some(price) = parse_row_f64(&record, price_idx, "price", row_number, &trade_path)
            else {
                continue;
            };
            let Some(volume) =
                parse_row_f64(&record, volume_idx, "volume", row_number, &trade_path)
            else {
                continue;
            };
            let Some(turnover) =
                parse_row_f64(&record, turnover_idx, "turnover", row_number, &trade_path)
            else {
                continue;
            };

            let raw_minute = minute_from_time_value(time_value);
            let trade_agg = role_trade.entry(raw_minute).or_default();
            if !trade_agg.first_seen {
                trade_agg.open = price;
                trade_agg.high = price;
                trade_agg.low = price;
                trade_agg.first_seen = true;
            }
            trade_agg.high = trade_agg.high.max(price);
            trade_agg.low = trade_agg.low.min(price);
            trade_agg.close = price;
            trade_agg.amount += turnover;
            trade_agg.volume += volume;
            trade_agg.num_trades += 1.0;
            if flag == 66 {
                trade_agg.act_buy_amount += turnover;
            } else if flag == 83 {
                trade_agg.act_sell_amount += turnover;
            }
            if let Some(prev) = prev_price {
                if price > prev {
                    trade_agg.up_jump += 1.0;
                } else if price < prev {
                    trade_agg.down_jump += 1.0;
                }
            }
            prev_price = Some(price);
        }
    }

    {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(&market_path)
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "打开盘口CSV失败: file={}, err={}",
                    market_path.display(),
                    err
                ))
            })?;
        let headers = rdr
            .byte_headers()
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "读取盘口CSV表头失败: file={}, err={}",
                    market_path.display(),
                    err
                ))
            })?
            .clone();
        let cols = csv_column_map(&headers, &market_path)?;
        let time_idx = column_index(&cols, "time", &market_path)?;
        let last_prc_idx = column_index(&cols, "last_prc", &market_path)?;
        let bid1_idx = column_index(&cols, "bid_vol1", &market_path)?;
        let ask1_idx = column_index(&cols, "ask_vol1", &market_path)?;
        let bid10_idx = column_index(&cols, "bid_vol10", &market_path)?;
        let ask10_idx = column_index(&cols, "ask_vol10", &market_path)?;

        for (row_idx, row) in rdr.byte_records().enumerate() {
            let row_number = row_idx + 2;
            let Ok(record) = row else {
                continue;
            };
            let Some(last_prc) =
                parse_row_f64(&record, last_prc_idx, "last_prc", row_number, &market_path)
            else {
                continue;
            };
            if last_prc <= 0.0 {
                continue;
            }
            let Some(time_value) =
                parse_row_i64(&record, time_idx, "time", row_number, &market_path)
            else {
                continue;
            };
            let Some(bid1) = parse_row_f64(&record, bid1_idx, "bid_vol1", row_number, &market_path)
            else {
                continue;
            };
            let Some(ask1) = parse_row_f64(&record, ask1_idx, "ask_vol1", row_number, &market_path)
            else {
                continue;
            };
            let Some(bid10) =
                parse_row_f64(&record, bid10_idx, "bid_vol10", row_number, &market_path)
            else {
                continue;
            };
            let Some(ask10) =
                parse_row_f64(&record, ask10_idx, "ask_vol10", row_number, &market_path)
            else {
                continue;
            };

            let raw_minute = minute_from_time_value(time_value);
            let market_agg = role_market.entry(raw_minute).or_default();
            if !market_agg.first_seen {
                market_agg.market_open = last_prc;
                market_agg.market_high = last_prc;
                market_agg.market_low = last_prc;
                market_agg.first_seen = true;
            }
            market_agg.market_high = market_agg.market_high.max(last_prc);
            market_agg.market_low = market_agg.market_low.min(last_prc);
            market_agg.market_close = last_prc;
            market_agg.bid_size_1_sum += bid1;
            market_agg.ask_size_1_sum += ask1;
            market_agg.bid_size_10_sum += bid10;
            market_agg.ask_size_10_sum += ask10;
            market_agg.count += 1;
        }
    }

    Ok(SymbolData {
        symbol: symbol.to_string(),
        role_trade,
        role_market,
    })
}

fn fill_role_data(symbol_data: &SymbolData, minutes: &[i32]) -> FilledRoleData {
    let mut open = Vec::with_capacity(minutes.len());
    let mut high = Vec::with_capacity(minutes.len());
    let mut low = Vec::with_capacity(minutes.len());
    let mut close = Vec::with_capacity(minutes.len());
    let mut amount = Vec::with_capacity(minutes.len());
    let mut volume = Vec::with_capacity(minutes.len());
    let mut num_trades = Vec::with_capacity(minutes.len());
    let mut act_buy_amount = Vec::with_capacity(minutes.len());
    let mut act_sell_amount = Vec::with_capacity(minutes.len());
    let mut up_jump = Vec::with_capacity(minutes.len());
    let mut down_jump = Vec::with_capacity(minutes.len());
    let mut bid_size_1 = Vec::with_capacity(minutes.len());
    let mut ask_size_1 = Vec::with_capacity(minutes.len());
    let mut bid_size_10 = Vec::with_capacity(minutes.len());
    let mut ask_size_10 = Vec::with_capacity(minutes.len());

    for minute in minutes {
        if let Some(market) = symbol_data.role_market.get(minute) {
            let trade = symbol_data.role_trade.get(minute);
            open.push(trade.map(|x| x.open).unwrap_or(market.market_open));
            high.push(trade.map(|x| x.high).unwrap_or(market.market_high));
            low.push(trade.map(|x| x.low).unwrap_or(market.market_low));
            close.push(trade.map(|x| x.close).unwrap_or(market.market_close));
            amount.push(trade.map(|x| x.amount).unwrap_or(0.0));
            volume.push(trade.map(|x| x.volume).unwrap_or(0.0));
            num_trades.push(trade.map(|x| x.num_trades).unwrap_or(0.0));
            act_buy_amount.push(trade.map(|x| x.act_buy_amount).unwrap_or(0.0));
            act_sell_amount.push(trade.map(|x| x.act_sell_amount).unwrap_or(0.0));
            up_jump.push(trade.map(|x| x.up_jump).unwrap_or(0.0));
            down_jump.push(trade.map(|x| x.down_jump).unwrap_or(0.0));
            bid_size_1.push(market.bid_size_1_sum / market.count.max(1) as f64);
            ask_size_1.push(market.ask_size_1_sum / market.count.max(1) as f64);
            bid_size_10.push(market.bid_size_10_sum / market.count.max(1) as f64);
            ask_size_10.push(market.ask_size_10_sum / market.count.max(1) as f64);
        } else {
            open.push(f64::NAN);
            high.push(f64::NAN);
            low.push(f64::NAN);
            close.push(f64::NAN);
            amount.push(0.0);
            volume.push(0.0);
            num_trades.push(0.0);
            act_buy_amount.push(0.0);
            act_sell_amount.push(0.0);
            up_jump.push(0.0);
            down_jump.push(0.0);
            bid_size_1.push(f64::NAN);
            ask_size_1.push(f64::NAN);
            bid_size_10.push(f64::NAN);
            ask_size_10.push(f64::NAN);
        }
    }

    let forward_fill = |values: &mut Vec<f64>| {
        let mut last = f64::NAN;
        for value in values.iter_mut() {
            if value.is_nan() {
                if !last.is_nan() {
                    *value = last;
                }
            } else {
                last = *value;
            }
        }
        let mut last = f64::NAN;
        for value in values.iter_mut().rev() {
            if value.is_nan() {
                if !last.is_nan() {
                    *value = last;
                }
            } else {
                last = *value;
            }
        }
    };

    forward_fill(&mut close);
    forward_fill(&mut bid_size_1);
    forward_fill(&mut ask_size_1);
    forward_fill(&mut bid_size_10);
    forward_fill(&mut ask_size_10);

    for i in 0..minutes.len() {
        if open[i].is_nan() {
            open[i] = close[i];
        }
        if high[i].is_nan() {
            high[i] = close[i];
        }
        if low[i].is_nan() {
            low[i] = close[i];
        }
    }

    FilledRoleData {
        symbol: symbol_data.symbol.clone(),
        open,
        high,
        low,
        close,
        amount,
        volume,
        num_trades,
        act_buy_amount,
        act_sell_amount,
        up_jump,
        down_jump,
        bid_size_1,
        ask_size_1,
        bid_size_10,
        ask_size_10,
    }
}

fn selection_requirements(method: &str) -> PyResult<(bool, bool)> {
    let flags = match method {
        "ret_corr" | "co_move" | "granger" | "amplitude_corr" | "book_tilt_corr" => (false, true),
        "vol_corr" | "turnover_corr" | "buy_ratio_corr" | "co_burst" => (true, false),
        "multidim" => (true, true),
        _ => {
            return Err(PyValueError::new_err(format!(
                "不支持的selection_method: {}",
                method
            )))
        }
    };
    Ok(flags)
}

fn load_selection_prepared(
    date: i32,
    symbol: &str,
    need_trade: bool,
    need_market: bool,
) -> PyResult<SelectionPrepared> {
    let trade_path = PathBuf::from(DATA_ROOT)
        .join(date.to_string())
        .join("transaction")
        .join(format!("{symbol}_{date}_transaction.csv"));
    let market_path = PathBuf::from(DATA_ROOT)
        .join(date.to_string())
        .join("market_data")
        .join(format!("{symbol}_{date}_market_data.csv"));

    let n = 239usize;
    let mut market_open = vec![f64::NAN; n];
    let mut market_high = vec![f64::NAN; n];
    let mut market_low = vec![f64::NAN; n];
    let mut market_close = vec![f64::NAN; n];
    let mut bid10_sum = vec![0.0; n];
    let mut ask10_sum = vec![0.0; n];
    let mut market_count = vec![0usize; n];

    let mut volume = vec![0.0; n];
    let mut amount = vec![0.0; n];
    let mut act_buy_amount = vec![0.0; n];
    let mut act_sell_amount = vec![0.0; n];

    if need_trade {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(&trade_path)
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "打开交易CSV失败: file={}, err={}",
                    trade_path.display(),
                    err
                ))
            })?;
        let headers = rdr
            .byte_headers()
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "读取交易CSV表头失败: file={}, err={}",
                    trade_path.display(),
                    err
                ))
            })?
            .clone();
        let cols = csv_column_map(&headers, &trade_path)?;
        let time_idx = column_index(&cols, "time", &trade_path)?;
        let volume_idx = column_index(&cols, "volume", &trade_path)?;
        let turnover_idx = column_index(&cols, "turnover", &trade_path)?;
        let flag_idx = column_index(&cols, "flag", &trade_path)?;

        for (row_idx, row) in rdr.byte_records().enumerate() {
            let row_number = row_idx + 2;
            let Ok(record) = row else {
                continue;
            };
            let Some(flag) = parse_row_i64(&record, flag_idx, "flag", row_number, &trade_path)
            else {
                continue;
            };
            let flag = flag as i32;
            if flag == 32 {
                continue;
            }
            let Some(time_value) =
                parse_row_i64(&record, time_idx, "time", row_number, &trade_path)
            else {
                continue;
            };
            let Some(idx) = role_minute_to_index(minute_from_time_value(time_value)) else {
                continue;
            };
            let Some(v) = parse_row_f64(&record, volume_idx, "volume", row_number, &trade_path)
            else {
                continue;
            };
            let Some(turnover) =
                parse_row_f64(&record, turnover_idx, "turnover", row_number, &trade_path)
            else {
                continue;
            };
            volume[idx] += v;
            amount[idx] += turnover;
            if flag == 66 {
                act_buy_amount[idx] += turnover;
            } else if flag == 83 {
                act_sell_amount[idx] += turnover;
            }
        }
    }

    if need_market {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(&market_path)
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "打开盘口CSV失败: file={}, err={}",
                    market_path.display(),
                    err
                ))
            })?;
        let headers = rdr
            .byte_headers()
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "读取盘口CSV表头失败: file={}, err={}",
                    market_path.display(),
                    err
                ))
            })?
            .clone();
        let cols = csv_column_map(&headers, &market_path)?;
        let time_idx = column_index(&cols, "time", &market_path)?;
        let last_prc_idx = column_index(&cols, "last_prc", &market_path)?;
        let bid10_idx = column_index(&cols, "bid_vol10", &market_path)?;
        let ask10_idx = column_index(&cols, "ask_vol10", &market_path)?;

        for (row_idx, row) in rdr.byte_records().enumerate() {
            let row_number = row_idx + 2;
            let Ok(record) = row else {
                continue;
            };
            let Some(last_prc) =
                parse_row_f64(&record, last_prc_idx, "last_prc", row_number, &market_path)
            else {
                continue;
            };
            if last_prc <= 0.0 {
                continue;
            }
            let Some(time_value) =
                parse_row_i64(&record, time_idx, "time", row_number, &market_path)
            else {
                continue;
            };
            let Some(idx) = role_minute_to_index(minute_from_time_value(time_value)) else {
                continue;
            };
            let Some(bid10) =
                parse_row_f64(&record, bid10_idx, "bid_vol10", row_number, &market_path)
            else {
                continue;
            };
            let Some(ask10) =
                parse_row_f64(&record, ask10_idx, "ask_vol10", row_number, &market_path)
            else {
                continue;
            };
            if market_count[idx] == 0 {
                market_open[idx] = last_prc;
                market_high[idx] = last_prc;
                market_low[idx] = last_prc;
            } else {
                market_high[idx] = market_high[idx].max(last_prc);
                market_low[idx] = market_low[idx].min(last_prc);
            }
            market_close[idx] = last_prc;
            bid10_sum[idx] += bid10;
            ask10_sum[idx] += ask10;
            market_count[idx] += 1;
        }

        let forward_fill = |values: &mut Vec<f64>| {
            let mut last = f64::NAN;
            for value in values.iter_mut() {
                if value.is_nan() {
                    if !last.is_nan() {
                        *value = last;
                    }
                } else {
                    last = *value;
                }
            }
            let mut last = f64::NAN;
            for value in values.iter_mut().rev() {
                if value.is_nan() {
                    if !last.is_nan() {
                        *value = last;
                    }
                } else {
                    last = *value;
                }
            }
            for value in values.iter_mut() {
                if value.is_nan() {
                    *value = 0.0;
                }
            }
        };

        let mut bid10_avg = vec![f64::NAN; n];
        let mut ask10_avg = vec![f64::NAN; n];
        for i in 0..n {
            if market_count[i] > 0 {
                bid10_avg[i] = bid10_sum[i] / market_count[i] as f64;
                ask10_avg[i] = ask10_sum[i] / market_count[i] as f64;
            }
        }
        forward_fill(&mut market_close);
        forward_fill(&mut bid10_avg);
        forward_fill(&mut ask10_avg);
        for i in 0..n {
            if market_open[i].is_nan() {
                market_open[i] = market_close[i];
            }
            if market_high[i].is_nan() {
                market_high[i] = market_close[i];
            }
            if market_low[i].is_nan() {
                market_low[i] = market_close[i];
            }
            bid10_sum[i] = bid10_avg[i];
            ask10_sum[i] = ask10_avg[i];
        }
    }

    let amount_ma = if need_trade {
        rolling_mean(&amount, SELECTION_BURST_WINDOW, 1)
    } else {
        vec![0.0; n]
    };
    let volume_ma = if need_trade {
        rolling_mean(&volume, SELECTION_BURST_WINDOW, 1)
    } else {
        vec![0.0; n]
    };

    let mut returns = vec![0.0; n];
    let mut turnover_burst = vec![0.0; n];
    let mut volume_burst = vec![0.0; n];
    let mut act_buy_ratio = vec![0.0; n];
    let mut amplitude = vec![0.0; n];
    let mut book_tilt = vec![0.0; n];
    for t in 0..n {
        if need_market {
            let prev_close = if t == 0 {
                market_close[t]
            } else {
                market_close[t - 1]
            };
            returns[t] = if market_open[t].abs() > 1e-12 {
                market_close[t] / market_open[t] - 1.0
            } else {
                market_close[t] / prev_close.max(1e-6) - 1.0
            };
            amplitude[t] = (market_high[t] - market_low[t]) / (prev_close.abs() + 1e-6);
            book_tilt[t] = (bid10_sum[t] - ask10_sum[t]) / (bid10_sum[t] + ask10_sum[t] + 1e-6);
        }
        if need_trade {
            turnover_burst[t] = amount[t] / (amount_ma[t] + 1e-6);
            volume_burst[t] = volume[t] / (volume_ma[t] + 1e-6);
            act_buy_ratio[t] = act_buy_amount[t] / (act_buy_amount[t] + act_sell_amount[t] + 1e-6);
        }
    }

    Ok(SelectionPrepared {
        symbol: symbol.to_string(),
        returns,
        volume,
        volume_burst,
        turnover_burst,
        act_buy_ratio,
        amplitude,
        book_tilt,
    })
}

fn selection_summary(prepared: &SelectionPrepared) -> Vec<f64> {
    vec![
        std_finite(&prepared.returns),
        (mean_finite(&prepared.volume) + 1.0).ln(),
        mean_finite(&prepared.turnover_burst),
        mean_finite(&prepared.amplitude),
        mean_finite(&prepared.act_buy_ratio),
        mean_finite(&prepared.book_tilt),
        skewness(&prepared.returns),
        kurtosis_excess(&prepared.returns),
    ]
}

fn select_symbols_by_method(
    prepared: &[SelectionPrepared],
    anchor_symbol: &str,
    size: usize,
    method: &str,
) -> PyResult<Vec<String>> {
    let mut sorted_symbols: Vec<String> = prepared.iter().map(|item| item.symbol.clone()).collect();
    sorted_symbols.sort();
    if size >= sorted_symbols.len() {
        return Ok(sorted_symbols);
    }

    let anchor_idx = prepared
        .iter()
        .position(|item| item.symbol == anchor_symbol)
        .ok_or_else(|| PyValueError::new_err(format!("未找到锚点股票: {}", anchor_symbol)))?;

    let mut score_map: HashMap<String, f64> = HashMap::new();
    match method {
        "ret_corr" | "vol_corr" | "turnover_corr" | "buy_ratio_corr" | "amplitude_corr"
        | "book_tilt_corr" | "co_move" | "co_burst" | "granger" => {
            for (idx, item) in prepared.iter().enumerate() {
                if idx == anchor_idx {
                    continue;
                }
                let score = match method {
                    "ret_corr" => correlation(&prepared[anchor_idx].returns, &item.returns),
                    "vol_corr" => correlation(&prepared[anchor_idx].volume, &item.volume),
                    "turnover_corr" => {
                        correlation(&prepared[anchor_idx].turnover_burst, &item.turnover_burst)
                    }
                    "buy_ratio_corr" => {
                        correlation(&prepared[anchor_idx].act_buy_ratio, &item.act_buy_ratio)
                    }
                    "amplitude_corr" => {
                        correlation(&prepared[anchor_idx].amplitude, &item.amplitude)
                    }
                    "book_tilt_corr" => {
                        correlation(&prepared[anchor_idx].book_tilt, &item.book_tilt)
                    }
                    "co_move" => co_movement_score(&prepared[anchor_idx].returns, &item.returns),
                    "co_burst" => {
                        co_burst_score(&prepared[anchor_idx].volume_burst, &item.volume_burst, 1.5)
                    }
                    "granger" => {
                        granger_like_score(&prepared[anchor_idx].returns, &item.returns, 3)
                    }
                    _ => 0.0,
                };
                score_map.insert(item.symbol.clone(), score);
            }
        }
        "multidim" => {
            let summaries: Vec<Vec<f64>> = prepared.iter().map(selection_summary).collect();
            let dim = summaries.first().map(|x| x.len()).unwrap_or(0);
            let mut means = vec![0.0; dim];
            let mut stds = vec![0.0; dim];
            for d in 0..dim {
                let col: Vec<f64> = summaries.iter().map(|row| row[d]).collect();
                means[d] = mean_finite(&col);
                stds[d] = std_finite(&col).max(1e-6);
            }
            let standardized: Vec<Vec<f64>> = summaries
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(d, value)| (value - means[d]) / stds[d])
                        .collect::<Vec<_>>()
                })
                .collect();
            let anchor_vec = &standardized[anchor_idx];
            for (idx, item) in prepared.iter().enumerate() {
                if idx == anchor_idx {
                    continue;
                }
                score_map.insert(
                    item.symbol.clone(),
                    -euclidean_distance(anchor_vec, &standardized[idx]),
                );
            }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "不支持的selection_method: {}",
                method
            )))
        }
    }

    let mut scored: Vec<(String, f64)> = score_map.into_iter().collect();
    scored.sort_by(|a, b| cmp_f64(&b.1, &a.1).then_with(|| a.0.cmp(&b.0)));
    let mut selected: Vec<String> = scored
        .into_iter()
        .take(size.saturating_sub(1))
        .map(|(symbol, _)| symbol)
        .collect();
    selected.push(anchor_symbol.to_string());
    selected.sort();
    Ok(selected)
}

fn select_symbols(
    date: i32,
    anchor_symbol: &str,
    size: usize,
    selection_method: &str,
) -> PyResult<Vec<String>> {
    if size == 0 {
        return Err(PyValueError::new_err("universe_size 必须大于0"));
    }
    let bucket_symbols = same_bucket_symbols(date, anchor_symbol)?;
    if bucket_symbols.is_empty() {
        return Err(PyValueError::new_err(format!(
            "未找到与锚点同分组的股票: {}",
            anchor_symbol
        )));
    }
    if selection_method == "code" {
        return select_code_neighbors(&bucket_symbols, anchor_symbol, size);
    }

    let (candidate_symbols, base_method): (Vec<String>, &str) =
        if let Some(inner) = selection_method.strip_prefix("code_") {
            let coarse_size = bucket_symbols
                .len()
                .min(size.saturating_mul(3).max(CODE_COARSE_SIZE));
            (
                select_code_neighbors(&bucket_symbols, anchor_symbol, coarse_size)?,
                inner,
            )
        } else {
            (bucket_symbols.clone(), selection_method)
        };

    let (need_trade, need_market) = selection_requirements(base_method)?;
    let prepared: Vec<SelectionPrepared> = candidate_symbols
        .iter()
        .map(|symbol| load_selection_prepared(date, symbol, need_trade, need_market))
        .collect::<PyResult<Vec<_>>>()?;
    select_symbols_by_method(
        &prepared,
        anchor_symbol,
        size.min(candidate_symbols.len()),
        base_method,
    )
}

fn mean_of_selected_at(matrix: &[Vec<f64>], t: usize, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&idx| matrix[idx][t]).sum::<f64>() / indices.len() as f64
}

fn std_of_selected_at(matrix: &[Vec<f64>], t: usize, indices: &[usize]) -> f64 {
    if indices.len() <= 1 {
        return 0.0;
    }
    let values: Vec<f64> = indices.iter().map(|&idx| matrix[idx][t]).collect();
    std_sample(&values)
}

fn cosine_top_k_from_ranked_fields(
    anchor_idx: usize,
    ranked_fields: &[Vec<f64>],
    top_k: usize,
) -> (Vec<usize>, Vec<f64>, f64) {
    let anchor_norm = ranked_fields
        .iter()
        .map(|field| field[anchor_idx] * field[anchor_idx])
        .sum::<f64>()
        .sqrt()
        .max(1e-8);
    let n = ranked_fields.first().map(|field| field.len()).unwrap_or(0);
    let mut sims: Vec<(usize, f64)> = (0..n)
        .filter(|idx| *idx != anchor_idx)
        .map(|idx| {
            let dot = ranked_fields
                .iter()
                .map(|field| field[anchor_idx] * field[idx])
                .sum::<f64>();
            let norm = ranked_fields
                .iter()
                .map(|field| field[idx] * field[idx])
                .sum::<f64>()
                .sqrt()
                .max(1e-8);
            (idx, dot / (anchor_norm * norm))
        })
        .collect();
    sims.sort_by(|a, b| cmp_f64(&b.1, &a.1));
    let k = usize::min(top_k, sims.len());
    let selected: Vec<usize> = sims.iter().take(k).map(|(idx, _)| *idx).collect();
    let selected_scores: Vec<f64> = sims.iter().take(k).map(|(_, score)| *score).collect();
    let mean_score = mean(&selected_scores);
    (selected, selected_scores, mean_score)
}

fn precompute_universe(subset: &[&SymbolData], anchor_symbol: &str) -> PrecomputedUniverse {
    let minutes = intraday_role_minutes();
    let filled: Vec<FilledRoleData> = subset
        .iter()
        .map(|symbol| fill_role_data(symbol, &minutes))
        .collect();
    let anchor_idx = filled
        .iter()
        .position(|item| item.symbol == anchor_symbol)
        .unwrap();
    let n = filled.len();
    let t_len = minutes.len();

    let mut r1 = vec![vec![0.0; t_len]; n];
    let mut amp = vec![vec![0.0; t_len]; n];
    let mut act_buy_ratio = vec![vec![0.0; t_len]; n];
    let mut jump_bias = vec![vec![0.0; t_len]; n];
    let mut book_tilt_1 = vec![vec![0.0; t_len]; n];
    let mut book_tilt_10 = vec![vec![0.0; t_len]; n];

    let mut windows: Vec<usize> = EXPERIMENTS
        .iter()
        .map(|exp| exp.role.burst_window)
        .collect();
    windows.sort();
    windows.dedup();
    let mut turn_burst_by_window: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
    let mut trade_burst_by_window: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
    for &window in &windows {
        turn_burst_by_window.insert(window, vec![vec![0.0; t_len]; n]);
        trade_burst_by_window.insert(window, vec![vec![0.0; t_len]; n]);
    }

    for (i, symbol) in filled.iter().enumerate() {
        for t in 0..t_len {
            let prev_close = if t == 0 {
                symbol.close[t]
            } else {
                symbol.close[t - 1]
            };
            r1[i][t] = if symbol.open[t].abs() > 1e-12 {
                symbol.close[t] / symbol.open[t] - 1.0
            } else {
                symbol.close[t] / prev_close.max(1e-6) - 1.0
            };
            amp[i][t] = (symbol.high[t] - symbol.low[t]) / (prev_close.abs() + 1e-6);
            act_buy_ratio[i][t] = symbol.act_buy_amount[t]
                / (symbol.act_buy_amount[t] + symbol.act_sell_amount[t] + 1e-6);
            jump_bias[i][t] = (symbol.up_jump[t] - symbol.down_jump[t])
                / (symbol.up_jump[t] + symbol.down_jump[t] + 1.0);
            book_tilt_1[i][t] = (symbol.bid_size_1[t] - symbol.ask_size_1[t])
                / (symbol.bid_size_1[t] + symbol.ask_size_1[t] + 1e-6);
            book_tilt_10[i][t] = (symbol.bid_size_10[t] - symbol.ask_size_10[t])
                / (symbol.bid_size_10[t] + symbol.ask_size_10[t] + 1e-6);
        }
        for &window in &windows {
            let amount_ma = rolling_mean(&symbol.amount, window, 1);
            let trade_ma = rolling_mean(&symbol.num_trades, window, 1);
            let turn_burst = turn_burst_by_window.get_mut(&window).unwrap();
            let trade_burst = trade_burst_by_window.get_mut(&window).unwrap();
            for t in 0..t_len {
                turn_burst[i][t] = symbol.amount[t] / (amount_ma[t] + 1e-6);
                trade_burst[i][t] = symbol.num_trades[t] / (trade_ma[t] + 1e-6);
            }
        }
    }

    PrecomputedUniverse {
        filled,
        anchor_idx,
        n,
        t_len,
        r1,
        amp,
        act_buy_ratio,
        jump_bias,
        book_tilt_1,
        book_tilt_10,
        turn_burst_by_window,
        trade_burst_by_window,
    }
}

fn compute_role_features(
    precomputed: &PrecomputedUniverse,
    cfg: RoleCfg,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let filled = &precomputed.filled;
    let anchor_idx = precomputed.anchor_idx;
    let n = precomputed.n;
    let t_len = precomputed.t_len;
    let r1 = &precomputed.r1;
    let amp = &precomputed.amp;
    let act_buy_ratio = &precomputed.act_buy_ratio;
    let jump_bias = &precomputed.jump_bias;
    let book_tilt_1 = &precomputed.book_tilt_1;
    let book_tilt_10 = &precomputed.book_tilt_10;
    let turn_burst = precomputed
        .turn_burst_by_window
        .get(&cfg.burst_window)
        .unwrap();
    let trade_burst = precomputed
        .trade_burst_by_window
        .get(&cfg.burst_window)
        .unwrap();

    let mut score_matrix = vec![vec![0.0; n]; t_len];
    let mut statement_matrix = vec![vec![false; n]; t_len];
    let mut peer_lists: Vec<Vec<usize>> = vec![Vec::new(); t_len];
    let mut peer_score_lists: Vec<Vec<f64>> = vec![Vec::new(); t_len];
    let mut peer_strength = vec![0.0; t_len];
    let mut self_heat = vec![0.0; t_len];
    let mut group_heat = vec![0.0; t_len];
    let mut threshold_vector = vec![0.0; t_len];
    let mut book_imbalance_rank_vector = vec![0.0; t_len];
    let mut turnover_burst_rank_vector = vec![0.0; t_len];
    let mut jump_bias_rank_vector = vec![0.0; t_len];
    let mut self_heat_rank_vector = vec![0.0; t_len];

    for t in 0..t_len {
        let fields = vec![
            r1.iter().map(|x| x[t]).collect::<Vec<_>>(),
            amp.iter().map(|x| x[t]).collect::<Vec<_>>(),
            turn_burst.iter().map(|x| x[t]).collect::<Vec<_>>(),
            act_buy_ratio.iter().map(|x| x[t]).collect::<Vec<_>>(),
            trade_burst.iter().map(|x| x[t]).collect::<Vec<_>>(),
            jump_bias.iter().map(|x| x[t]).collect::<Vec<_>>(),
            book_tilt_1.iter().map(|x| x[t]).collect::<Vec<_>>(),
            book_tilt_10.iter().map(|x| x[t]).collect::<Vec<_>>(),
        ];
        let ranked_fields: Vec<Vec<f64>> = fields
            .iter()
            .map(|field| rank_pct_average(&winsorize(field)))
            .collect();
        let (peers, peer_scores, mean_sim) =
            cosine_top_k_from_ranked_fields(anchor_idx, &ranked_fields, cfg.top_k);
        peer_lists[t] = peers.clone();
        peer_score_lists[t] = peer_scores;
        peer_strength[t] = mean_sim;
        for i in 0..n {
            score_matrix[t][i] = (ranked_fields[2][i]
                + ranked_fields[4][i]
                + ranked_fields[3][i]
                + ranked_fields[5][i])
                * 0.25;
        }
        let threshold = quantile_linear(&score_matrix[t], cfg.statement_quantile);
        threshold_vector[t] = threshold;
        for i in 0..n {
            statement_matrix[t][i] = score_matrix[t][i] >= threshold;
        }
        self_heat[t] = score_matrix[t][anchor_idx];
        group_heat[t] = mean_of_selected(&score_matrix[t], &peers);
        book_imbalance_rank_vector[t] = ranked_fields[7][anchor_idx];
        turnover_burst_rank_vector[t] = ranked_fields[2][anchor_idx];
        jump_bias_rank_vector[t] = ranked_fields[5][anchor_idx];
        let score_rank = rank_pct_average(&score_matrix[t]);
        self_heat_rank_vector[t] = score_rank[anchor_idx];
    }

    let mut all_scores = Vec::with_capacity(t_len * n);
    for row in &score_matrix {
        all_scores.extend(row.iter().copied());
    }
    let hot_threshold = quantile_linear(&all_scores, 0.8);
    let cold_threshold = quantile_linear(&all_scores, 0.4);
    let market_heat: Vec<f64> = score_matrix.iter().map(|row| mean(row)).collect();

    let mut peer_strength_vector = vec![0.0; t_len];
    let mut self_score_vector = vec![0.0; t_len];
    let mut group_score_vector = vec![0.0; t_len];
    let mut loneliness_score_vector = vec![0.0; t_len];
    let mut statement_score_vector = vec![0.0; t_len];
    let mut return_gap_vector = vec![0.0; t_len];
    let mut volume_share_vector = vec![0.0; t_len];
    let mut peer_jaccard_vector = vec![1.0; t_len];
    let mut act_buy_advantage_vector = vec![0.0; t_len];
    let mut trade_burst_gap_vector = vec![0.0; t_len];
    let mut amplitude_ratio_vector = vec![0.0; t_len];
    let mut group_count_vector = vec![0.0; t_len];
    let mut lead_lag_score_vector = vec![0.0; t_len];
    let mut peer_return_std_vector = vec![0.0; t_len];
    let mut book_tilt_diff_vector = vec![0.0; t_len];
    let mut volume_weighted_peer_strength_vector = vec![0.0; t_len];

    let mut peer_jaccard = Vec::new();
    let mut lonely_flags = Vec::new();
    let mut leader_flags = Vec::new();
    let mut follower_flags = Vec::new();
    let mut false_start_flags = Vec::new();
    let mut relay_lengths = Vec::new();
    let mut latent_consensus = Vec::new();
    let mut solo_surge = Vec::new();
    let mut reverse_clearheaded = Vec::new();
    let mut late_confirmation = Vec::new();
    let mut leader_convictions = Vec::new();
    let mut follower_remorse = Vec::new();
    let mut false_start_amplitudes = Vec::new();
    let mut echo_lags = Vec::new();
    let mut crowd_sync = Vec::new();
    let mut contradiction_flags = Vec::new();
    let mut dispersion_values = Vec::new();
    let mut morning_rank = Vec::new();
    let mut afternoon_rank = Vec::new();

    for t in 0..t_len {
        let current_peers = &peer_lists[t];
        peer_strength_vector[t] = peer_strength[t];
        self_score_vector[t] = self_heat[t];
        group_score_vector[t] = group_heat[t];
        loneliness_score_vector[t] = 1.0 - peer_strength[t];
        statement_score_vector[t] = self_heat[t] - threshold_vector[t];
        group_count_vector[t] = current_peers.len() as f64;
        return_gap_vector[t] = r1[anchor_idx][t] - mean_of_selected_at(r1, t, current_peers);
        let peer_volume_sum: f64 = current_peers
            .iter()
            .map(|peer| filled[*peer].volume[t])
            .sum();
        volume_share_vector[t] = filled[anchor_idx].volume[t] / (peer_volume_sum + 1e-6);
        act_buy_advantage_vector[t] =
            act_buy_ratio[anchor_idx][t] - mean_of_selected_at(act_buy_ratio, t, current_peers);
        trade_burst_gap_vector[t] =
            trade_burst[anchor_idx][t] - mean_of_selected_at(trade_burst, t, current_peers);
        let peer_amp_mean = mean_of_selected_at(amp, t, current_peers);
        amplitude_ratio_vector[t] = amp[anchor_idx][t] / (peer_amp_mean + 1e-6);
        peer_return_std_vector[t] = std_of_selected_at(r1, t, current_peers);
        book_tilt_diff_vector[t] =
            book_tilt_10[anchor_idx][t] - mean_of_selected_at(book_tilt_10, t, current_peers);
        let weighted_sum: f64 = current_peers
            .iter()
            .zip(peer_score_lists[t].iter())
            .map(|(peer, score)| filled[*peer].volume[t].max(0.0) * *score)
            .sum();
        let weight_total: f64 = current_peers
            .iter()
            .map(|peer| filled[*peer].volume[t].max(0.0))
            .sum();
        volume_weighted_peer_strength_vector[t] = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            peer_strength[t]
        };

        if t > 0 {
            let prev_set: HashSet<usize> = peer_lists[t - 1].iter().copied().collect();
            let cur_set: HashSet<usize> = current_peers.iter().copied().collect();
            let inter = prev_set.intersection(&cur_set).count() as f64;
            let union = prev_set.union(&cur_set).count().max(1) as f64;
            let jacc = inter / union;
            peer_jaccard.push(jacc);
            peer_jaccard_vector[t] = jacc;
        }

        let lonely = peer_strength[t] < cfg.loneliness_threshold;
        lonely_flags.push(lonely);
        let group_is_hot = group_heat[t] >= hot_threshold;
        let self_statement = statement_matrix[t][anchor_idx];
        let peer_statement_share = if current_peers.is_empty() {
            0.0
        } else {
            current_peers
                .iter()
                .map(|peer| statement_matrix[t][*peer] as i32 as f64)
                .sum::<f64>()
                / current_peers.len() as f64
        };
        crowd_sync.push(peer_statement_share);
        dispersion_values.push(std_sample(&score_matrix[t]));
        contradiction_flags.push(
            ((self_statement && group_heat[t] <= cold_threshold)
                || (group_is_hot && self_heat[t] <= cold_threshold)) as i32 as f64,
        );
        latent_consensus.push((group_is_hot && !self_statement) as i32 as f64);
        solo_surge.push((self_statement && group_heat[t] <= cold_threshold) as i32 as f64);
        reverse_clearheaded.push((group_is_hot && self_heat[t] <= cold_threshold) as i32 as f64);
        if t < t_len / 2 {
            morning_rank.push(self_heat_rank_vector[t]);
        } else {
            afternoon_rank.push(self_heat_rank_vector[t]);
        }

        let future_range = (t + 1)..usize::min(t + 1 + cfg.relay_horizon, t_len);
        let past_start = t.saturating_sub(cfg.relay_horizon);
        let past_range = past_start..t;
        let mut future_vals = Vec::new();
        for tt in future_range.clone() {
            for peer in current_peers {
                future_vals.push(statement_matrix[tt][*peer] as i32 as f64);
            }
        }
        let mut past_vals = Vec::new();
        for tt in past_range.clone() {
            for peer in current_peers {
                past_vals.push(statement_matrix[tt][*peer] as i32 as f64);
            }
        }
        let future_response = mean(&future_vals);
        let past_response = mean(&past_vals);
        lead_lag_score_vector[t] = future_response - past_response;

        if self_statement {
            let leader = future_response >= 0.3 && future_response > past_response;
            let follower = past_response >= 0.3 && past_response > future_response;
            let false_start = future_response < 0.15;
            leader_flags.push(leader as i32 as f64);
            follower_flags.push(follower as i32 as f64);
            false_start_flags.push(false_start as i32 as f64);

            let mut relay_len = 0.0;
            let mut future_peer_step_response = Vec::new();
            for step in 1..=cfg.relay_horizon {
                if t + step >= t_len {
                    break;
                }
                let response = if current_peers.is_empty() {
                    0.0
                } else {
                    mean(
                        &current_peers
                            .iter()
                            .map(|peer| statement_matrix[t + step][*peer] as i32 as f64)
                            .collect::<Vec<_>>(),
                    )
                };
                future_peer_step_response.push(response);
                if response >= 0.3 {
                    relay_len += 1.0;
                } else {
                    break;
                }
            }
            relay_lengths.push(relay_len);
            echo_lags.push(first_lag_above(&future_peer_step_response, 0.3));

            if leader {
                let future_anchor_returns: Vec<f64> = ((t + 1)
                    ..usize::min(t + 1 + cfg.relay_horizon, t_len))
                    .map(|tt| r1[anchor_idx][tt])
                    .collect();
                leader_convictions.push(mean(&future_anchor_returns));
            }
            if follower {
                let future_anchor_returns: Vec<f64> = ((t + 1)
                    ..usize::min(t + 1 + cfg.relay_horizon, t_len))
                    .map(|tt| r1[anchor_idx][tt])
                    .collect();
                follower_remorse.push((mean(&future_anchor_returns) < 0.0) as i32 as f64);
            }
            if false_start {
                false_start_amplitudes.push((self_heat[t] - market_heat[t]).abs());
            }
        }

        if group_is_hot && !self_statement {
            let future_self: Vec<f64> = ((t + 1)..usize::min(t + 1 + cfg.relay_horizon, t_len))
                .map(|tt| statement_matrix[tt][anchor_idx] as i32 as f64)
                .collect();
            late_confirmation.push((mean(&future_self) >= 0.5) as i32 as f64);
        }
    }

    let half = t_len / 2;
    let mut morning_union: HashSet<usize> = HashSet::new();
    let mut afternoon_union: HashSet<usize> = HashSet::new();
    for peers in peer_lists.iter().take(half) {
        morning_union.extend(peers.iter().copied());
    }
    for peers in peer_lists.iter().skip(half) {
        afternoon_union.extend(peers.iter().copied());
    }
    let theme_union = morning_union.union(&afternoon_union).count().max(1) as f64;
    let theme_inter = morning_union.intersection(&afternoon_union).count() as f64;

    let open_set: HashSet<usize> = peer_lists
        .first()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let close_set: HashSet<usize> = peer_lists
        .last()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let open_close_union = open_set.union(&close_set).count().max(1) as f64;
    let open_close_inter = open_set.intersection(&close_set).count() as f64;

    let market_heat_tail =
        mean(&market_heat[market_heat.len().saturating_sub(cfg.relay_horizon)..]);
    let tail_group_heat =
        mean(&group_score_vector[group_score_vector.len().saturating_sub(cfg.relay_horizon)..]);
    let head_group_heat =
        mean(&group_score_vector[..usize::min(cfg.relay_horizon, group_score_vector.len())]);

    let scalars = vec![
        mean(&peer_jaccard),
        mean(&peer_strength_vector),
        peer_strength_vector.iter().copied().fold(0.0, f64::max),
        lonely_flags
            .iter()
            .map(|flag| *flag as i32 as f64)
            .sum::<f64>()
            / lonely_flags.len().max(1) as f64,
        mean(&leader_flags),
        mean(&follower_flags),
        mean(&false_start_flags),
        mean(&relay_lengths),
        1.0 - theme_inter / theme_union,
        open_close_inter / open_close_union,
        mean(&latent_consensus),
        mean(&solo_surge),
        mean(&reverse_clearheaded),
        mean(&late_confirmation),
        mean(&market_heat),
        market_heat_tail,
        std_sample(&peer_strength_vector),
        longest_streak(&lonely_flags),
        mean(&crowd_sync),
        mean(&leader_convictions),
        mean(&follower_remorse),
        mean(&false_start_amplitudes),
        relay_lengths.iter().copied().fold(0.0, f64::max),
        mean(&group_score_vector[half..]) - mean(&group_score_vector[..half]),
        mean(&contradiction_flags),
        mean(&echo_lags),
        mean(&afternoon_rank) - mean(&morning_rank),
        mean(&peer_return_std_vector),
        mean(&dispersion_values),
        tail_group_heat - head_group_heat,
    ];

    let vectors = vec![
        peer_strength_vector,
        self_score_vector,
        group_score_vector,
        loneliness_score_vector,
        statement_score_vector,
        book_imbalance_rank_vector,
        return_gap_vector,
        volume_share_vector,
        peer_jaccard_vector,
        act_buy_advantage_vector,
        trade_burst_gap_vector,
        amplitude_ratio_vector,
        turnover_burst_rank_vector,
        jump_bias_rank_vector,
        self_heat_rank_vector,
        group_count_vector,
        lead_lag_score_vector,
        peer_return_std_vector,
        book_tilt_diff_vector,
        volume_weighted_peer_strength_vector,
    ];

    (scalars, vectors)
}

#[pyfunction]
#[pyo3(signature = (date, symbol, universe_size=100, selection_method="code"))]
pub fn personalized_meeting_features(
    date: i32,
    symbol: &str,
    universe_size: usize,
    selection_method: &str,
) -> PyResult<(Vec<f64>, Vec<String>, Vec<Vec<f64>>, Vec<String>)> {
    let selected_symbols = select_symbols(date, symbol, universe_size, selection_method)?;
    let loaded: Vec<SymbolData> = selected_symbols
        .iter()
        .map(|current_symbol| load_symbol_data(date, current_symbol))
        .collect::<PyResult<Vec<_>>>()?;
    let subset: Vec<&SymbolData> = loaded.iter().collect();
    let precomputed = precompute_universe(&subset, symbol);

    let mut scalar_values = Vec::new();
    let mut vector_values = Vec::new();
    for experiment in EXPERIMENTS {
        let (mut scalars, mut vectors) = compute_role_features(&precomputed, experiment.role);
        scalar_values.append(&mut scalars);
        vector_values.append(&mut vectors);
    }
    Ok((
        scalar_values,
        scalar_feature_names(),
        vector_values,
        vector_feature_names(),
    ))
}
