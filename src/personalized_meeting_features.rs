use csv::ReaderBuilder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::read_dir;
use std::path::{Path, PathBuf};

const DATA_ROOT: &str = "/ssd_data/stock";

const DIRECT_BASE_NAMES: [&str; 48] = [
    "alice_chapter_count",
    "alice_chapter_avg_duration",
    "alice_chapter_switch_speed",
    "alice_turn_count",
    "alice_avg_buy_ratio",
    "alice_buy_ratio_std",
    "alice_silent_count",
    "alice_silent_ratio",
    "alice_state_entropy",
    "alice_n_unique_states",
    "alice_dominant_state_ratio",
    "gstay_up_urgency",
    "gstay_down_urgency",
    "gstay_urgency_asymmetry",
    "gstay_bid_fragmentation",
    "gstay_ask_fragmentation",
    "gstay_bid_high_freq_count",
    "gstay_ask_high_freq_count",
    "gstay_act_buy_ratio",
    "gstay_act_sell_ratio",
    "gstay_act_asymmetry",
    "gstay_act_buy_avg_size",
    "gstay_act_sell_avg_size",
    "gstay_urgency_ratio",
    "gstay_bid_total_mean",
    "gstay_ask_total_mean",
    "gstay_orderbook_imbalance",
    "gstay_bid_depth_ratio",
    "gstay_ask_depth_ratio",
    "gstay_most_active_interval_idx",
    "gstay_least_active_interval_idx",
    "gstay_total_intervals",
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
];

const VECTOR_BASE_NAMES: [&str; 4] = [
    "alice_chapter_boundary_vector",
    "alice_turn_vector",
    "alice_silent_vector",
    "gstay_time_personality_vector",
];

#[derive(Clone, Copy)]
struct SingleCfg {
    chapter_window: usize,
    chapter_threshold: f64,
    turn_high_threshold: f64,
    turn_low_threshold: f64,
    silence_window: usize,
    silence_ratio: f64,
    fragmentation_threshold: usize,
    time_personality_interval: i32,
}

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
    single: SingleCfg,
    role: RoleCfg,
    universe_size: usize,
}

const EXPERIMENTS: [Experiment; 4] = [
    Experiment {
        tag: "base",
        single: SingleCfg {
            chapter_window: 10,
            chapter_threshold: 1.8,
            turn_high_threshold: 0.6,
            turn_low_threshold: 0.4,
            silence_window: 5,
            silence_ratio: 0.3,
            fragmentation_threshold: 5,
            time_personality_interval: 5,
        },
        role: RoleCfg {
            top_k: 8,
            burst_window: 5,
            statement_quantile: 0.9,
            loneliness_threshold: 0.25,
            relay_horizon: 3,
        },
        universe_size: 64,
    },
    Experiment {
        tag: "sensitive",
        single: SingleCfg {
            chapter_window: 5,
            chapter_threshold: 1.5,
            turn_high_threshold: 0.55,
            turn_low_threshold: 0.45,
            silence_window: 3,
            silence_ratio: 0.25,
            fragmentation_threshold: 3,
            time_personality_interval: 3,
        },
        role: RoleCfg {
            top_k: 6,
            burst_window: 3,
            statement_quantile: 0.85,
            loneliness_threshold: 0.35,
            relay_horizon: 2,
        },
        universe_size: 32,
    },
    Experiment {
        tag: "smooth",
        single: SingleCfg {
            chapter_window: 20,
            chapter_threshold: 2.2,
            turn_high_threshold: 0.7,
            turn_low_threshold: 0.3,
            silence_window: 10,
            silence_ratio: 0.4,
            fragmentation_threshold: 8,
            time_personality_interval: 10,
        },
        role: RoleCfg {
            top_k: 12,
            burst_window: 8,
            statement_quantile: 0.95,
            loneliness_threshold: 0.2,
            relay_horizon: 5,
        },
        universe_size: 96,
    },
    Experiment {
        tag: "crowded",
        single: SingleCfg {
            chapter_window: 10,
            chapter_threshold: 1.8,
            turn_high_threshold: 0.6,
            turn_low_threshold: 0.4,
            silence_window: 5,
            silence_ratio: 0.3,
            fragmentation_threshold: 5,
            time_personality_interval: 5,
        },
        role: RoleCfg {
            top_k: 16,
            burst_window: 5,
            statement_quantile: 0.85,
            loneliness_threshold: 0.4,
            relay_horizon: 4,
        },
        universe_size: 128,
    },
];

#[derive(Clone, Default)]
struct DirectMinuteAgg {
    volume: f64,
    total_count: usize,
    buy_count: usize,
    open: f64,
    close: f64,
    sum_volume: f64,
    first_seen: bool,
}

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
struct DirectData {
    minute_keys: Vec<i32>,
    volumes: Vec<f64>,
    trade_counts: Vec<f64>,
    buy_counts: Vec<f64>,
    opens: Vec<f64>,
    closes: Vec<f64>,
    avg_sizes: Vec<f64>,
    total_volume: f64,
    act_buy_volume: f64,
    act_sell_volume: f64,
    act_buy_sum_size: f64,
    act_buy_count: usize,
    act_sell_sum_size: f64,
    act_sell_count: usize,
    bid_count: HashMap<i64, usize>,
    bid_volume: HashMap<i64, f64>,
    ask_count: HashMap<i64, usize>,
    ask_volume: HashMap<i64, f64>,
    bid_total_mean: f64,
    ask_total_mean: f64,
    bid_depth_ratio: f64,
    ask_depth_ratio: f64,
}

#[derive(Clone)]
struct SymbolData {
    symbol: String,
    role_trade: HashMap<i32, RoleTradeAgg>,
    role_market: HashMap<i32, RoleMarketAgg>,
    direct: Option<DirectData>,
}

#[derive(Clone)]
struct FilledRoleData {
    symbol: String,
    minutes: Vec<i32>,
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

fn direct_feature_names() -> Vec<String> {
    EXPERIMENTS
        .iter()
        .flat_map(|exp| {
            DIRECT_BASE_NAMES
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
    Ok((direct_feature_names(), vector_feature_names()))
}

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.total_cmp(b)
}

fn quantile_linear(values: &[f64], q: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return f64::NAN;
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
    if !lo.is_finite() || !hi.is_finite() {
        return vec![0.0; values.len()];
    }
    let fill = (lo + hi) * 0.5;
    values
        .iter()
        .map(|v| if v.is_finite() { v.clamp(lo, hi) } else { fill })
        .collect()
}

fn rank_pct_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
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
    let mut out = vec![f64::NAN; values.len()];
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

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_sample(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return f64::NAN;
    }
    let m = mean(values);
    let var = values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

fn entropy_from_probabilities(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * (p + 1e-10).log2())
        .sum()
}

fn minute_from_time_value(time_value: i64) -> i32 {
    let hour = (time_value / 10_000_000) as i32;
    let minute = ((time_value / 100_000) % 100) as i32;
    hour * 60 + minute
}

fn intraday_adjusted_minute(raw_minute: i32) -> Option<i32> {
    if (570..=690).contains(&raw_minute) {
        Some(raw_minute)
    } else if (780..=897).contains(&raw_minute) {
        Some(raw_minute - 90)
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

    // 兼容上游把整数列写成 32.0 / 66.0 这类字符串。
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

fn select_symbols(date: i32, anchor_symbol: &str, size: usize) -> PyResult<Vec<String>> {
    let all_symbols = available_symbols(date)?;
    let mut base: Vec<String> = all_symbols
        .into_iter()
        .filter(|symbol| symbol != anchor_symbol)
        .take(size.saturating_sub(1))
        .collect();
    base.push(anchor_symbol.to_string());
    base.sort();
    Ok(base)
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

fn load_symbol_data(date: i32, symbol: &str, include_direct: bool) -> PyResult<SymbolData> {
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

    let mut direct_minutes: BTreeMap<i32, DirectMinuteAgg> = BTreeMap::new();
    let mut total_volume = 0.0;
    let mut act_buy_volume = 0.0;
    let mut act_sell_volume = 0.0;
    let mut act_buy_sum_size = 0.0;
    let mut act_buy_count = 0usize;
    let mut act_sell_sum_size = 0.0;
    let mut act_sell_count = 0usize;
    let mut bid_count: HashMap<i64, usize> = HashMap::new();
    let mut bid_volume: HashMap<i64, f64> = HashMap::new();
    let mut ask_count: HashMap<i64, usize> = HashMap::new();
    let mut ask_volume: HashMap<i64, f64> = HashMap::new();
    let mut bid_total_sum = 0.0;
    let mut ask_total_sum = 0.0;
    let mut bid1_sum = 0.0;
    let mut ask1_sum = 0.0;
    let mut bid5_sum = 0.0;
    let mut ask5_sum = 0.0;
    let mut market_intraday_count = 0usize;
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
        let ask_idx = column_index(&cols, "ask_order", &trade_path)?;
        let bid_idx = column_index(&cols, "bid_order", &trade_path)?;

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
            let raw_minute = minute_from_time_value(time_value);
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
            let Some(ask_order) =
                parse_row_i64(&record, ask_idx, "ask_order", row_number, &trade_path)
            else {
                continue;
            };
            let Some(bid_order) =
                parse_row_i64(&record, bid_idx, "bid_order", row_number, &trade_path)
            else {
                continue;
            };

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

            if include_direct {
                if let Some(adj_minute) = intraday_adjusted_minute(raw_minute) {
                    let direct_agg = direct_minutes.entry(adj_minute).or_default();
                    if !direct_agg.first_seen {
                        direct_agg.open = price;
                        direct_agg.first_seen = true;
                    }
                    direct_agg.close = price;
                    direct_agg.volume += volume;
                    direct_agg.sum_volume += volume;
                    direct_agg.total_count += 1;
                    if flag == 66 {
                        direct_agg.buy_count += 1;
                    }

                    total_volume += volume;
                    if flag == 66 {
                        act_buy_volume += volume;
                        act_buy_sum_size += volume;
                        act_buy_count += 1;
                    } else if flag == 83 {
                        act_sell_volume += volume;
                        act_sell_sum_size += volume;
                        act_sell_count += 1;
                    }

                    *bid_count.entry(bid_order).or_insert(0) += 1;
                    *bid_volume.entry(bid_order).or_insert(0.0) += volume;
                    *ask_count.entry(ask_order).or_insert(0) += 1;
                    *ask_volume.entry(ask_order).or_insert(0.0) += volume;
                }
            }
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
        let total_bid_idx = column_index(&cols, "total_bid_vol", &market_path)?;
        let total_ask_idx = column_index(&cols, "total_ask_vol", &market_path)?;
        let bid1_idx = column_index(&cols, "bid_vol1", &market_path)?;
        let ask1_idx = column_index(&cols, "ask_vol1", &market_path)?;
        let bid5_idx = column_index(&cols, "bid_vol5", &market_path)?;
        let ask5_idx = column_index(&cols, "ask_vol5", &market_path)?;
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
            let raw_minute = minute_from_time_value(time_value);
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

            if include_direct {
                if let Some(_) = intraday_adjusted_minute(raw_minute) {
                    let Some(total_bid_vol) = parse_row_f64(
                        &record,
                        total_bid_idx,
                        "total_bid_vol",
                        row_number,
                        &market_path,
                    ) else {
                        continue;
                    };
                    let Some(total_ask_vol) = parse_row_f64(
                        &record,
                        total_ask_idx,
                        "total_ask_vol",
                        row_number,
                        &market_path,
                    ) else {
                        continue;
                    };
                    let Some(bid5) =
                        parse_row_f64(&record, bid5_idx, "bid_vol5", row_number, &market_path)
                    else {
                        continue;
                    };
                    let Some(ask5) =
                        parse_row_f64(&record, ask5_idx, "ask_vol5", row_number, &market_path)
                    else {
                        continue;
                    };
                    bid_total_sum += total_bid_vol;
                    ask_total_sum += total_ask_vol;
                    bid1_sum += bid1;
                    ask1_sum += ask1;
                    bid5_sum += bid5;
                    ask5_sum += ask5;
                    market_intraday_count += 1;
                }
            }
        }
    }

    let direct = if include_direct {
        let mut minute_keys = Vec::with_capacity(direct_minutes.len());
        let mut volumes = Vec::with_capacity(direct_minutes.len());
        let mut trade_counts = Vec::with_capacity(direct_minutes.len());
        let mut buy_counts = Vec::with_capacity(direct_minutes.len());
        let mut opens = Vec::with_capacity(direct_minutes.len());
        let mut closes = Vec::with_capacity(direct_minutes.len());
        let mut avg_sizes = Vec::with_capacity(direct_minutes.len());
        for (minute, agg) in direct_minutes {
            minute_keys.push(minute);
            volumes.push(agg.volume);
            trade_counts.push(agg.total_count as f64);
            buy_counts.push(agg.buy_count as f64);
            opens.push(agg.open);
            closes.push(agg.close);
            avg_sizes.push(agg.sum_volume / agg.total_count.max(1) as f64);
        }
        Some(DirectData {
            minute_keys,
            volumes,
            trade_counts,
            buy_counts,
            opens,
            closes,
            avg_sizes,
            total_volume,
            act_buy_volume,
            act_sell_volume,
            act_buy_sum_size,
            act_buy_count,
            act_sell_sum_size,
            act_sell_count,
            bid_count,
            bid_volume,
            ask_count,
            ask_volume,
            bid_total_mean: bid_total_sum / market_intraday_count.max(1) as f64,
            ask_total_mean: ask_total_sum / market_intraday_count.max(1) as f64,
            bid_depth_ratio: bid5_sum / (bid1_sum + 1e-6),
            ask_depth_ratio: ask5_sum / (ask1_sum + 1e-6),
        })
    } else {
        None
    };

    Ok(SymbolData {
        symbol: symbol.to_string(),
        role_trade,
        role_market,
        direct,
    })
}

fn compute_direct_features(direct: &DirectData, cfg: SingleCfg) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = direct.minute_keys.len();
    let chapter_ma_count =
        rolling_mean(&direct.trade_counts, cfg.chapter_window, cfg.chapter_window);
    let chapter_ma_volume = rolling_mean(&direct.volumes, cfg.chapter_window, cfg.chapter_window);
    let mut chapter_vector = vec![0.0; n];
    for i in 0..n {
        let count_ma = chapter_ma_count[i];
        let vol_ma = chapter_ma_volume[i];
        if !count_ma.is_nan() {
            let count_ratio = direct.trade_counts[i] / count_ma;
            let vol_ratio = direct.volumes[i] / vol_ma;
            if count_ratio > cfg.chapter_threshold
                || count_ratio < 1.0 / cfg.chapter_threshold
                || vol_ratio > cfg.chapter_threshold
                || vol_ratio < 1.0 / cfg.chapter_threshold
            {
                chapter_vector[i] = 1.0;
            }
        }
    }
    let chapter_count = chapter_vector.iter().sum::<f64>() + 1.0;
    let chapter_avg_duration = n as f64 / chapter_count;
    let chapter_switch_speed = chapter_count / n.max(1) as f64;

    let mut turn_vector = vec![0.0; n];
    let buy_ratios: Vec<f64> = direct
        .buy_counts
        .iter()
        .zip(direct.trade_counts.iter())
        .map(|(buy, total)| buy / total.max(1.0))
        .collect();
    for i in 1..n {
        let prev = buy_ratios[i - 1];
        let now = buy_ratios[i];
        if (prev > cfg.turn_high_threshold && now < cfg.turn_low_threshold)
            || (prev < cfg.turn_low_threshold && now > cfg.turn_high_threshold)
        {
            turn_vector[i] = 1.0;
        }
    }
    let turn_count = turn_vector.iter().sum::<f64>();
    let avg_buy_ratio = mean(&buy_ratios);
    let buy_ratio_std = std_sample(&buy_ratios);

    let silence_ma = rolling_mean(&direct.volumes, cfg.silence_window, cfg.silence_window);
    let mut silence_vector = vec![0.0; n];
    for i in 0..n {
        if !silence_ma[i].is_nan() && direct.volumes[i] < silence_ma[i] * cfg.silence_ratio {
            silence_vector[i] = 1.0;
        }
    }
    let silent_count = silence_vector.iter().sum::<f64>();
    let silent_ratio = silent_count / n.max(1) as f64;

    let trade_min = direct
        .trade_counts
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let trade_max = direct
        .trade_counts
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let buy_min = buy_ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let buy_max = buy_ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let trade_width = ((trade_max - trade_min).abs() + 1e-12) / 5.0;
    let buy_width = ((buy_max - buy_min).abs() + 1e-12) / 5.0;
    let mut state_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..n {
        let a = (((direct.trade_counts[i] - trade_min) / trade_width).floor() as isize).clamp(0, 4)
            as usize;
        let b = (((buy_ratios[i] - buy_min) / buy_width).floor() as isize).clamp(0, 4) as usize;
        *state_counts.entry((a, b)).or_insert(0) += 1;
    }
    let state_probs: Vec<f64> = state_counts
        .values()
        .map(|count| *count as f64 / n.max(1) as f64)
        .collect();
    let state_entropy = entropy_from_probabilities(&state_probs);
    let n_unique_states = state_counts.len() as f64;
    let dominant_state_ratio = state_probs.iter().copied().fold(0.0, f64::max);

    let mut up_sizes = Vec::new();
    let mut down_sizes = Vec::new();
    for i in 0..n {
        if direct.closes[i] > direct.opens[i] {
            up_sizes.push(direct.avg_sizes[i]);
        } else if direct.closes[i] < direct.opens[i] {
            down_sizes.push(direct.avg_sizes[i]);
        }
    }
    let overall_avg = mean(&direct.avg_sizes);
    let up_urgency = mean(&up_sizes) / overall_avg.max(1e-12);
    let down_urgency = mean(&down_sizes) / overall_avg.max(1e-12);
    let urgency_asymmetry = up_urgency - down_urgency;

    let bid_fragmentation = direct
        .bid_count
        .iter()
        .filter(|(_, count)| **count > cfg.fragmentation_threshold)
        .map(|(order_id, _)| direct.bid_volume.get(order_id).copied().unwrap_or(0.0))
        .sum::<f64>()
        / direct.total_volume.max(1e-12);
    let ask_fragmentation = direct
        .ask_count
        .iter()
        .filter(|(_, count)| **count > cfg.fragmentation_threshold)
        .map(|(order_id, _)| direct.ask_volume.get(order_id).copied().unwrap_or(0.0))
        .sum::<f64>()
        / direct.total_volume.max(1e-12);
    let bid_high_freq_count = direct
        .bid_count
        .values()
        .filter(|count| **count > cfg.fragmentation_threshold)
        .count() as f64;
    let ask_high_freq_count = direct
        .ask_count
        .values()
        .filter(|count| **count > cfg.fragmentation_threshold)
        .count() as f64;

    let act_buy_ratio = direct.act_buy_volume / direct.total_volume.max(1e-12);
    let act_sell_ratio = direct.act_sell_volume / direct.total_volume.max(1e-12);
    let act_asymmetry = act_buy_ratio - act_sell_ratio;
    let act_buy_avg_size = direct.act_buy_sum_size / direct.act_buy_count.max(1) as f64;
    let act_sell_avg_size = direct.act_sell_sum_size / direct.act_sell_count.max(1) as f64;
    let urgency_ratio = act_buy_avg_size / act_sell_avg_size.max(1e-12);
    let orderbook_imbalance = (direct.bid_total_mean - direct.ask_total_mean)
        / (direct.bid_total_mean + direct.ask_total_mean + 1e-6);

    let mut interval_counts: BTreeMap<i32, f64> = BTreeMap::new();
    for (minute, count) in direct.minute_keys.iter().zip(direct.trade_counts.iter()) {
        *interval_counts
            .entry(*minute / cfg.time_personality_interval)
            .or_insert(0.0) += *count;
    }
    let time_vector: Vec<f64> = interval_counts.values().copied().collect();
    let mut most_active_interval_idx = 0;
    let mut least_active_interval_idx = 0;
    let mut most_active_value = f64::NEG_INFINITY;
    let mut least_active_value = f64::INFINITY;
    for (interval, count) in &interval_counts {
        if *count > most_active_value {
            most_active_value = *count;
            most_active_interval_idx = *interval;
        }
        if *count < least_active_value {
            least_active_value = *count;
            least_active_interval_idx = *interval;
        }
    }
    let total_intervals = time_vector.len() as f64;

    (
        vec![
            chapter_count,
            chapter_avg_duration,
            chapter_switch_speed,
            turn_count,
            avg_buy_ratio,
            buy_ratio_std,
            silent_count,
            silent_ratio,
            state_entropy,
            n_unique_states,
            dominant_state_ratio,
            up_urgency,
            down_urgency,
            urgency_asymmetry,
            bid_fragmentation,
            ask_fragmentation,
            bid_high_freq_count,
            ask_high_freq_count,
            act_buy_ratio,
            act_sell_ratio,
            act_asymmetry,
            act_buy_avg_size,
            act_sell_avg_size,
            urgency_ratio,
            direct.bid_total_mean,
            direct.ask_total_mean,
            orderbook_imbalance,
            direct.bid_depth_ratio,
            direct.ask_depth_ratio,
            most_active_interval_idx as f64,
            least_active_interval_idx as f64,
            total_intervals,
        ],
        vec![chapter_vector, turn_vector, silence_vector, time_vector],
    )
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
        minutes: minutes.to_vec(),
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

fn mean_of_selected(values: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return f64::NAN;
    }
    indices.iter().map(|&idx| values[idx]).sum::<f64>() / indices.len() as f64
}

fn cosine_top_k(
    anchor_idx: usize,
    anchor: &[f64],
    others: &[Vec<f64>],
    top_k: usize,
) -> (Vec<usize>, f64) {
    let anchor_norm = anchor.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8);
    let mut sims: Vec<(usize, f64)> = others
        .iter()
        .enumerate()
        .filter_map(|(idx, vec)| {
            if idx == anchor_idx {
                None
            } else {
                let denom = vec.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8) * anchor_norm;
                let dot = anchor
                    .iter()
                    .zip(vec.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                Some((idx, dot / denom))
            }
        })
        .collect();
    sims.sort_by(|a, b| cmp_f64(&b.1, &a.1));
    let k = usize::min(top_k, sims.len());
    let selected: Vec<usize> = sims.iter().take(k).map(|(idx, _)| *idx).collect();
    let selected_scores: Vec<f64> = sims.iter().take(k).map(|(_, score)| *score).collect();
    let mean_score = mean(&selected_scores);
    (selected, mean_score)
}

fn compute_role_features(subset: &[&SymbolData], anchor_symbol: &str, cfg: RoleCfg) -> Vec<f64> {
    let mut minute_set: BTreeMap<i32, ()> = BTreeMap::new();
    for symbol in subset {
        for minute in symbol.role_market.keys() {
            minute_set.insert(*minute, ());
        }
    }
    let minutes: Vec<i32> = minute_set.keys().copied().collect();
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
    let mut turn_burst = vec![vec![0.0; t_len]; n];
    let mut trade_burst = vec![vec![0.0; t_len]; n];
    let mut act_buy_ratio = vec![vec![0.0; t_len]; n];
    let mut jump_bias = vec![vec![0.0; t_len]; n];
    let mut book_tilt_1 = vec![vec![0.0; t_len]; n];
    let mut book_tilt_10 = vec![vec![0.0; t_len]; n];

    for (i, symbol) in filled.iter().enumerate() {
        let amount_ma = rolling_mean(&symbol.amount, cfg.burst_window, 1);
        let trade_ma = rolling_mean(&symbol.num_trades, cfg.burst_window, 1);
        for t in 0..t_len {
            let prev_close = if t == 0 {
                symbol.close[t]
            } else {
                symbol.close[t - 1]
            };
            r1[i][t] = if symbol.open[t] != 0.0 {
                symbol.close[t] / symbol.open[t] - 1.0
            } else {
                symbol.close[t] / prev_close.max(1e-6) - 1.0
            };
            amp[i][t] = (symbol.high[t] - symbol.low[t]) / (prev_close.abs() + 1e-6);
            turn_burst[i][t] = symbol.amount[t] / (amount_ma[t] + 1e-6);
            trade_burst[i][t] = symbol.num_trades[t] / (trade_ma[t] + 1e-6);
            act_buy_ratio[i][t] = symbol.act_buy_amount[t]
                / (symbol.act_buy_amount[t] + symbol.act_sell_amount[t] + 1e-6);
            jump_bias[i][t] = (symbol.up_jump[t] - symbol.down_jump[t])
                / (symbol.up_jump[t] + symbol.down_jump[t] + 1.0);
            book_tilt_1[i][t] = (symbol.bid_size_1[t] - symbol.ask_size_1[t])
                / (symbol.bid_size_1[t] + symbol.ask_size_1[t] + 1e-6);
            book_tilt_10[i][t] = (symbol.bid_size_10[t] - symbol.ask_size_10[t])
                / (symbol.bid_size_10[t] + symbol.ask_size_10[t] + 1e-6);
        }
    }

    let mut score_matrix = vec![vec![0.0; n]; t_len];
    let mut statement_matrix = vec![vec![false; n]; t_len];
    let mut peer_lists: Vec<Vec<usize>> = vec![Vec::new(); t_len];
    let mut peer_strength = vec![f64::NAN; t_len];
    let mut self_heat = vec![0.0; t_len];
    let mut group_heat = vec![f64::NAN; t_len];

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
        let vectors: Vec<Vec<f64>> = (0..n)
            .map(|idx| {
                ranked_fields
                    .iter()
                    .map(|field| field[idx])
                    .collect::<Vec<_>>()
            })
            .collect();
        let (peers, mean_sim) = cosine_top_k(anchor_idx, &vectors[anchor_idx], &vectors, cfg.top_k);
        peer_lists[t] = peers.clone();
        peer_strength[t] = mean_sim;
        for i in 0..n {
            score_matrix[t][i] = (ranked_fields[2][i]
                + ranked_fields[4][i]
                + ranked_fields[3][i]
                + ranked_fields[5][i])
                * 0.25;
        }
        let threshold = quantile_linear(&score_matrix[t], cfg.statement_quantile);
        for i in 0..n {
            statement_matrix[t][i] = score_matrix[t][i] >= threshold;
        }
        self_heat[t] = score_matrix[t][anchor_idx];
        group_heat[t] = mean_of_selected(&score_matrix[t], &peers);
    }

    let mut all_scores = Vec::with_capacity(t_len * n);
    for row in &score_matrix {
        all_scores.extend(row.iter().copied());
    }
    let hot_threshold = quantile_linear(&all_scores, 0.8);
    let cold_threshold = quantile_linear(&all_scores, 0.4);
    let market_heat: Vec<f64> = score_matrix.iter().map(|row| mean(row)).collect();

    let mut peer_jaccard = Vec::new();
    let mut lonely = Vec::new();
    let mut leader_flags = Vec::new();
    let mut follower_flags = Vec::new();
    let mut false_start_flags = Vec::new();
    let mut relay_lengths = Vec::new();
    let mut latent_consensus = Vec::new();
    let mut solo_surge = Vec::new();
    let mut reverse_clearheaded = Vec::new();
    let mut late_confirmation = Vec::new();

    for t in 0..t_len {
        let current_peers = &peer_lists[t];
        lonely.push((peer_strength[t] < cfg.loneliness_threshold) as i32 as f64);
        if t > 0 {
            let prev_set: HashSet<usize> = peer_lists[t - 1].iter().copied().collect();
            let cur_set: HashSet<usize> = current_peers.iter().copied().collect();
            let inter = prev_set.intersection(&cur_set).count() as f64;
            let union = prev_set.union(&cur_set).count().max(1) as f64;
            peer_jaccard.push(inter / union);
        }
        if current_peers.is_empty() {
            continue;
        }

        let group_is_hot = group_heat[t].is_finite() && group_heat[t] >= hot_threshold;
        let self_statement = statement_matrix[t][anchor_idx];
        latent_consensus.push((group_is_hot && !self_statement) as i32 as f64);
        solo_surge.push((self_statement && group_heat[t] <= cold_threshold) as i32 as f64);
        reverse_clearheaded.push((group_is_hot && self_heat[t] <= cold_threshold) as i32 as f64);

        if self_statement {
            let future_range = (t + 1)..usize::min(t + 1 + cfg.relay_horizon, t_len);
            let past_start = t.saturating_sub(cfg.relay_horizon);
            let past_range = past_start..t;
            let mut future_vals = Vec::with_capacity(future_range.len() * current_peers.len());
            for tt in future_range.clone() {
                for peer in current_peers {
                    future_vals.push(statement_matrix[tt][*peer] as i32 as f64);
                }
            }
            let mut past_vals = Vec::with_capacity(past_range.len() * current_peers.len());
            for tt in past_range.clone() {
                for peer in current_peers {
                    past_vals.push(statement_matrix[tt][*peer] as i32 as f64);
                }
            }
            let future_response = if future_vals.is_empty() {
                0.0
            } else {
                mean(&future_vals)
            };
            let past_response = if past_vals.is_empty() {
                0.0
            } else {
                mean(&past_vals)
            };
            leader_flags
                .push((future_response >= 0.3 && future_response > past_response) as i32 as f64);
            follower_flags
                .push((past_response >= 0.3 && past_response > future_response) as i32 as f64);
            false_start_flags.push((future_response < 0.15) as i32 as f64);
            let mut relay_len = 0.0;
            for step in 1..=cfg.relay_horizon {
                if t + step >= t_len {
                    break;
                }
                let response = mean(
                    &current_peers
                        .iter()
                        .map(|peer| statement_matrix[t + step][*peer] as i32 as f64)
                        .collect::<Vec<_>>(),
                );
                if response >= 0.3 {
                    relay_len += 1.0;
                } else {
                    break;
                }
            }
            relay_lengths.push(relay_len);
        }

        if group_is_hot && !self_statement {
            let future_self: Vec<f64> = ((t + 1)..usize::min(t + 1 + cfg.relay_horizon, t_len))
                .map(|tt| statement_matrix[tt][anchor_idx] as i32 as f64)
                .collect();
            let confirm = if future_self.is_empty() {
                0.0
            } else {
                (mean(&future_self) >= 0.5) as i32 as f64
            };
            late_confirmation.push(confirm);
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

    vec![
        mean(&peer_jaccard),
        mean(
            &peer_strength
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>(),
        ),
        peer_strength
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max),
        mean(&lonely),
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
        mean(&market_heat[market_heat.len().saturating_sub(cfg.relay_horizon)..]),
    ]
}

fn subset_from_loaded<'a>(
    loaded: &'a [SymbolData],
    anchor_symbol: &str,
    size: usize,
) -> Vec<&'a SymbolData> {
    let mut base: Vec<&SymbolData> = loaded
        .iter()
        .filter(|item| item.symbol != anchor_symbol)
        .take(size.saturating_sub(1))
        .collect();
    let anchor = loaded
        .iter()
        .find(|item| item.symbol == anchor_symbol)
        .unwrap();
    base.push(anchor);
    base.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    base
}

#[pyfunction]
#[pyo3(signature = (date, symbol))]
pub fn personalized_meeting_features(
    date: i32,
    symbol: &str,
) -> PyResult<(Vec<f64>, Vec<String>, Vec<Vec<f64>>, Vec<String>)> {
    let max_universe = EXPERIMENTS
        .iter()
        .map(|item| item.universe_size)
        .max()
        .unwrap();
    let selected_symbols = select_symbols(date, symbol, max_universe)?;
    let loaded: Vec<SymbolData> = selected_symbols
        .iter()
        .map(|current_symbol| load_symbol_data(date, current_symbol, current_symbol == symbol))
        .collect::<PyResult<Vec<_>>>()?;
    let anchor_direct = loaded
        .iter()
        .find(|item| item.symbol == symbol)
        .and_then(|item| item.direct.clone())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "未找到锚点股票的direct特征: date={}, symbol={}",
                date, symbol
            ))
        })?;

    let mut direct_values = Vec::new();
    let mut vector_values = Vec::new();
    for experiment in EXPERIMENTS {
        let subset = subset_from_loaded(&loaded, symbol, experiment.universe_size);
        let (mut single_values, mut single_vectors) =
            compute_direct_features(&anchor_direct, experiment.single);
        let role_values = compute_role_features(&subset, symbol, experiment.role);
        single_values.extend(role_values);
        direct_values.extend(single_values);
        vector_values.append(&mut single_vectors);
    }
    Ok((
        direct_values,
        direct_feature_names(),
        vector_values,
        vector_feature_names(),
    ))
}
