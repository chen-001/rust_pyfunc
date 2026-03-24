use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

const PEAK_WINDOW: usize = 5;
const FORWARD_SHORT: usize = 10;
const FORWARD_LONG: usize = 20;
const MIN_MOVE_TICK: f64 = 2.0;
const ROUND_NEAR_RATIO: f64 = 0.08;
const GRID_TICK_MULTIPLIERS: [usize; 4] = [2, 3, 4, 5];
const GRID_DIVISORS: [usize; 2] = [8, 6];
const SNAPSHOT_TOLERANCE_NS: i64 = 3_000_000_000;
const CONTEXT_DECAY_TICK: f64 = 2.0;
const EVENT_SMOOTH_SPAN: usize = 9;
const SIDE_RATIO_PRIOR_VOLUME: f64 = 1000.0;
const BOOK_LEVELS: usize = 10;

#[inline]
fn safe_div(num: f64, den: f64) -> f64 {
    if !num.is_finite() || !den.is_finite() || den == 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

#[inline]
fn clean_book_value(v: f64) -> f64 {
    if v == 0.0 {
        f64::NAN
    } else {
        v
    }
}

#[inline]
fn is_close(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let tol = 1e-8 + 1e-5 * b.abs();
    (a - b).abs() <= tol
}

#[inline]
fn clip_positive(v: f64) -> f64 {
    if v.is_nan() {
        f64::NAN
    } else {
        v.max(0.0)
    }
}

fn min_positive_tick(prices: &[f64]) -> Option<f64> {
    let mut uniq: Vec<f64> = prices.iter().copied().filter(|v| v.is_finite()).collect();
    if uniq.len() < 2 {
        return None;
    }
    uniq.sort_by(|a, b| a.total_cmp(b));
    uniq.dedup_by(|a, b| *a == *b);
    if uniq.len() < 2 {
        return None;
    }

    let mut best = f64::INFINITY;
    for i in 1..uniq.len() {
        let diff = uniq[i] - uniq[i - 1];
        if diff > 0.0 && diff < best {
            best = diff;
        }
    }
    if best.is_finite() {
        Some(best)
    } else {
        None
    }
}

fn ewm_mean_adjust_false(values: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let beta = 1.0 - alpha;
    let mut out = vec![f64::NAN; values.len()];
    let mut initialized = false;
    let mut last = f64::NAN;
    let mut nan_gap = 0usize;

    for (i, &raw) in values.iter().enumerate() {
        let x = if raw.is_finite() { raw } else { f64::NAN };
        if x.is_nan() {
            if initialized {
                out[i] = last;
                nan_gap += 1;
            }
            continue;
        }

        if !initialized {
            initialized = true;
            last = x;
            out[i] = x;
            nan_gap = 0;
            continue;
        }

        let effective_alpha = alpha / (alpha + beta.powi((nan_gap + 1) as i32));
        last += effective_alpha * (x - last);
        out[i] = last;
        nan_gap = 0;
    }

    out
}

fn future_extreme(values: &[f64], window: usize, want_min: bool) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let end = (i + window).min(n.saturating_sub(1));
        if i + 1 > end {
            continue;
        }
        let mut acc = if want_min {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        let mut found = false;
        for &v in &values[(i + 1)..=end] {
            if !v.is_finite() {
                continue;
            }
            found = true;
            if want_min {
                acc = acc.min(v);
            } else {
                acc = acc.max(v);
            }
        }
        if found {
            out[i] = acc;
        }
    }
    out
}

fn centered_rolling_extreme(values: &[f64], window: usize, want_min: bool) -> Vec<f64> {
    let n = values.len();
    let half = window / 2;
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window {
        return out;
    }
    for i in half..(n - half) {
        let slice = &values[(i - half)..=(i + half)];
        let mut acc = if want_min {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        let mut valid = true;
        for &v in slice {
            if !v.is_finite() {
                valid = false;
                break;
            }
            if want_min {
                acc = acc.min(v);
            } else {
                acc = acc.max(v);
            }
        }
        if valid {
            out[i] = acc;
        }
    }
    out
}

fn build_feature_names() -> Vec<String> {
    let mut names = vec![
        "peak_context_smooth".to_string(),
        "trough_context_smooth".to_string(),
        "peak_event_smooth".to_string(),
        "peak_strength_short_smooth".to_string(),
        "peak_strength_long_smooth".to_string(),
        "trough_event_smooth".to_string(),
        "trough_strength_short_smooth".to_string(),
        "trough_strength_long_smooth".to_string(),
        "peak_ask1_touch_smooth".to_string(),
        "peak_ask2_touch_smooth".to_string(),
        "trough_bid1_touch_smooth".to_string(),
        "trough_bid2_touch_smooth".to_string(),
        "ask_near_ratio".to_string(),
        "ask_mid_ratio".to_string(),
        "ask_deep_ratio".to_string(),
        "bid_near_ratio".to_string(),
        "bid_mid_ratio".to_string(),
        "bid_deep_ratio".to_string(),
        "ask_max_level_smooth".to_string(),
        "bid_max_level_smooth".to_string(),
        "ask_max_ratio".to_string(),
        "bid_max_ratio".to_string(),
        "ask_weighted_tick_distance".to_string(),
        "bid_weighted_tick_distance".to_string(),
        "book_imbalance_near".to_string(),
        "book_imbalance_all".to_string(),
        "ask_exact_ratio_smooth".to_string(),
        "ask_touch_ratio".to_string(),
        "bid_exact_ratio_smooth".to_string(),
        "bid_touch_ratio".to_string(),
        "trade_buy_ratio_smooth".to_string(),
        "trade_sell_ratio_smooth".to_string(),
        "trade_signed_volume_ratio_smooth".to_string(),
        "trade_signed_turnover_ratio_smooth".to_string(),
        "trade_vwap_gap_tick_smooth".to_string(),
        "peak_trade_sell_pressure_smooth".to_string(),
        "trough_trade_buy_support_smooth".to_string(),
        "peak_deep_wall_smooth".to_string(),
        "trough_deep_support_smooth".to_string(),
        "near_vs_deep_ask".to_string(),
        "near_vs_deep_bid".to_string(),
        "peak_depth_gap_smooth".to_string(),
        "trough_depth_gap_smooth".to_string(),
    ];

    for mult in GRID_TICK_MULTIPLIERS {
        let prefix = format!("tick_{mult}");
        names.push(format!("{prefix}_roundness"));
        names.push(format!("{prefix}_peak_round_signal_smooth"));
        names.push(format!("{prefix}_peak_round_wall_smooth"));
        names.push(format!("{prefix}_peak_nonround_wall_smooth"));
        names.push(format!("{prefix}_trough_round_signal_smooth"));
        names.push(format!("{prefix}_trough_round_support_smooth"));
        names.push(format!("{prefix}_trough_nonround_support_smooth"));
        names.push(format!("{prefix}_round_near_event_bias_smooth"));
    }
    for divisor in GRID_DIVISORS {
        let prefix = format!("range_{divisor}");
        names.push(format!("{prefix}_roundness"));
        names.push(format!("{prefix}_peak_round_signal_smooth"));
        names.push(format!("{prefix}_peak_round_wall_smooth"));
        names.push(format!("{prefix}_peak_nonround_wall_smooth"));
        names.push(format!("{prefix}_trough_round_signal_smooth"));
        names.push(format!("{prefix}_trough_round_support_smooth"));
        names.push(format!("{prefix}_trough_nonround_support_smooth"));
        names.push(format!("{prefix}_round_near_event_bias_smooth"));
    }

    names
}

#[pyfunction]
#[pyo3(signature = (
    trade_exchtime,
    trade_price,
    trade_volume,
    trade_turnover,
    trade_flag,
    market_exchtime,
    market_last_prc,
    ask_prc,
    ask_vol,
    bid_prc,
    bid_vol
))]
pub fn compute_integer_small_peak_features(
    py: Python<'_>,
    trade_exchtime: PyReadonlyArray1<i64>,
    trade_price: PyReadonlyArray1<f64>,
    trade_volume: PyReadonlyArray1<f64>,
    trade_turnover: PyReadonlyArray1<f64>,
    trade_flag: PyReadonlyArray1<i32>,
    market_exchtime: PyReadonlyArray1<i64>,
    market_last_prc: PyReadonlyArray1<f64>,
    ask_prc: PyReadonlyArray2<f64>,
    ask_vol: PyReadonlyArray2<f64>,
    bid_prc: PyReadonlyArray2<f64>,
    bid_vol: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let trade_exchtime = trade_exchtime.as_array();
    let trade_price = trade_price.as_array();
    let trade_volume = trade_volume.as_array();
    let trade_turnover = trade_turnover.as_array();
    let trade_flag = trade_flag.as_array();
    let market_exchtime = market_exchtime.as_array();
    let market_last_prc = market_last_prc.as_array();
    let ask_prc = ask_prc.as_array();
    let ask_vol = ask_vol.as_array();
    let bid_prc = bid_prc.as_array();
    let bid_vol = bid_vol.as_array();

    let n_trade = trade_exchtime.len();
    let n_snap = market_exchtime.len();
    let feature_names = build_feature_names();

    if n_trade == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "trade 数据不能为空",
        ));
    }
    if n_snap == 0 {
        let empty = Array2::<f64>::zeros((0, feature_names.len()));
        return Ok((empty.into_pyarray(py).to_owned(), feature_names));
    }
    if trade_price.len() != n_trade
        || trade_volume.len() != n_trade
        || trade_turnover.len() != n_trade
        || trade_flag.len() != n_trade
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "trade 输入数组长度不一致",
        ));
    }
    if market_last_prc.len() != n_snap {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "market 输入数组长度不一致",
        ));
    }
    if ask_prc.nrows() != n_snap
        || ask_vol.nrows() != n_snap
        || bid_prc.nrows() != n_snap
        || bid_vol.nrows() != n_snap
        || ask_prc.ncols() != BOOK_LEVELS
        || ask_vol.ncols() != BOOK_LEVELS
        || bid_prc.ncols() != BOOK_LEVELS
        || bid_vol.ncols() != BOOK_LEVELS
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "盘口矩阵形状必须是 (n_snapshots, 10)",
        ));
    }

    let trade_prices_vec: Vec<f64> = trade_price.iter().copied().collect();
    let tick_size = min_positive_tick(&trade_prices_vec).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("无法从 trade_price 中推断有效 tick_size")
    })?;

    let last_price: Vec<f64> = market_last_prc.iter().copied().collect();
    let mut day_low = f64::INFINITY;
    let mut day_high = f64::NEG_INFINITY;
    for &v in &last_price {
        if v.is_finite() {
            day_low = day_low.min(v);
            day_high = day_high.max(v);
        }
    }
    let day_range = if day_low.is_finite() && day_high.is_finite() {
        day_high - day_low
    } else {
        0.0
    };

    let mut grid_names = Vec::with_capacity(GRID_TICK_MULTIPLIERS.len() + GRID_DIVISORS.len());
    let mut grid_values = Vec::with_capacity(GRID_TICK_MULTIPLIERS.len() + GRID_DIVISORS.len());
    for mult in GRID_TICK_MULTIPLIERS {
        grid_names.push(format!("tick_{mult}"));
        grid_values.push(tick_size * mult as f64);
    }
    for divisor in GRID_DIVISORS {
        grid_names.push(format!("range_{divisor}"));
        let raw = day_range / divisor as f64;
        let grid = ((raw / tick_size).round_ties_even() * tick_size).max(tick_size * 2.0);
        grid_values.push(grid);
    }

    let mut total_volume = vec![0.0; n_snap];
    let mut total_turnover = vec![0.0; n_snap];
    let mut buy_volume = vec![0.0; n_snap];
    let mut sell_volume = vec![0.0; n_snap];
    let mut buy_turnover = vec![0.0; n_snap];
    let mut sell_turnover = vec![0.0; n_snap];
    let market_times: Vec<i64> = market_exchtime.iter().copied().collect();
    let mut snap_ptr = 0usize;
    for i in 0..n_trade {
        let t = trade_exchtime[i];
        while snap_ptr + 1 < n_snap && market_times[snap_ptr + 1] <= t {
            snap_ptr += 1;
        }
        if market_times[snap_ptr] <= t && t - market_times[snap_ptr] <= SNAPSHOT_TOLERANCE_NS {
            let vol = trade_volume[i];
            let turnover = trade_turnover[i];
            total_volume[snap_ptr] += vol;
            total_turnover[snap_ptr] += turnover;
            match trade_flag[i] {
                66 => {
                    buy_volume[snap_ptr] += vol;
                    buy_turnover[snap_ptr] += turnover;
                }
                83 => {
                    sell_volume[snap_ptr] += vol;
                    sell_turnover[snap_ptr] += turnover;
                }
                _ => {}
            }
        }
    }

    let mut trade_vwap = vec![0.0; n_snap];
    for i in 0..n_snap {
        trade_vwap[i] = if total_volume[i] > 0.0 {
            total_turnover[i] / total_volume[i]
        } else {
            last_price[i]
        };
    }

    let mut ask_prc_clean = vec![f64::NAN; n_snap * BOOK_LEVELS];
    let mut bid_prc_clean = vec![f64::NAN; n_snap * BOOK_LEVELS];

    let mut ask_total = vec![0.0; n_snap];
    let mut bid_total = vec![0.0; n_snap];
    let mut ask_near = vec![0.0; n_snap];
    let mut ask_mid = vec![0.0; n_snap];
    let mut ask_deep = vec![0.0; n_snap];
    let mut bid_near = vec![0.0; n_snap];
    let mut bid_mid = vec![0.0; n_snap];
    let mut bid_deep = vec![0.0; n_snap];
    let mut ask_weighted_tick_distance = vec![f64::NAN; n_snap];
    let mut bid_weighted_tick_distance = vec![f64::NAN; n_snap];
    let mut ask_max_level = vec![1.0; n_snap];
    let mut bid_max_level = vec![1.0; n_snap];
    let mut ask_max_ratio = vec![f64::NAN; n_snap];
    let mut bid_max_ratio = vec![f64::NAN; n_snap];
    let mut ask_exact_ratio = vec![f64::NAN; n_snap];
    let mut bid_exact_ratio = vec![f64::NAN; n_snap];
    let mut ask_touch_ratio = vec![f64::NAN; n_snap];
    let mut bid_touch_ratio = vec![f64::NAN; n_snap];

    for i in 0..n_snap {
        let mut ask_weighted_sum = 0.0;
        let mut bid_weighted_sum = 0.0;
        let mut ask_exact_sum = 0.0;
        let mut bid_exact_sum = 0.0;
        let mut ask_touch_sum = 0.0;
        let mut bid_touch_sum = 0.0;
        let mut ask_max_vol = 0.0;
        let mut bid_max_vol = 0.0;
        let lp = last_price[i];

        for level in 0..BOOK_LEVELS {
            let idx = i * BOOK_LEVELS + level;
            let ap = clean_book_value(ask_prc[[i, level]]);
            let av_raw = clean_book_value(ask_vol[[i, level]]);
            let bp = clean_book_value(bid_prc[[i, level]]);
            let bv_raw = clean_book_value(bid_vol[[i, level]]);
            ask_prc_clean[idx] = ap;
            bid_prc_clean[idx] = bp;

            let av = if av_raw.is_finite() { av_raw } else { 0.0 };
            let bv = if bv_raw.is_finite() { bv_raw } else { 0.0 };

            ask_total[i] += av;
            bid_total[i] += bv;

            if level < 3 {
                ask_near[i] += av;
                bid_near[i] += bv;
            } else if level < 6 {
                ask_mid[i] += av;
                bid_mid[i] += bv;
            } else {
                ask_deep[i] += av;
                bid_deep[i] += bv;
            }

            if av > ask_max_vol {
                ask_max_vol = av;
                ask_max_level[i] = (level + 1) as f64;
            }
            if bv > bid_max_vol {
                bid_max_vol = bv;
                bid_max_level[i] = (level + 1) as f64;
            }

            if ap.is_finite() && lp.is_finite() {
                let dist = (ap - lp) / tick_size;
                ask_weighted_sum += av * dist;
                if dist.abs() <= 0.5 {
                    ask_exact_sum += av;
                }
                if (0.0..=2.0).contains(&dist) {
                    ask_touch_sum += av;
                }
            }
            if bp.is_finite() && lp.is_finite() {
                let dist = (lp - bp) / tick_size;
                bid_weighted_sum += bv * dist;
                if dist.abs() <= 0.5 {
                    bid_exact_sum += bv;
                }
                if (0.0..=2.0).contains(&dist) {
                    bid_touch_sum += bv;
                }
            }
        }

        ask_weighted_tick_distance[i] = safe_div(ask_weighted_sum, ask_total[i]);
        bid_weighted_tick_distance[i] = safe_div(bid_weighted_sum, bid_total[i]);
        ask_max_ratio[i] = safe_div(ask_max_vol, ask_total[i]);
        bid_max_ratio[i] = safe_div(bid_max_vol, bid_total[i]);
        ask_exact_ratio[i] = safe_div(ask_exact_sum, ask_total[i]);
        bid_exact_ratio[i] = safe_div(bid_exact_sum, bid_total[i]);
        ask_touch_ratio[i] = safe_div(ask_touch_sum, ask_total[i]);
        bid_touch_ratio[i] = safe_div(bid_touch_sum, bid_total[i]);
    }

    let future_min_short = future_extreme(&last_price, FORWARD_SHORT, true);
    let future_min_long = future_extreme(&last_price, FORWARD_LONG, true);
    let future_max_short = future_extreme(&last_price, FORWARD_SHORT, false);
    let future_max_long = future_extreme(&last_price, FORWARD_LONG, false);
    let roll_max = centered_rolling_extreme(&last_price, PEAK_WINDOW, false);
    let roll_min = centered_rolling_extreme(&last_price, PEAK_WINDOW, true);

    let mut retreat_short = vec![f64::NAN; n_snap];
    let mut retreat_long = vec![f64::NAN; n_snap];
    let mut rebound_short = vec![f64::NAN; n_snap];
    let mut rebound_long = vec![f64::NAN; n_snap];
    let mut peak_mask = vec![0.0; n_snap];
    let mut trough_mask = vec![0.0; n_snap];
    let mut peak_context = vec![0.0; n_snap];
    let mut trough_context = vec![0.0; n_snap];
    let mut retreat_short_positive = vec![f64::NAN; n_snap];
    let mut retreat_long_positive = vec![f64::NAN; n_snap];
    let mut rebound_short_positive = vec![f64::NAN; n_snap];
    let mut rebound_long_positive = vec![f64::NAN; n_snap];
    let mut peak_strength_short = vec![f64::NAN; n_snap];
    let mut peak_strength_long = vec![f64::NAN; n_snap];
    let mut trough_strength_short = vec![f64::NAN; n_snap];
    let mut trough_strength_long = vec![f64::NAN; n_snap];
    let mut peak_ask1_touch = vec![0.0; n_snap];
    let mut peak_ask2_touch = vec![0.0; n_snap];
    let mut trough_bid1_touch = vec![0.0; n_snap];
    let mut trough_bid2_touch = vec![0.0; n_snap];

    for i in 0..n_snap {
        let price = last_price[i];
        retreat_short[i] = safe_div(price - future_min_short[i], tick_size);
        retreat_long[i] = safe_div(price - future_min_long[i], tick_size);
        rebound_short[i] = safe_div(future_max_short[i] - price, tick_size);
        rebound_long[i] = safe_div(future_max_long[i] - price, tick_size);

        let prev = if i > 0 { last_price[i - 1] } else { f64::NAN };
        let next = if i + 1 < n_snap {
            last_price[i + 1]
        } else {
            f64::NAN
        };
        let is_peak = price.is_finite()
            && roll_max[i].is_finite()
            && prev.is_finite()
            && next.is_finite()
            && price == roll_max[i]
            && price > prev
            && price >= next
            && retreat_short[i].is_finite()
            && retreat_short[i] >= MIN_MOVE_TICK;
        let is_trough = price.is_finite()
            && roll_min[i].is_finite()
            && prev.is_finite()
            && next.is_finite()
            && price == roll_min[i]
            && price < prev
            && price <= next
            && rebound_short[i].is_finite()
            && rebound_short[i] >= MIN_MOVE_TICK;

        peak_mask[i] = if is_peak { 1.0 } else { 0.0 };
        trough_mask[i] = if is_trough { 1.0 } else { 0.0 };

        let peak_gap_tick = if roll_max[i].is_finite() && price.is_finite() {
            ((roll_max[i] - price) / tick_size).max(0.0)
        } else {
            f64::INFINITY
        };
        let trough_gap_tick = if roll_min[i].is_finite() && price.is_finite() {
            ((price - roll_min[i]) / tick_size).max(0.0)
        } else {
            f64::INFINITY
        };

        let pc = (-peak_gap_tick / CONTEXT_DECAY_TICK).exp();
        let tc = (-trough_gap_tick / CONTEXT_DECAY_TICK).exp();
        peak_context[i] = if pc.is_finite() { pc } else { 0.0 };
        trough_context[i] = if tc.is_finite() { tc } else { 0.0 };

        retreat_short_positive[i] = clip_positive(retreat_short[i]);
        retreat_long_positive[i] = clip_positive(retreat_long[i]);
        rebound_short_positive[i] = clip_positive(rebound_short[i]);
        rebound_long_positive[i] = clip_positive(rebound_long[i]);

        peak_strength_short[i] = peak_context[i] * retreat_short_positive[i];
        peak_strength_long[i] = peak_context[i] * retreat_long_positive[i];
        trough_strength_short[i] = trough_context[i] * rebound_short_positive[i];
        trough_strength_long[i] = trough_context[i] * rebound_long_positive[i];

        let base = i * BOOK_LEVELS;
        peak_ask1_touch[i] = peak_context[i]
            * if is_close(ask_prc_clean[base], price) {
                1.0
            } else {
                0.0
            };
        peak_ask2_touch[i] = peak_context[i]
            * if is_close(ask_prc_clean[base + 1], price) {
                1.0
            } else {
                0.0
            };
        trough_bid1_touch[i] = trough_context[i]
            * if is_close(bid_prc_clean[base], price) {
                1.0
            } else {
                0.0
            };
        trough_bid2_touch[i] = trough_context[i]
            * if is_close(bid_prc_clean[base + 1], price) {
                1.0
            } else {
                0.0
            };
    }

    let mut trade_buy_ratio = vec![0.0; n_snap];
    let mut trade_sell_ratio = vec![0.0; n_snap];
    let mut trade_signed_volume_ratio = vec![0.0; n_snap];
    let mut trade_signed_turnover_ratio = vec![0.0; n_snap];
    let mut trade_vwap_gap_tick = vec![f64::NAN; n_snap];
    for i in 0..n_snap {
        let volume_den = total_volume[i] + 2.0 * SIDE_RATIO_PRIOR_VOLUME;
        trade_buy_ratio[i] = (buy_volume[i] + SIDE_RATIO_PRIOR_VOLUME) / volume_den;
        trade_sell_ratio[i] = (sell_volume[i] + SIDE_RATIO_PRIOR_VOLUME) / volume_den;
        trade_signed_volume_ratio[i] = trade_buy_ratio[i] - trade_sell_ratio[i];
        let turnover_den = total_turnover[i] + 2.0 * SIDE_RATIO_PRIOR_VOLUME * last_price[i];
        trade_signed_turnover_ratio[i] = (buy_turnover[i]
            + SIDE_RATIO_PRIOR_VOLUME * last_price[i]
            - sell_turnover[i]
            - SIDE_RATIO_PRIOR_VOLUME * last_price[i])
            / turnover_den;
        trade_vwap_gap_tick[i] = safe_div(trade_vwap[i] - last_price[i], tick_size);
    }

    let mut feature_columns: Vec<Vec<f64>> = Vec::with_capacity(feature_names.len());

    feature_columns.push(ewm_mean_adjust_false(&peak_context, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(&trough_context, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(&peak_mask, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(
        &peak_strength_short,
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &peak_strength_long,
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(&trough_mask, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(
        &trough_strength_short,
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &trough_strength_long,
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(&peak_ask1_touch, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(&peak_ask2_touch, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(&trough_bid1_touch, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(&trough_bid2_touch, EVENT_SMOOTH_SPAN));
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(ask_near[i], ask_total[i]))
            .collect(),
    );
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(ask_mid[i], ask_total[i]))
            .collect(),
    );
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(ask_deep[i], ask_total[i]))
            .collect(),
    );
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(bid_near[i], bid_total[i]))
            .collect(),
    );
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(bid_mid[i], bid_total[i]))
            .collect(),
    );
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(bid_deep[i], bid_total[i]))
            .collect(),
    );
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| ask_max_level[i] / 10.0)
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| bid_max_level[i] / 10.0)
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ask_max_ratio.clone());
    feature_columns.push(bid_max_ratio.clone());
    feature_columns.push(ask_weighted_tick_distance.clone());
    feature_columns.push(bid_weighted_tick_distance.clone());
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(bid_near[i] - ask_near[i], bid_near[i] + ask_near[i]))
            .collect(),
    );
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(bid_total[i] - ask_total[i], bid_total[i] + ask_total[i]))
            .collect(),
    );
    feature_columns.push(ewm_mean_adjust_false(&ask_exact_ratio, EVENT_SMOOTH_SPAN));
    feature_columns.push(ask_touch_ratio.clone());
    feature_columns.push(ewm_mean_adjust_false(&bid_exact_ratio, EVENT_SMOOTH_SPAN));
    feature_columns.push(bid_touch_ratio.clone());
    feature_columns.push(ewm_mean_adjust_false(&trade_buy_ratio, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(&trade_sell_ratio, EVENT_SMOOTH_SPAN));
    feature_columns.push(ewm_mean_adjust_false(
        &trade_signed_volume_ratio,
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &trade_signed_turnover_ratio,
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &trade_vwap_gap_tick,
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| peak_context[i] * trade_sell_ratio[i])
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| trough_context[i] * trade_buy_ratio[i])
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| peak_context[i] * safe_div(ask_deep[i], ask_total[i]) * ask_max_ratio[i])
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| trough_context[i] * safe_div(bid_deep[i], bid_total[i]) * bid_max_ratio[i])
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(ask_near[i], ask_deep[i]))
            .collect(),
    );
    feature_columns.push(
        (0..n_snap)
            .map(|i| safe_div(bid_near[i], bid_deep[i]))
            .collect(),
    );
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| {
                peak_context[i] * (ask_weighted_tick_distance[i] - bid_weighted_tick_distance[i])
            })
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));
    feature_columns.push(ewm_mean_adjust_false(
        &(0..n_snap)
            .map(|i| {
                trough_context[i] * (bid_weighted_tick_distance[i] - ask_weighted_tick_distance[i])
            })
            .collect::<Vec<f64>>(),
        EVENT_SMOOTH_SPAN,
    ));

    for &grid_value in &grid_values {
        let round_distance: Vec<f64> = last_price
            .iter()
            .map(|&p| {
                if !p.is_finite() {
                    f64::NAN
                } else {
                    let rounded = (p / grid_value).round_ties_even() * grid_value;
                    (p - rounded).abs()
                }
            })
            .collect();
        let roundness: Vec<f64> = round_distance
            .iter()
            .map(|&d| {
                if d.is_finite() {
                    (-(d / (grid_value / 2.0))).exp()
                } else {
                    f64::NAN
                }
            })
            .collect();
        let nonroundness: Vec<f64> = roundness.iter().map(|&v| 1.0 - v).collect();
        let round_near_flag: Vec<f64> = round_distance
            .iter()
            .map(|&d| {
                if d.is_finite() && d <= grid_value * ROUND_NEAR_RATIO {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let peak_round_wall: Vec<f64> = (0..n_snap)
            .map(|i| {
                let ask_near_ratio = safe_div(ask_near[i], ask_total[i]);
                roundness[i]
                    * peak_strength_short[i]
                    * (ask_exact_ratio[i] + ask_touch_ratio[i] + ask_near_ratio)
                    / 3.0
            })
            .collect();
        let trough_round_support: Vec<f64> = (0..n_snap)
            .map(|i| {
                let bid_near_ratio = safe_div(bid_near[i], bid_total[i]);
                roundness[i]
                    * trough_strength_short[i]
                    * (bid_exact_ratio[i] + bid_touch_ratio[i] + bid_near_ratio)
                    / 3.0
            })
            .collect();
        let peak_nonround_wall: Vec<f64> = (0..n_snap)
            .map(|i| {
                nonroundness[i]
                    * peak_context[i]
                    * retreat_short_positive[i]
                    * safe_div(ask_deep[i], ask_total[i])
            })
            .collect();
        let trough_nonround_support: Vec<f64> = (0..n_snap)
            .map(|i| {
                nonroundness[i]
                    * trough_context[i]
                    * rebound_short_positive[i]
                    * safe_div(bid_deep[i], bid_total[i])
            })
            .collect();
        let round_near_event_bias: Vec<f64> = (0..n_snap)
            .map(|i| round_near_flag[i] * (peak_context[i] - trough_context[i]))
            .collect();

        feature_columns.push(roundness.clone());
        feature_columns.push(ewm_mean_adjust_false(
            &(0..n_snap)
                .map(|i| roundness[i] * peak_strength_short[i])
                .collect::<Vec<f64>>(),
            EVENT_SMOOTH_SPAN,
        ));
        feature_columns.push(ewm_mean_adjust_false(&peak_round_wall, EVENT_SMOOTH_SPAN));
        feature_columns.push(ewm_mean_adjust_false(
            &peak_nonround_wall,
            EVENT_SMOOTH_SPAN,
        ));
        feature_columns.push(ewm_mean_adjust_false(
            &(0..n_snap)
                .map(|i| roundness[i] * trough_strength_short[i])
                .collect::<Vec<f64>>(),
            EVENT_SMOOTH_SPAN,
        ));
        feature_columns.push(ewm_mean_adjust_false(
            &trough_round_support,
            EVENT_SMOOTH_SPAN,
        ));
        feature_columns.push(ewm_mean_adjust_false(
            &trough_nonround_support,
            EVENT_SMOOTH_SPAN,
        ));
        feature_columns.push(ewm_mean_adjust_false(
            &round_near_event_bias,
            EVENT_SMOOTH_SPAN,
        ));
    }

    let n_features = feature_columns.len();
    if n_features != feature_names.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "特征列数量不匹配: names={}, values={}",
            feature_names.len(),
            n_features
        )));
    }

    let mut result = Array2::<f64>::zeros((n_snap, n_features));
    for col in 0..n_features {
        for row in 0..n_snap {
            let v = feature_columns[col][row];
            result[[row, col]] = if v.is_finite() { v } else { 0.0 };
        }
    }

    Ok((result.into_pyarray(py).to_owned(), feature_names))
}
