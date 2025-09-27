use std::cmp::Ordering;
use std::f64::NAN;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const LEVEL_CAP: usize = 10;
const NAN_F64: f64 = NAN;

struct BuyIdx;
impl BuyIdx {
    const N_AT_TOUCH: usize = 0;
    const VISIBLE_LEVELS: usize = 1;
    const DEPTH_INIT: usize = 2;
    const DEPTH_MAX: usize = 3;
    const DEPTH_MIN: usize = 4;
    const DEPTH_TWAP: usize = 5;
    const DEPLETION_TOTAL: usize = 6;
    const DEPLETION_TRADED: usize = 7;
    const DEPLETION_CANCEL: usize = 8;
    const DEPLETION_SHARE: usize = 9;
    const REFILL_DEPTH_INIT: usize = 10;
    const REFILL_SPEED: usize = 11;
    const SHIELD_DEPTH: usize = 12;
    const EXEC_AT_X_SELL: usize = 13;
    const TRADE_SIZE_MEDIAN: usize = 14;
    const CONSUMPTION_SPEED: usize = 15;
    const TIME_TO_BREAK: usize = 16;
    const TIME_UNDER_X: usize = 17;
    const TIME_TO_REFILL: usize = 18;
    const UNDERSHOOT_BP: usize = 19;
    const OVERSHOOT_BP: usize = 20;
    const BREAK_FORCE: usize = 21;
    const RECOVERY_RETURN: usize = 22;
    const TOBI_PRE: usize = 23;
    const TOBI_TREND: usize = 24;
    const SPREAD_AT_TOUCH: usize = 25;
    const SPREAD_CHANGE: usize = 26;
    const QUOTE_REV_RATE: usize = 27;
    const RV_POST_REFILL: usize = 28;
    const VPIN_AROUND_A: usize = 29;
    const ADVERSE_POST_FILL: usize = 30;
    const ADVERSE_POST_REFILL: usize = 31;
    const OFI_POST_REFILL: usize = 32;
    const SUPPORT_SURVIVAL: usize = 33;
    const BOUNCE_SUCCESS: usize = 34;
    const NEXT_INTERARRIVAL: usize = 35;
    const QUEUE_SHAPE_SLOPE: usize = 36;
    const LEN: usize = 37;
}

struct SellIdx;
impl SellIdx {
    const N_AT_TOUCH: usize = 0;
    const VISIBLE_LEVELS: usize = 1;
    const DEPTH_INIT: usize = 2;
    const DEPTH_MAX: usize = 3;
    const DEPTH_MIN: usize = 4;
    const DEPTH_TWAP: usize = 5;
    const DEPLETION_TOTAL: usize = 6;
    const DEPLETION_TRADED: usize = 7;
    const DEPLETION_CANCEL: usize = 8;
    const DEPLETION_SHARE: usize = 9;
    const REFILL_DEPTH_INIT: usize = 10;
    const REFILL_SPEED: usize = 11;
    const SHIELD_DEPTH: usize = 12;
    const EXEC_AT_Y_BUY: usize = 13;
    const TRADE_SIZE_MEDIAN: usize = 14;
    const CONSUMPTION_SPEED: usize = 15;
    const TIME_TO_BREAK: usize = 16;
    const TIME_ABOVE_Y: usize = 17;
    const TIME_TO_REFILL: usize = 18;
    const OVERSHOOT_BP: usize = 19;
    const UNDERSHOOT_BP: usize = 20;
    const BREAK_FORCE: usize = 21;
    const RECOVERY_RETURN: usize = 22;
    const TOBI_PRE: usize = 23;
    const TOBI_TREND: usize = 24;
    const SPREAD_AT_TOUCH: usize = 25;
    const SPREAD_CHANGE: usize = 26;
    const QUOTE_REV_RATE: usize = 27;
    const RV_POST_REFILL: usize = 28;
    const VPIN_AROUND_A: usize = 29;
    const ADVERSE_POST_FILL: usize = 30;
    const ADVERSE_POST_REFILL: usize = 31;
    const OFI_POST_REFILL: usize = 32;
    const RESISTANCE_SURVIVAL: usize = 33;
    const BOUNCE_SUCCESS: usize = 34;
    const NEXT_INTERARRIVAL: usize = 35;
    const QUEUE_SHAPE_SLOPE: usize = 36;
    const LEN: usize = 37;
}

#[derive(Clone, Copy, Debug)]
struct BookLevel {
    level: i32,
    price: f64,
    volume: f64,
}

#[derive(Clone, Debug)]
struct Snapshot {
    timestamp: i64,
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

impl Snapshot {
    fn new(timestamp: i64, mut bids: Vec<BookLevel>, mut asks: Vec<BookLevel>) -> Self {
        bids.sort_by_key(|lvl| lvl.level);
        asks.sort_by_key(|lvl| lvl.level);
        Self {
            timestamp,
            bids,
            asks,
        }
    }

    fn best_bid(&self) -> Option<&BookLevel> {
        self.bids.iter().find(|lvl| lvl.level == 1)
    }

    fn best_ask(&self) -> Option<&BookLevel> {
        self.asks.iter().find(|lvl| lvl.level == 1)
    }

    fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b.price + a.price) * 0.5),
            _ => None,
        }
    }

    fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a.price - b.price),
            _ => None,
        }
    }

    fn tob_imbalance(&self) -> Option<f64> {
        let bid_sum: f64 = self.bids.iter().map(|lvl| lvl.volume.max(0.0)).sum();
        let ask_sum: f64 = self.asks.iter().map(|lvl| lvl.volume.max(0.0)).sum();
        let denom = bid_sum + ask_sum;
        if denom > 0.0 {
            Some((bid_sum - ask_sum) / denom)
        } else {
            None
        }
    }

    fn find_bid_by_price(&self, price: f64, eps: f64) -> Option<&BookLevel> {
        self.bids
            .iter()
            .find(|lvl| (lvl.price - price).abs() <= eps)
    }

    fn find_ask_by_price(&self, price: f64, eps: f64) -> Option<&BookLevel> {
        self.asks
            .iter()
            .find(|lvl| (lvl.price - price).abs() <= eps)
    }

    fn visible_bid_levels(&self) -> usize {
        self.bids.len()
    }

    fn visible_ask_levels(&self) -> usize {
        self.asks.len()
    }

    fn sum_bid_volume_from_price(&self, price_floor: f64, eps: f64) -> f64 {
        self.bids
            .iter()
            .filter(|lvl| lvl.price + eps >= price_floor)
            .map(|lvl| lvl.volume.max(0.0))
            .sum()
    }

    fn sum_ask_volume_up_to_price(&self, price_cap: f64, eps: f64) -> f64 {
        self.asks
            .iter()
            .filter(|lvl| lvl.price <= price_cap + eps)
            .map(|lvl| lvl.volume.max(0.0))
            .sum()
    }
}

#[derive(Debug, Clone)]
struct PriceCycleCore {
    price: f64,
    a_trade_idx: usize,
    b_trade_idx: usize,
    c_trade_idx: Option<usize>,
    a_time: i64,
    b_time: i64,
    c_time: Option<i64>,
    b_price: f64,
}

#[derive(Debug, Clone)]
struct PriceCycleFull {
    core: PriceCycleCore,
    d_snapshot_idx: Option<usize>,
    d_time: Option<i64>,
}

#[derive(Debug, Clone)]
struct CycleBuilder {
    price: f64,
    a_trade_idx: usize,
    a_time: i64,
    b_trade_idx: Option<usize>,
    b_time: Option<i64>,
    b_price: Option<f64>,
    c_trade_idx: Option<usize>,
    c_time: Option<i64>,
}

impl CycleBuilder {
    fn new(price: f64, a_trade_idx: usize, a_time: i64) -> Self {
        Self {
            price,
            a_trade_idx,
            a_time,
            b_trade_idx: None,
            b_time: None,
            b_price: None,
            c_trade_idx: None,
            c_time: None,
        }
    }

    fn finalize(self) -> Option<PriceCycleCore> {
        match (self.b_trade_idx, self.b_time, self.b_price) {
            (Some(b_idx), Some(b_time), Some(b_price)) => Some(PriceCycleCore {
                price: self.price,
                a_trade_idx: self.a_trade_idx,
                b_trade_idx: b_idx,
                c_trade_idx: self.c_trade_idx,
                a_time: self.a_time,
                b_time,
                c_time: self.c_time,
                b_price,
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Comparison {
    Above,
    Equal,
    Below,
}

#[derive(Clone, Copy, Debug)]
enum Side {
    Buy,
    Sell,
}

#[inline(always)]
fn compare_with_eps(value: f64, target: f64, eps: f64) -> Comparison {
    if value > target + eps {
        Comparison::Above
    } else if value < target - eps {
        Comparison::Below
    } else {
        Comparison::Equal
    }
}

fn detect_buy_cycles_for_price(
    price_level: f64,
    trades_time: &[i64],
    trades_price: &[f64],
    drop_threshold: f64,
    above_min_ns: i64,
    eps: f64,
) -> Vec<PriceCycleCore> {
    let mut cycles = Vec::new();
    let mut current: Option<CycleBuilder> = None;
    let mut above_start: Option<i64> = None;
    let mut last_relation = Comparison::Below;

    for (idx, (&time, &price)) in trades_time.iter().zip(trades_price.iter()).enumerate() {
        let relation = compare_with_eps(price, price_level, eps);
        let prior_above_start = above_start;

        if relation == Comparison::Equal {
            if let Some(start_time) = prior_above_start {
                if time - start_time >= above_min_ns {
                    if let Some(builder) = current.take() {
                        if let Some(core) = builder.finalize() {
                            cycles.push(core);
                        }
                    }
                    current = Some(CycleBuilder::new(price_level, idx, time));
                }
            }
        }

        if let Some(builder) = current.as_mut() {
            if builder.b_trade_idx.is_none() {
                let threshold_price = price_level - drop_threshold;
                if price <= threshold_price + eps {
                    builder.b_trade_idx = Some(idx);
                    builder.b_time = Some(time);
                    builder.b_price = Some(price);
                }
            } else if builder.c_trade_idx.is_none() {
                if relation == Comparison::Above {
                    builder.c_trade_idx = Some(idx);
                    builder.c_time = Some(time);
                } else if relation == Comparison::Equal {
                    if let Some(start_time) = prior_above_start {
                        if time - start_time >= above_min_ns {
                            let finished = current.take();
                            if let Some(finished_builder) = finished {
                                if let Some(core) = finished_builder.finalize() {
                                    cycles.push(core);
                                }
                            }
                            current = Some(CycleBuilder::new(price_level, idx, time));
                        }
                    }
                }
            } else {
                if relation == Comparison::Equal {
                    if let Some(start_time) = prior_above_start {
                        if time - start_time >= above_min_ns {
                            let finished = current.take();
                            if let Some(finished_builder) = finished {
                                if let Some(core) = finished_builder.finalize() {
                                    cycles.push(core);
                                }
                            }
                            current = Some(CycleBuilder::new(price_level, idx, time));
                        }
                    }
                }
            }
        }

        match relation {
            Comparison::Above => {
                if !matches!(last_relation, Comparison::Above) {
                    above_start = Some(time);
                }
            }
            _ => {
                above_start = None;
            }
        }
        last_relation = relation;
    }

    if let Some(builder) = current {
        if let Some(core) = builder.finalize() {
            cycles.push(core);
        }
    }

    cycles
}

fn detect_sell_cycles_for_price(
    price_level: f64,
    trades_time: &[i64],
    trades_price: &[f64],
    rise_threshold: f64,
    above_min_ns: i64,
    eps: f64,
) -> Vec<PriceCycleCore> {
    let mut cycles = Vec::new();
    let mut current: Option<CycleBuilder> = None;
    let mut below_start: Option<i64> = None;
    let mut last_relation = Comparison::Above;

    for (idx, (&time, &price)) in trades_time.iter().zip(trades_price.iter()).enumerate() {
        let relation = compare_with_eps(price, price_level, eps);
        let prior_below_start = below_start;

        if relation == Comparison::Equal {
            if let Some(start_time) = prior_below_start {
                if time - start_time >= above_min_ns {
                    if let Some(builder) = current.take() {
                        if let Some(core) = builder.finalize() {
                            cycles.push(core);
                        }
                    }
                    current = Some(CycleBuilder::new(price_level, idx, time));
                }
            }
        }

        if let Some(builder) = current.as_mut() {
            if builder.b_trade_idx.is_none() {
                let threshold_price = price_level + rise_threshold;
                if price >= threshold_price - eps {
                    builder.b_trade_idx = Some(idx);
                    builder.b_time = Some(time);
                    builder.b_price = Some(price);
                }
            } else if builder.c_trade_idx.is_none() {
                if relation == Comparison::Below {
                    builder.c_trade_idx = Some(idx);
                    builder.c_time = Some(time);
                } else if relation == Comparison::Equal {
                    if let Some(start_time) = prior_below_start {
                        if time - start_time >= above_min_ns {
                            let finished = current.take();
                            if let Some(finished_builder) = finished {
                                if let Some(core) = finished_builder.finalize() {
                                    cycles.push(core);
                                }
                            }
                            current = Some(CycleBuilder::new(price_level, idx, time));
                        }
                    }
                }
            } else {
                if relation == Comparison::Equal {
                    if let Some(start_time) = prior_below_start {
                        if time - start_time >= above_min_ns {
                            let finished = current.take();
                            if let Some(finished_builder) = finished {
                                if let Some(core) = finished_builder.finalize() {
                                    cycles.push(core);
                                }
                            }
                            current = Some(CycleBuilder::new(price_level, idx, time));
                        }
                    }
                }
            }
        }

        match relation {
            Comparison::Below => {
                if !matches!(last_relation, Comparison::Below) {
                    below_start = Some(time);
                }
            }
            _ => {
                below_start = None;
            }
        }
        last_relation = relation;
    }

    if let Some(builder) = current {
        if let Some(core) = builder.finalize() {
            cycles.push(core);
        }
    }

    cycles
}

#[derive(Default, Debug)]
struct QueueComputationResult {
    visible: bool,
    level_at_touch: Option<i32>,
    visible_levels: Option<f64>,
    depth_init: Option<f64>,
    depth_max: Option<f64>,
    depth_min: Option<f64>,
    depth_twap: Option<f64>,
    depletion_total: Option<f64>,
    shield_depth: Option<f64>,
    queue_shape_slope: Option<f64>,
}

fn compute_queue_shape_slope(snapshot: &Snapshot, target_level: i32, side: Side) -> Option<f64> {
    let mut xs = [0.0f64; 5];
    let mut ys = [0.0f64; 5];
    let mut count = 0usize;

    let levels_iter: Box<dyn Iterator<Item = &BookLevel>> = match side {
        Side::Buy => Box::new(snapshot.bids.iter()),
        Side::Sell => Box::new(snapshot.asks.iter()),
    };

    for lvl in levels_iter {
        if (lvl.level - target_level).abs() <= 2 {
            if lvl.volume > 0.0 {
                if count < xs.len() {
                    xs[count] = lvl.level as f64;
                    ys[count] = (lvl.volume.max(1e-9)).ln();
                    count += 1;
                }
            }
        }
    }

    if count < 2 {
        return None;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    for i in 0..count {
        let x = xs[i];
        let y = ys[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    let n = count as f64;
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-9 {
        None
    } else {
        Some((n * sum_xy - sum_x * sum_y) / denom)
    }
}

fn compute_queue_metrics(
    snapshots: &[Snapshot],
    start_time: i64,
    end_time: i64,
    price: f64,
    side: Side,
    eps: f64,
) -> QueueComputationResult {
    let mut result = QueueComputationResult::default();
    let Some(start_idx) = asof_snapshot_index(snapshots, start_time) else {
        return result;
    };

    let end_idx = upper_bound_snapshot(snapshots, end_time);
    if start_idx >= snapshots.len() {
        return result;
    }

    let mut twap_num = 0.0;
    let mut twap_den = 0.0;
    let mut prev_depth_for_depletion: Option<f64> = None;

    let mut idx = start_idx;
    let mut first_snapshot_processed = false;

    while idx < snapshots.len() && idx < end_idx {
        let snapshot = &snapshots[idx];
        let depth_opt = match side {
            Side::Buy => snapshot.find_bid_by_price(price, eps),
            Side::Sell => snapshot.find_ask_by_price(price, eps),
        };

        if !first_snapshot_processed {
            result.visible_levels = Some(match side {
                Side::Buy => snapshot.visible_bid_levels() as f64,
                Side::Sell => snapshot.visible_ask_levels() as f64,
            });

            match side {
                Side::Buy => {
                    result.shield_depth = Some(snapshot.sum_bid_volume_from_price(price, eps));
                }
                Side::Sell => {
                    result.shield_depth = Some(snapshot.sum_ask_volume_up_to_price(price, eps));
                }
            }

            if let Some(lvl) = depth_opt {
                result.level_at_touch = Some(lvl.level);
                result.depth_init = Some(lvl.volume);
                if let Some(slope) = compute_queue_shape_slope(snapshot, lvl.level, side) {
                    result.queue_shape_slope = Some(slope);
                }
            }
            first_snapshot_processed = true;
        }

        if let Some(lvl) = depth_opt {
            let vol = lvl.volume;
            result.visible = true;
            result.depth_max = Some(result.depth_max.map_or(vol, |cur| cur.max(vol)));
            result.depth_min = Some(result.depth_min.map_or(vol, |cur| cur.min(vol)));
            if let Some(prev) = prev_depth_for_depletion {
                if vol < prev {
                    let delta = prev - vol;
                    result.depletion_total = Some(result.depletion_total.unwrap_or(0.0) + delta);
                }
            }
            prev_depth_for_depletion = Some(vol);
        } else {
            prev_depth_for_depletion = None;
        }

        let current_time = snapshot.timestamp.max(start_time);
        let next_time = if idx + 1 < snapshots.len() {
            snapshots[idx + 1].timestamp.min(end_time)
        } else {
            end_time
        };
        let interval_end = if next_time < end_time {
            next_time
        } else {
            end_time
        };
        let interval_start = current_time;
        let delta = (interval_end - interval_start).max(0);
        if delta > 0 {
            if let Some(lvl) = depth_opt {
                twap_num += lvl.volume * delta as f64;
                twap_den += delta as f64;
            }
        }

        if snapshot.timestamp >= end_time {
            break;
        }
        idx += 1;
    }

    if result.visible {
        if twap_den > 0.0 {
            result.depth_twap = Some(twap_num / twap_den);
        }
        if result.depletion_total.is_none() {
            result.depletion_total = Some(0.0);
        }
    } else {
        // ensure queue-related metrics remain None if never visible
        result.level_at_touch = None;
        result.visible_levels = None;
        result.depth_init = None;
        result.depth_max = None;
        result.depth_min = None;
        result.depth_twap = None;
        result.depletion_total = None;
        result.shield_depth = None;
        result.queue_shape_slope = None;
    }

    result
}

fn locate_refill_snapshot(
    snapshots: &[Snapshot],
    start_time: i64,
    price: f64,
    side: Side,
    eps: f64,
) -> Option<(usize, i64)> {
    if snapshots.is_empty() {
        return None;
    }
    let mut idx = lower_bound_snapshot(snapshots, start_time);
    while idx < snapshots.len() {
        let snapshot = &snapshots[idx];
        let found = match side {
            Side::Buy => snapshot.find_bid_by_price(price, eps).is_some(),
            Side::Sell => snapshot.find_ask_by_price(price, eps).is_some(),
        };
        if found {
            return Some((idx, snapshot.timestamp));
        }
        idx += 1;
    }
    None
}

fn linear_regression_slope(times: &[i64], values: &[f64]) -> Option<f64> {
    if times.len() < 2 || values.len() != times.len() {
        return None;
    }
    let t0 = times[0] as f64;
    let mut sum_t = 0.0;
    let mut sum_v = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_tv = 0.0;
    for (&t_ns, &v) in times.iter().zip(values.iter()) {
        let t = (t_ns as f64 - t0) / 1_000_000.0; // use milliseconds baseline
        sum_t += t;
        sum_v += v;
        sum_tt += t * t;
        sum_tv += t * v;
    }
    let n = times.len() as f64;
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-9 {
        None
    } else {
        Some((n * sum_tv - sum_t * sum_v) / denom)
    }
}

fn compute_refill_speed(
    snapshots: &[Snapshot],
    start_time: i64,
    window_ns: i64,
    price: f64,
    side: Side,
    eps: f64,
) -> Option<f64> {
    if window_ns <= 0 {
        return None;
    }
    let end_time = start_time + window_ns;
    let mut times = Vec::new();
    let mut volumes = Vec::new();

    let mut idx = lower_bound_snapshot(snapshots, start_time);
    if idx > 0 {
        idx -= 1;
    }
    while idx < snapshots.len() {
        let snap = &snapshots[idx];
        if snap.timestamp > end_time {
            break;
        }
        let depth_opt = match side {
            Side::Buy => snap.find_bid_by_price(price, eps),
            Side::Sell => snap.find_ask_by_price(price, eps),
        };
        if let Some(level) = depth_opt {
            if snap.timestamp >= start_time {
                times.push(snap.timestamp);
                volumes.push(level.volume);
            } else if snap.timestamp < start_time {
                // include starting point at start_time using same volume
                times.push(start_time);
                volumes.push(level.volume);
            }
        }
        idx += 1;
    }

    if times.len() < 2 {
        None
    } else {
        linear_regression_slope(&times, &volumes)
    }
}

fn realized_variance(
    snapshots: &[Snapshot],
    log_mid: &[Option<f64>],
    start_time: i64,
    window_ns: i64,
) -> Option<f64> {
    if snapshots.is_empty() || log_mid.len() != snapshots.len() {
        return None;
    }
    let end_time = start_time + window_ns;
    let asof_idx = match asof_snapshot_index(snapshots, start_time) {
        Some(idx) => idx,
        None => return None,
    };
    let mut last_log = log_mid[asof_idx]?;
    let mut rv = 0.0;
    let mut has = false;

    let mut idx = asof_idx + 1;
    while idx < snapshots.len() {
        let snap = &snapshots[idx];
        if snap.timestamp > end_time {
            break;
        }
        if let Some(log_val) = log_mid[idx] {
            if snap.timestamp >= start_time {
                let diff = log_val - last_log;
                rv += diff * diff;
                has = true;
            }
            last_log = log_val;
        }
        idx += 1;
    }

    if has {
        Some(rv)
    } else {
        None
    }
}

fn mid_price_asof(snapshots: &[Snapshot], snapshot_mid: &[Option<f64>], time: i64) -> Option<f64> {
    let idx = asof_snapshot_index(snapshots, time)?;
    snapshot_mid.get(idx).and_then(|opt| *opt)
}

fn build_snapshots(
    ask_exchtime: &[i64],
    bid_exchtime: &[i64],
    ask_price: &[f64],
    ask_volume: &[f64],
    ask_number: &[i32],
    bid_price: &[f64],
    bid_volume: &[f64],
    bid_number: &[i32],
) -> Vec<Snapshot> {
    let mut snapshots = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    let ask_len = ask_exchtime.len();
    let bid_len = bid_exchtime.len();

    while i < ask_len || j < bid_len {
        let next_time = match (ask_exchtime.get(i), bid_exchtime.get(j)) {
            (Some(&at), Some(&bt)) => at.min(bt),
            (Some(&at), None) => at,
            (None, Some(&bt)) => bt,
            (None, None) => break,
        };

        let mut bids = Vec::with_capacity(LEVEL_CAP);
        while j < bid_len && bid_exchtime[j] == next_time {
            bids.push(BookLevel {
                level: bid_number[j],
                price: bid_price[j],
                volume: bid_volume[j],
            });
            j += 1;
        }

        let mut asks = Vec::with_capacity(LEVEL_CAP);
        while i < ask_len && ask_exchtime[i] == next_time {
            asks.push(BookLevel {
                level: ask_number[i],
                price: ask_price[i],
                volume: ask_volume[i],
            });
            i += 1;
        }

        if !bids.is_empty() || !asks.is_empty() {
            snapshots.push(Snapshot::new(next_time, bids, asks));
        }
    }

    snapshots
}

#[derive(Clone, Debug, Default)]
struct TradePrefixes {
    vol_buy: Vec<f64>,
    vol_sell: Vec<f64>,
}

fn build_trade_prefixes(prices: &[f64], volumes: &[f64], flags: &[i32]) -> TradePrefixes {
    let mut vol_buy = Vec::with_capacity(prices.len() + 1);
    let mut vol_sell = Vec::with_capacity(prices.len() + 1);
    let mut buy_acc = 0.0f64;
    let mut sell_acc = 0.0f64;
    vol_buy.push(0.0);
    vol_sell.push(0.0);
    for i in 0..prices.len() {
        let flag = flags[i];
        let vol = volumes[i];
        if flag == 66 {
            buy_acc += vol;
        } else if flag == 83 {
            sell_acc += vol;
        }
        vol_buy.push(buy_acc);
        vol_sell.push(sell_acc);
    }
    TradePrefixes { vol_buy, vol_sell }
}

#[inline(always)]
fn prefix_range(prefix: &[f64], start: usize, end: usize) -> f64 {
    if end <= start {
        0.0
    } else {
        prefix[end] - prefix[start]
    }
}

fn lower_bound_trade(times: &[i64], ts: i64) -> usize {
    let mut left = 0usize;
    let mut right = times.len();
    while left < right {
        let mid = (left + right) / 2;
        if times[mid] < ts {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn upper_bound_trade(times: &[i64], ts: i64) -> usize {
    let mut left = 0usize;
    let mut right = times.len();
    while left < right {
        let mid = (left + right) / 2;
        if times[mid] <= ts {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn asof_snapshot_index(snapshots: &[Snapshot], ts: i64) -> Option<usize> {
    if snapshots.is_empty() {
        return None;
    }
    let mut left = 0usize;
    let mut right = snapshots.len();
    while left < right {
        let mid = (left + right) / 2;
        if snapshots[mid].timestamp <= ts {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    if left == 0 {
        None
    } else {
        Some(left - 1)
    }
}

fn lower_bound_snapshot(snapshots: &[Snapshot], ts: i64) -> usize {
    let mut left = 0usize;
    let mut right = snapshots.len();
    while left < right {
        let mid = (left + right) / 2;
        if snapshots[mid].timestamp < ts {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn upper_bound_snapshot(snapshots: &[Snapshot], ts: i64) -> usize {
    let mut left = 0usize;
    let mut right = snapshots.len();
    while left < right {
        let mid = (left + right) / 2;
        if snapshots[mid].timestamp <= ts {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[derive(Debug, Clone, Default)]
struct CycleMetrics {
    values: Vec<f64>,
}

impl CycleMetrics {
    fn new(feature_len: usize) -> Self {
        Self {
            values: vec![NAN_F64; feature_len],
        }
    }

    fn set(&mut self, idx: usize, value: f64) {
        if idx < self.values.len() {
            self.values[idx] = value;
        }
    }

    fn get(&self, idx: usize) -> f64 {
        self.values[idx]
    }
}

struct PriceAggregate {
    price: f64,
    metrics_buy: Vec<CycleMetrics>,
    metrics_sell: Vec<CycleMetrics>,
    cycles_buy_count: usize,
    cycles_sell_count: usize,
}

impl PriceAggregate {
    fn new(price: f64) -> Self {
        Self {
            price,
            metrics_buy: Vec::new(),
            metrics_sell: Vec::new(),
            cycles_buy_count: 0,
            cycles_sell_count: 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_buy_cycle_metrics(
    cycle: &PriceCycleFull,
    trades_time: &[i64],
    trades_price: &[f64],
    trades_volume: &[f64],
    trades_flag: &[i32],
    trade_prefix: &TradePrefixes,
    snapshots: &[Snapshot],
    snapshot_mid: &[Option<f64>],
    snapshot_log_mid: &[Option<f64>],
    snapshot_tobi: &[Option<f64>],
    snapshot_spread: &[Option<f64>],
    window_ns: i64,
    eps: f64,
) -> CycleMetrics {
    let mut metrics = CycleMetrics::new(BuyIdx::LEN);
    let core = &cycle.core;
    let price_level = core.price;

    // Queue metrics between A and B
    let queue_result = compute_queue_metrics(
        snapshots,
        core.a_time,
        core.b_time,
        price_level,
        Side::Buy,
        eps,
    );

    if let Some(level) = queue_result.level_at_touch {
        metrics.set(BuyIdx::N_AT_TOUCH, level as f64);
    }
    if let Some(visible) = queue_result.visible_levels {
        metrics.set(BuyIdx::VISIBLE_LEVELS, visible);
    }
    if let Some(depth_init) = queue_result.depth_init {
        metrics.set(BuyIdx::DEPTH_INIT, depth_init);
    }
    if let Some(depth_max) = queue_result.depth_max {
        metrics.set(BuyIdx::DEPTH_MAX, depth_max);
    }
    if let Some(depth_min) = queue_result.depth_min {
        metrics.set(BuyIdx::DEPTH_MIN, depth_min);
    }
    if let Some(depth_twap) = queue_result.depth_twap {
        metrics.set(BuyIdx::DEPTH_TWAP, depth_twap);
    }
    if let Some(depletion_total) = queue_result.depletion_total {
        metrics.set(BuyIdx::DEPLETION_TOTAL, depletion_total);
    }
    if let Some(shield) = queue_result.shield_depth {
        metrics.set(BuyIdx::SHIELD_DEPTH, shield);
    }
    if let Some(slope) = queue_result.queue_shape_slope {
        metrics.set(BuyIdx::QUEUE_SHAPE_SLOPE, slope);
    }

    // Trades between A and B for depletion/trade metrics
    let a_idx = core.a_trade_idx;
    let b_idx = core.b_trade_idx.max(a_idx);
    let mut traded_depletion = 0.0;
    let mut trade_sizes = Vec::new();
    for idx in a_idx..=b_idx {
        if idx >= trades_price.len() {
            break;
        }
        if (trades_price[idx] - price_level).abs() <= eps && trades_flag[idx] == 83 {
            traded_depletion += trades_volume[idx];
            trade_sizes.push(trades_volume[idx]);
        }
    }
    metrics.set(
        BuyIdx::DEPLETION_TRADED,
        if traded_depletion > 0.0 {
            traded_depletion
        } else {
            0.0
        },
    );
    metrics.set(
        BuyIdx::EXEC_AT_X_SELL,
        if traded_depletion > 0.0 {
            traded_depletion
        } else {
            0.0
        },
    );

    if queue_result.depletion_total.is_some() {
        let cancel = (queue_result.depletion_total.unwrap_or(0.0) - traded_depletion).max(0.0);
        metrics.set(BuyIdx::DEPLETION_CANCEL, cancel);
        let denom = traded_depletion + cancel;
        if denom > 0.0 {
            metrics.set(BuyIdx::DEPLETION_SHARE, traded_depletion / denom);
        }
    }

    if !trade_sizes.is_empty() {
        let mut sorted = trade_sizes;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) * 0.5
        } else {
            sorted[mid]
        };
        metrics.set(BuyIdx::TRADE_SIZE_MEDIAN, median);
    }

    let time_to_break_ns = core.b_time - core.a_time;
    metrics.set(BuyIdx::TIME_TO_BREAK, ns_to_ms(time_to_break_ns));
    metrics.set(
        BuyIdx::CONSUMPTION_SPEED,
        if time_to_break_ns > 0 {
            traded_depletion / ns_to_ms(time_to_break_ns).max(1e-6)
        } else {
            NAN_F64
        },
    );

    if let Some(c_time) = core.c_time {
        metrics.set(BuyIdx::TIME_UNDER_X, ns_to_ms(c_time - core.b_time));
    }

    if let Some(d_time) = cycle.d_time {
        if let Some(c_time) = core.c_time {
            metrics.set(BuyIdx::TIME_TO_REFILL, ns_to_ms(d_time - c_time));
        }
    }

    // Depth after D
    if let Some(d_idx) = cycle.d_snapshot_idx {
        let snapshot_d = &snapshots[d_idx];
        if let Some(level) = snapshot_d.find_bid_by_price(price_level, eps) {
            metrics.set(BuyIdx::REFILL_DEPTH_INIT, level.volume);
        }
        if let Some(speed) = compute_refill_speed(
            snapshots,
            cycle.d_time.unwrap_or(core.b_time),
            window_ns,
            price_level,
            Side::Buy,
            eps,
        ) {
            metrics.set(BuyIdx::REFILL_SPEED, speed);
        }
    }

    // Undershoot after B
    let mut min_price = f64::INFINITY;
    let mut idx = core.b_trade_idx + 1;
    let end_idx = core.c_trade_idx.unwrap_or(trades_price.len());
    while idx < trades_price.len() && idx < end_idx {
        min_price = min_price.min(trades_price[idx]);
        idx += 1;
    }
    if min_price.is_finite() {
        let delta = (price_level - min_price) / price_level;
        metrics.set(BuyIdx::UNDERSHOOT_BP, 10_000.0 * delta.max(0.0));
    } else {
        metrics.set(BuyIdx::UNDERSHOOT_BP, 0.0);
    }

    // Overshoot after C within window
    if let Some(c_time) = core.c_time {
        let window_end = c_time + window_ns;
        let start_idx = core
            .c_trade_idx
            .unwrap_or_else(|| lower_bound_trade(trades_time, c_time));
        let end_idx_trade = upper_bound_trade(trades_time, window_end);
        let mut max_price = f64::NEG_INFINITY;
        for idx in start_idx..end_idx_trade {
            if trades_time[idx] >= c_time && trades_time[idx] <= window_end {
                max_price = max_price.max(trades_price[idx]);
            }
        }
        if max_price.is_finite() {
            let delta = (max_price - price_level) / price_level;
            metrics.set(BuyIdx::OVERSHOOT_BP, 10_000.0 * delta.max(0.0));
        }
    }

    // Break force
    let duration_s = ns_to_s(time_to_break_ns);
    if duration_s > 0.0 {
        let delta_bp = 10_000.0 * (price_level - core.b_price) / price_level;
        metrics.set(BuyIdx::BREAK_FORCE, delta_bp / duration_s);
    }

    // Recovery return at 1s after C
    if let Some(c_time) = core.c_time {
        if let Some(mid_c) = mid_price_asof(snapshots, snapshot_mid, c_time) {
            let mid_future_time = c_time + 1_000_000_000;
            if let Some(mid_future) = mid_price_asof(snapshots, snapshot_mid, mid_future_time) {
                let delta = (mid_future - mid_c) / price_level;
                metrics.set(BuyIdx::RECOVERY_RETURN, 10_000.0 * delta);
            }
        }
    }

    // Spread metrics
    let mut spread_at_a: Option<f64> = None;
    if let Some(a_snap_idx) = asof_snapshot_index(snapshots, core.a_time) {
        if let Some(spread) = snapshot_spread.get(a_snap_idx).and_then(|opt| *opt) {
            metrics.set(BuyIdx::SPREAD_AT_TOUCH, spread);
            spread_at_a = Some(spread);
        }
        if let Some(tobi) = snapshot_tobi.get(a_snap_idx).and_then(|opt| *opt) {
            metrics.set(BuyIdx::TOBI_PRE, tobi);
        }
        if window_ns > 0 {
            let window_start = core.a_time.saturating_sub(window_ns);
            let start_idx_snap = lower_bound_snapshot(snapshots, window_start);
            let end_idx_snap = upper_bound_snapshot(snapshots, core.a_time);
            let mut times = Vec::new();
            let mut tob_values = Vec::new();
            let mut revisions = 0usize;
            for idx in start_idx_snap..end_idx_snap {
                let ts = snapshots[idx].timestamp;
                if ts > core.a_time {
                    break;
                }
                if let Some(val) = snapshot_tobi[idx] {
                    times.push(ts);
                    tob_values.push(val);
                }
                revisions += 1;
            }
            if let Some(slope) = linear_regression_slope(&times, &tob_values) {
                metrics.set(BuyIdx::TOBI_TREND, slope);
            }
            let window_seconds = window_ns as f64 / 1_000_000_000.0;
            if window_seconds > 0.0 {
                metrics.set(BuyIdx::QUOTE_REV_RATE, revisions as f64 / window_seconds);
            }
        }
    }

    if let Some(b_snap_idx) = asof_snapshot_index(snapshots, core.b_time) {
        if let (Some(spread_b), Some(spread_a)) = (
            snapshot_spread.get(b_snap_idx).and_then(|opt| *opt),
            spread_at_a,
        ) {
            metrics.set(BuyIdx::SPREAD_CHANGE, spread_b - spread_a);
        }
    }

    // Realized variance post refill
    if let Some(d_time) = cycle.d_time {
        if let Some(rv) = realized_variance(snapshots, snapshot_log_mid, d_time, window_ns) {
            metrics.set(BuyIdx::RV_POST_REFILL, rv);
        }
    }

    // VPIN around A
    let half_window = window_ns / 2;
    let window_start = core.a_time.saturating_sub(half_window);
    let window_end = core.a_time + half_window;
    let start_idx_trade = lower_bound_trade(trades_time, window_start);
    let end_idx_trade = upper_bound_trade(trades_time, window_end);
    let buy_vol = prefix_range(&trade_prefix.vol_buy, start_idx_trade, end_idx_trade);
    let sell_vol = prefix_range(&trade_prefix.vol_sell, start_idx_trade, end_idx_trade);
    let total_vol = buy_vol + sell_vol;
    if total_vol > 0.0 {
        metrics.set(
            BuyIdx::VPIN_AROUND_A,
            (buy_vol - sell_vol).abs() / total_vol,
        );
    }

    // Adverse post fill
    if let Some(mid_a) = mid_price_asof(snapshots, snapshot_mid, core.a_time) {
        if let Some(mid_future) = mid_price_asof(snapshots, snapshot_mid, core.a_time + window_ns) {
            let delta = (mid_future - mid_a) / price_level;
            metrics.set(BuyIdx::ADVERSE_POST_FILL, 10_000.0 * delta);
        }
    }

    if let Some(d_time) = cycle.d_time {
        if let Some(mid_d) = mid_price_asof(snapshots, snapshot_mid, d_time) {
            if let Some(mid_future) = mid_price_asof(snapshots, snapshot_mid, d_time + window_ns) {
                let delta = (mid_future - mid_d) / price_level;
                metrics.set(BuyIdx::ADVERSE_POST_REFILL, 10_000.0 * delta);
            }
        }
    }

    if let Some(d_time) = cycle.d_time {
        let start_idx = lower_bound_trade(trades_time, d_time);
        let end_idx = upper_bound_trade(trades_time, d_time + window_ns);
        let buy_vol = prefix_range(&trade_prefix.vol_buy, start_idx, end_idx);
        let sell_vol = prefix_range(&trade_prefix.vol_sell, start_idx, end_idx);
        let total = buy_vol + sell_vol;
        if total > 0.0 {
            metrics.set(BuyIdx::OFI_POST_REFILL, (buy_vol - sell_vol) / total);
        }
    }

    // Survival metrics after D
    if let Some(d_time) = cycle.d_time {
        let survival_window_end = d_time + window_ns;
        let mut success = 1.0;
        let mut survival_ms = window_ns as f64 / 1_000_000.0;
        let start_idx = upper_bound_trade(trades_time, d_time);
        for idx in start_idx..trades_time.len() {
            let ts = trades_time[idx];
            if ts > survival_window_end {
                break;
            }
            if trades_price[idx] <= price_level + eps {
                survival_ms = ns_to_ms(ts - d_time);
                success = 0.0;
                break;
            }
        }
        metrics.set(BuyIdx::SUPPORT_SURVIVAL, survival_ms);
        metrics.set(BuyIdx::BOUNCE_SUCCESS, success);
    }

    metrics
}

#[allow(clippy::too_many_arguments)]
fn compute_sell_cycle_metrics(
    cycle: &PriceCycleFull,
    trades_time: &[i64],
    trades_price: &[f64],
    trades_volume: &[f64],
    trades_flag: &[i32],
    trade_prefix: &TradePrefixes,
    snapshots: &[Snapshot],
    snapshot_mid: &[Option<f64>],
    snapshot_log_mid: &[Option<f64>],
    snapshot_tobi: &[Option<f64>],
    snapshot_spread: &[Option<f64>],
    window_ns: i64,
    eps: f64,
) -> CycleMetrics {
    let mut metrics = CycleMetrics::new(SellIdx::LEN);
    let core = &cycle.core;
    let price_level = core.price;

    // Queue metrics between A' and B'
    let queue_result = compute_queue_metrics(
        snapshots,
        core.a_time,
        core.b_time,
        price_level,
        Side::Sell,
        eps,
    );

    if let Some(level) = queue_result.level_at_touch {
        metrics.set(SellIdx::N_AT_TOUCH, level as f64);
    }
    if let Some(visible) = queue_result.visible_levels {
        metrics.set(SellIdx::VISIBLE_LEVELS, visible);
    }
    if let Some(depth_init) = queue_result.depth_init {
        metrics.set(SellIdx::DEPTH_INIT, depth_init);
    }
    if let Some(depth_max) = queue_result.depth_max {
        metrics.set(SellIdx::DEPTH_MAX, depth_max);
    }
    if let Some(depth_min) = queue_result.depth_min {
        metrics.set(SellIdx::DEPTH_MIN, depth_min);
    }
    if let Some(depth_twap) = queue_result.depth_twap {
        metrics.set(SellIdx::DEPTH_TWAP, depth_twap);
    }
    if let Some(depletion_total) = queue_result.depletion_total {
        metrics.set(SellIdx::DEPLETION_TOTAL, depletion_total);
    }
    if let Some(shield) = queue_result.shield_depth {
        metrics.set(SellIdx::SHIELD_DEPTH, shield);
    }
    if let Some(slope) = queue_result.queue_shape_slope {
        metrics.set(SellIdx::QUEUE_SHAPE_SLOPE, slope);
    }

    let a_idx = core.a_trade_idx;
    let b_idx = core.b_trade_idx.max(a_idx);
    let mut traded_depletion = 0.0;
    let mut trade_sizes = Vec::new();
    for idx in a_idx..=b_idx {
        if idx >= trades_price.len() {
            break;
        }
        if (trades_price[idx] - price_level).abs() <= eps && trades_flag[idx] == 66 {
            traded_depletion += trades_volume[idx];
            trade_sizes.push(trades_volume[idx]);
        }
    }
    metrics.set(
        SellIdx::DEPLETION_TRADED,
        if traded_depletion > 0.0 {
            traded_depletion
        } else {
            0.0
        },
    );
    metrics.set(
        SellIdx::EXEC_AT_Y_BUY,
        if traded_depletion > 0.0 {
            traded_depletion
        } else {
            0.0
        },
    );

    if queue_result.depletion_total.is_some() {
        let cancel = (queue_result.depletion_total.unwrap_or(0.0) - traded_depletion).max(0.0);
        metrics.set(SellIdx::DEPLETION_CANCEL, cancel);
        let denom = traded_depletion + cancel;
        if denom > 0.0 {
            metrics.set(SellIdx::DEPLETION_SHARE, traded_depletion / denom);
        }
    }

    if !trade_sizes.is_empty() {
        let mut sorted = trade_sizes;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) * 0.5
        } else {
            sorted[mid]
        };
        metrics.set(SellIdx::TRADE_SIZE_MEDIAN, median);
    }

    let time_to_break_ns = core.b_time - core.a_time;
    metrics.set(SellIdx::TIME_TO_BREAK, ns_to_ms(time_to_break_ns));
    metrics.set(
        SellIdx::CONSUMPTION_SPEED,
        if time_to_break_ns > 0 {
            traded_depletion / ns_to_ms(time_to_break_ns).max(1e-6)
        } else {
            NAN_F64
        },
    );

    if let Some(c_time) = core.c_time {
        metrics.set(SellIdx::TIME_ABOVE_Y, ns_to_ms(c_time - core.b_time));
    }

    if let Some(d_time) = cycle.d_time {
        if let Some(c_time) = core.c_time {
            metrics.set(SellIdx::TIME_TO_REFILL, ns_to_ms(d_time - c_time));
        }
    }

    if let Some(d_idx) = cycle.d_snapshot_idx {
        let snapshot_d = &snapshots[d_idx];
        if let Some(level) = snapshot_d.find_ask_by_price(price_level, eps) {
            metrics.set(SellIdx::REFILL_DEPTH_INIT, level.volume);
        }
        if let Some(speed) = compute_refill_speed(
            snapshots,
            cycle.d_time.unwrap_or(core.b_time),
            window_ns,
            price_level,
            Side::Sell,
            eps,
        ) {
            metrics.set(SellIdx::REFILL_SPEED, speed);
        }
    }

    // Overshoot after B (price up)
    let mut max_price = f64::NEG_INFINITY;
    let mut idx_iter = core.b_trade_idx + 1;
    let end_idx = core.c_trade_idx.unwrap_or(trades_price.len());
    while idx_iter < trades_price.len() && idx_iter < end_idx {
        max_price = max_price.max(trades_price[idx_iter]);
        idx_iter += 1;
    }
    if max_price.is_finite() {
        let delta = (max_price - price_level) / price_level;
        metrics.set(SellIdx::OVERSHOOT_BP, 10_000.0 * delta.max(0.0));
    } else {
        metrics.set(SellIdx::OVERSHOOT_BP, 0.0);
    }

    // Undershoot after C within window (price falls below)
    if let Some(c_time) = core.c_time {
        let window_end = c_time + window_ns;
        let start_idx = core
            .c_trade_idx
            .unwrap_or_else(|| lower_bound_trade(trades_time, c_time));
        let end_idx_trade = upper_bound_trade(trades_time, window_end);
        let mut min_price = f64::INFINITY;
        for idx in start_idx..end_idx_trade {
            if trades_time[idx] >= c_time && trades_time[idx] <= window_end {
                min_price = min_price.min(trades_price[idx]);
            }
        }
        if min_price.is_finite() {
            let delta = (price_level - min_price) / price_level;
            metrics.set(SellIdx::UNDERSHOOT_BP, 10_000.0 * delta.max(0.0));
        }
    }

    let duration_s = ns_to_s(time_to_break_ns);
    if duration_s > 0.0 {
        let delta_bp = 10_000.0 * (core.b_price - price_level) / price_level;
        metrics.set(SellIdx::BREAK_FORCE, delta_bp / duration_s);
    }

    if let Some(c_time) = core.c_time {
        if let Some(mid_c) = mid_price_asof(snapshots, snapshot_mid, c_time) {
            let mid_future_time = c_time + 1_000_000_000;
            if let Some(mid_future) = mid_price_asof(snapshots, snapshot_mid, mid_future_time) {
                let delta = (mid_future - mid_c) / price_level;
                metrics.set(SellIdx::RECOVERY_RETURN, 10_000.0 * delta);
            }
        }
    }

    let mut spread_at_a: Option<f64> = None;
    if let Some(a_snap_idx) = asof_snapshot_index(snapshots, core.a_time) {
        if let Some(spread) = snapshot_spread.get(a_snap_idx).and_then(|opt| *opt) {
            metrics.set(SellIdx::SPREAD_AT_TOUCH, spread);
            spread_at_a = Some(spread);
        }
        if let Some(tobi) = snapshot_tobi.get(a_snap_idx).and_then(|opt| *opt) {
            metrics.set(SellIdx::TOBI_PRE, tobi);
        }
        if window_ns > 0 {
            let window_start = core.a_time.saturating_sub(window_ns);
            let start_idx_snap = lower_bound_snapshot(snapshots, window_start);
            let end_idx_snap = upper_bound_snapshot(snapshots, core.a_time);
            let mut times = Vec::new();
            let mut tob_values = Vec::new();
            let mut revisions = 0usize;
            for idx in start_idx_snap..end_idx_snap {
                let ts = snapshots[idx].timestamp;
                if ts > core.a_time {
                    break;
                }
                if let Some(val) = snapshot_tobi[idx] {
                    times.push(ts);
                    tob_values.push(val);
                }
                revisions += 1;
            }
            if let Some(slope) = linear_regression_slope(&times, &tob_values) {
                metrics.set(SellIdx::TOBI_TREND, slope);
            }
            let window_seconds = window_ns as f64 / 1_000_000_000.0;
            if window_seconds > 0.0 {
                metrics.set(SellIdx::QUOTE_REV_RATE, revisions as f64 / window_seconds);
            }
        }
    }

    if let Some(b_snap_idx) = asof_snapshot_index(snapshots, core.b_time) {
        if let (Some(spread_b), Some(spread_a_val)) = (
            snapshot_spread.get(b_snap_idx).and_then(|opt| *opt),
            spread_at_a,
        ) {
            metrics.set(SellIdx::SPREAD_CHANGE, spread_b - spread_a_val);
        }
    }

    if let Some(d_time) = cycle.d_time {
        if let Some(rv) = realized_variance(snapshots, snapshot_log_mid, d_time, window_ns) {
            metrics.set(SellIdx::RV_POST_REFILL, rv);
        }
    }

    let half_window = window_ns / 2;
    let window_start = core.a_time.saturating_sub(half_window);
    let window_end = core.a_time + half_window;
    let start_idx_trade = lower_bound_trade(trades_time, window_start);
    let end_idx_trade = upper_bound_trade(trades_time, window_end);
    let buy_vol = prefix_range(&trade_prefix.vol_buy, start_idx_trade, end_idx_trade);
    let sell_vol = prefix_range(&trade_prefix.vol_sell, start_idx_trade, end_idx_trade);
    let total_vol = buy_vol + sell_vol;
    if total_vol > 0.0 {
        metrics.set(
            SellIdx::VPIN_AROUND_A,
            (buy_vol - sell_vol).abs() / total_vol,
        );
    }

    if let Some(mid_a) = mid_price_asof(snapshots, snapshot_mid, core.a_time) {
        if let Some(mid_future) = mid_price_asof(snapshots, snapshot_mid, core.a_time + window_ns) {
            let delta = (mid_future - mid_a) / price_level;
            metrics.set(SellIdx::ADVERSE_POST_FILL, 10_000.0 * delta);
        }
    }

    if let Some(d_time) = cycle.d_time {
        if let Some(mid_d) = mid_price_asof(snapshots, snapshot_mid, d_time) {
            if let Some(mid_future) = mid_price_asof(snapshots, snapshot_mid, d_time + window_ns) {
                let delta = (mid_future - mid_d) / price_level;
                metrics.set(SellIdx::ADVERSE_POST_REFILL, 10_000.0 * delta);
            }
        }
    }

    if let Some(d_time) = cycle.d_time {
        let start_idx = lower_bound_trade(trades_time, d_time);
        let end_idx = upper_bound_trade(trades_time, d_time + window_ns);
        let buy_vol = prefix_range(&trade_prefix.vol_buy, start_idx, end_idx);
        let sell_vol = prefix_range(&trade_prefix.vol_sell, start_idx, end_idx);
        let total = buy_vol + sell_vol;
        if total > 0.0 {
            metrics.set(SellIdx::OFI_POST_REFILL, (buy_vol - sell_vol) / total);
        }
    }

    if let Some(d_time) = cycle.d_time {
        let survival_window_end = d_time + window_ns;
        let mut success = 1.0;
        let mut survival_ms = window_ns as f64 / 1_000_000.0;
        let start_idx = upper_bound_trade(trades_time, d_time);
        for idx in start_idx..trades_time.len() {
            let ts = trades_time[idx];
            if ts > survival_window_end {
                break;
            }
            if trades_price[idx] >= price_level - eps {
                survival_ms = ns_to_ms(ts - d_time);
                success = 0.0;
                break;
            }
        }
        metrics.set(SellIdx::RESISTANCE_SURVIVAL, survival_ms);
        metrics.set(SellIdx::BOUNCE_SUCCESS, success);
    }

    metrics
}

fn is_round_number(price: f64, tick: f64, eps: f64) -> bool {
    let rounded = price.round();
    if (price - rounded).abs() <= eps {
        return true;
    }
    // Consider half-integer as round (e.g., .5)
    let half = (price * 2.0).round() * 0.5;
    if (price - half).abs() <= eps {
        return true;
    }
    // If tick is >= 0.01, treat multiples of 0.1 as round as well
    let cent_base = (price * 10.0).round() / 10.0;
    if (price - cent_base).abs() <= eps {
        return true;
    }
    if tick > 0.0 {
        let extended = (price / (tick * 10.0)).round() * (tick * 10.0);
        if (price - extended).abs() <= eps {
            return true;
        }
    }
    false
}

fn build_price_grid(
    trades: &[f64],
    provided_grid: Option<&[f64]>,
    use_trades: bool,
    tick: f64,
) -> Vec<f64> {
    let mut prices = Vec::new();
    if let Some(grid) = provided_grid {
        prices.extend(grid.iter().copied());
    } else if use_trades {
        prices.extend(trades.iter().copied());
    }
    if tick > 0.0 {
        for value in &mut prices {
            *value = (*value / tick).round() * tick;
        }
    }
    let dedup_eps = effective_eps(tick);
    prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    prices.dedup_by(|a, b| (*a - *b).abs() <= dedup_eps);
    prices
}

const EPSILON_BASE: f64 = 1e-9;

fn effective_eps(tick: f64) -> f64 {
    EPSILON_BASE.max(tick / 10.0)
}

#[inline]
fn ns_to_ms(ns: i64) -> f64 {
    ns as f64 / 1_000_000.0
}

#[inline]
fn ns_to_s(ns: i64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_price_cycle_features(
    exchtime_trade: &PyArray1<i64>,
    price_trade: &PyArray1<f64>,
    volume_trade: &PyArray1<f64>,
    flag_trade: &PyArray1<i32>,
    ask_exchtime: &PyArray1<i64>,
    bid_exchtime: &PyArray1<i64>,
    bid_price: &PyArray1<f64>,
    bid_volume: &PyArray1<f64>,
    bid_number: &PyArray1<i32>,
    ask_price: &PyArray1<f64>,
    ask_volume: &PyArray1<f64>,
    ask_number: &PyArray1<i32>,
    tick: f64,
    drop_threshold: f64,
    rise_threshold: f64,
    window_ms: i64,
    above_min_ms: i64,
    use_trade_prices_as_grid: bool,
    price_grid_opt: Option<&PyArray1<f64>>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let trades_time = unsafe { exchtime_trade.as_slice()? };
    let trades_price = unsafe { price_trade.as_slice()? };
    let trades_volume = unsafe { volume_trade.as_slice()? };
    let trades_flag = unsafe { flag_trade.as_slice()? };

    let ask_time = unsafe { ask_exchtime.as_slice()? };
    let bid_time = unsafe { bid_exchtime.as_slice()? };
    let ask_price_slice = unsafe { ask_price.as_slice()? };
    let ask_volume_slice = unsafe { ask_volume.as_slice()? };
    let ask_number_slice = unsafe { ask_number.as_slice()? };
    let bid_price_slice = unsafe { bid_price.as_slice()? };
    let bid_volume_slice = unsafe { bid_volume.as_slice()? };
    let bid_number_slice = unsafe { bid_number.as_slice()? };

    if trades_time.len() != trades_price.len()
        || trades_time.len() != trades_volume.len()
        || trades_time.len() != trades_flag.len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "trade arrays length mismatch",
        ));
    }

    if ask_time.len() != ask_price_slice.len()
        || ask_time.len() != ask_volume_slice.len()
        || ask_time.len() != ask_number_slice.len()
        || bid_time.len() != bid_price_slice.len()
        || bid_time.len() != bid_volume_slice.len()
        || bid_time.len() != bid_number_slice.len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "snapshot arrays length mismatch",
        ));
    }

    let provided_grid = if let Some(arr) = price_grid_opt {
        let slice = unsafe { arr.as_slice()? };
        Some(slice)
    } else {
        None
    };

    let eps = effective_eps(tick);
    let price_grid = build_price_grid(trades_price, provided_grid, use_trade_prices_as_grid, tick);

    let snapshots = build_snapshots(
        ask_time,
        bid_time,
        ask_price_slice,
        ask_volume_slice,
        ask_number_slice,
        bid_price_slice,
        bid_volume_slice,
        bid_number_slice,
    );

    let trade_prefix = build_trade_prefixes(trades_price, trades_volume, trades_flag);

    let window_ns = window_ms * 1_000_000;
    let above_min_ns = above_min_ms * 1_000_000;

    let mut snapshot_mid = Vec::with_capacity(snapshots.len());
    let mut snapshot_log_mid = Vec::with_capacity(snapshots.len());
    let mut snapshot_tobi = Vec::with_capacity(snapshots.len());
    let mut snapshot_spread = Vec::with_capacity(snapshots.len());
    for snap in &snapshots {
        let mid = snap.mid_price();
        snapshot_mid.push(mid);
        snapshot_log_mid.push(mid.and_then(|m| if m > 0.0 { Some(m.ln()) } else { None }));
        snapshot_tobi.push(snap.tob_imbalance());
        snapshot_spread.push(snap.spread());
    }

    let mut aggregates: Vec<PriceAggregate> = price_grid
        .iter()
        .copied()
        .map(PriceAggregate::new)
        .collect();

    for aggregate in aggregates.iter_mut() {
        let price_level = aggregate.price;

        // Buy side cycles
        let buy_cores = detect_buy_cycles_for_price(
            price_level,
            trades_time,
            trades_price,
            drop_threshold,
            above_min_ns,
            eps,
        );
        let mut buy_fulls = Vec::with_capacity(buy_cores.len());
        let mut buy_metrics = Vec::with_capacity(buy_cores.len());
        for core in buy_cores {
            let mut full = PriceCycleFull {
                core,
                d_snapshot_idx: None,
                d_time: None,
            };
            if let Some(c_time) = full.core.c_time {
                if let Some((d_idx, d_time)) =
                    locate_refill_snapshot(&snapshots, c_time, price_level, Side::Buy, eps)
                {
                    full.d_snapshot_idx = Some(d_idx);
                    full.d_time = Some(d_time);
                }
            }
            let metrics = compute_buy_cycle_metrics(
                &full,
                trades_time,
                trades_price,
                trades_volume,
                trades_flag,
                &trade_prefix,
                &snapshots,
                &snapshot_mid,
                &snapshot_log_mid,
                &snapshot_tobi,
                &snapshot_spread,
                window_ns,
                eps,
            );
            buy_fulls.push(full);
            buy_metrics.push(metrics);
        }
        for idx in 0..buy_metrics.len() {
            if idx + 1 < buy_metrics.len() {
                let inter_ns = buy_fulls[idx + 1].core.a_time - buy_fulls[idx].core.b_time;
                if inter_ns > 0 {
                    buy_metrics[idx].set(BuyIdx::NEXT_INTERARRIVAL, ns_to_ms(inter_ns));
                }
            }
        }
        aggregate.cycles_buy_count = buy_metrics.len();
        aggregate.metrics_buy = buy_metrics;

        // Sell side cycles
        let sell_cores = detect_sell_cycles_for_price(
            price_level,
            trades_time,
            trades_price,
            rise_threshold,
            above_min_ns,
            eps,
        );
        let mut sell_fulls = Vec::with_capacity(sell_cores.len());
        let mut sell_metrics = Vec::with_capacity(sell_cores.len());
        for core in sell_cores {
            let mut full = PriceCycleFull {
                core,
                d_snapshot_idx: None,
                d_time: None,
            };
            if let Some(c_time) = full.core.c_time {
                if let Some((d_idx, d_time)) =
                    locate_refill_snapshot(&snapshots, c_time, price_level, Side::Sell, eps)
                {
                    full.d_snapshot_idx = Some(d_idx);
                    full.d_time = Some(d_time);
                }
            }
            let metrics = compute_sell_cycle_metrics(
                &full,
                trades_time,
                trades_price,
                trades_volume,
                trades_flag,
                &trade_prefix,
                &snapshots,
                &snapshot_mid,
                &snapshot_log_mid,
                &snapshot_tobi,
                &snapshot_spread,
                window_ns,
                eps,
            );
            sell_fulls.push(full);
            sell_metrics.push(metrics);
        }
        for idx in 0..sell_metrics.len() {
            if idx + 1 < sell_metrics.len() {
                let inter_ns = sell_fulls[idx + 1].core.a_time - sell_fulls[idx].core.b_time;
                if inter_ns > 0 {
                    sell_metrics[idx].set(SellIdx::NEXT_INTERARRIVAL, ns_to_ms(inter_ns));
                }
            }
        }
        aggregate.cycles_sell_count = sell_metrics.len();
        aggregate.metrics_sell = sell_metrics;
    }

    let buy_feature_names = vec![
        "n_at_touch_buy",
        "visible_levels_bid_at_touch_buy",
        "depth_x_init_buy",
        "depth_x_max_buy",
        "depth_x_min_buy",
        "depth_x_twap_buy",
        "depletion_total_x_buy",
        "traded_depletion_x_buy",
        "cancel_depletion_x_buy",
        "depletion_trade_share_buy",
        "refill_depth_init_buy",
        "refill_speed_buy",
        "shield_depth_from_best_to_x_buy",
        "exec_at_x_sell_buy",
        "trade_size_median_at_x_sell_buy",
        "consumption_speed_buy",
        "time_to_break_ms_buy",
        "time_under_x_ms_buy",
        "time_to_refill_ms_buy",
        "undershoot_bp_buy",
        "overshoot_bp_buy",
        "break_force_bp_per_s_buy",
        "recovery_return_bp_1s_buy",
        "tobi_pre_touch_buy",
        "tobi_trend_pre_buy",
        "spread_at_touch_buy",
        "spread_change_A_to_B_buy",
        "quote_rev_rate_pre_buy",
        "rv_post_refill_buy",
        "vpin_like_A_window_buy",
        "adverse_post_fill_bp_buy",
        "adverse_post_refill_bp_buy",
        "ofi_post_refill_buy",
        "support_survival_ms_buy",
        "bounce_success_flag_buy",
        "next_interarrival_ms_buy",
        "queue_shape_slope_near_x_buy",
    ];

    let sell_feature_names = vec![
        "n_at_touch_sell",
        "visible_levels_ask_at_touch_sell",
        "depth_y_init_sell",
        "depth_y_max_sell",
        "depth_y_min_sell",
        "depth_y_twap_sell",
        "depletion_total_y_sell",
        "traded_depletion_y_sell",
        "cancel_depletion_y_sell",
        "depletion_trade_share_sell",
        "refill_depth_init_sell",
        "refill_speed_sell",
        "shield_depth_from_best_to_y_sell",
        "exec_at_y_buy_sell",
        "trade_size_median_at_y_buy_sell",
        "consumption_speed_sell",
        "time_to_break_ms_sell",
        "time_above_y_ms_sell",
        "time_to_refill_ms_sell",
        "overshoot_bp_sell",
        "undershoot_bp_sell",
        "break_force_bp_per_s_sell",
        "recovery_return_bp_1s_sell",
        "tobi_pre_touch_sell",
        "tobi_trend_pre_sell",
        "spread_at_touch_sell",
        "spread_change_A_to_B_sell",
        "quote_rev_rate_pre_sell",
        "rv_post_refill_sell",
        "vpin_like_A_window_sell",
        "adverse_post_fill_bp_sell",
        "adverse_post_refill_bp_sell",
        "ofi_post_refill_sell",
        "resistance_survival_ms_sell",
        "bounce_success_flag_sell",
        "next_interarrival_ms_sell",
        "queue_shape_slope_near_y_sell",
    ];

    let total_features = buy_feature_names.len() + sell_feature_names.len() + 3;
    let mut feature_matrix = Array2::<f64>::from_elem((aggregates.len(), total_features), NAN_F64);
    let mut cycles_buy_vec = Vec::with_capacity(aggregates.len());
    let mut cycles_sell_vec = Vec::with_capacity(aggregates.len());

    for (row_idx, aggregate) in aggregates.iter().enumerate() {
        cycles_buy_vec.push(aggregate.cycles_buy_count as f64);
        cycles_sell_vec.push(aggregate.cycles_sell_count as f64);
        let round_flag = if is_round_number(aggregate.price, tick, eps) {
            1.0
        } else {
            0.0
        };

        // Aggregation placeholders: will be populated once metrics computed.
        let mut offset = 0usize;
        if !aggregate.metrics_buy.is_empty() {
            for col in 0..buy_feature_names.len() {
                let mut sum = 0.0;
                let mut valid = 0usize;
                for metrics in &aggregate.metrics_buy {
                    let value = metrics.get(col);
                    if !value.is_nan() {
                        sum += value;
                        valid += 1;
                    }
                }
                let feature_value = if valid > 0 {
                    sum / valid as f64
                } else {
                    NAN_F64
                };
                feature_matrix[[row_idx, offset + col]] = feature_value;
            }
        }
        offset += buy_feature_names.len();

        if !aggregate.metrics_sell.is_empty() {
            for col in 0..sell_feature_names.len() {
                let mut sum = 0.0;
                let mut valid = 0usize;
                for metrics in &aggregate.metrics_sell {
                    let value = metrics.get(col);
                    if !value.is_nan() {
                        sum += value;
                        valid += 1;
                    }
                }
                let feature_value = if valid > 0 {
                    sum / valid as f64
                } else {
                    NAN_F64
                };
                feature_matrix[[row_idx, offset + col]] = feature_value;
            }
        }
        offset += sell_feature_names.len();

        feature_matrix[[row_idx, offset]] = aggregate.cycles_buy_count as f64;
        feature_matrix[[row_idx, offset + 1]] = aggregate.cycles_sell_count as f64;
        feature_matrix[[row_idx, offset + 2]] = round_flag;
    }

    let prices_np = price_grid.into_pyarray(py);
    let feature_names: Vec<String> = buy_feature_names
        .iter()
        .chain(sell_feature_names.iter())
        .chain(["cycles_count_buy", "cycles_count_sell", "round_number_flag"].iter())
        .map(|name| (*name).to_string())
        .collect();

    let features_np = PyArray2::from_array(py, &feature_matrix);
    let cycles_buy_np = cycles_buy_vec.into_pyarray(py);
    let cycles_sell_np = cycles_sell_vec.into_pyarray(py);

    let result = PyDict::new(py);
    result.set_item("prices", prices_np)?;
    result.set_item("feature_names", feature_names)?;
    result.set_item("features", features_np)?;
    result.set_item("cycles_count_buy", cycles_buy_np)?;
    result.set_item("cycles_count_sell", cycles_sell_np)?;

    Ok(result.into())
}
