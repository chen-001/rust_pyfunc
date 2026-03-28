use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

const FIELD_OPEN: usize = 0;
const FIELD_CLOSE: usize = 1;
const FIELD_HIGH: usize = 2;
const FIELD_LOW: usize = 3;
const FIELD_AMOUNT: usize = 4;
const FIELD_VOLUME: usize = 5;
const FIELD_ACT_BUY_AMT: usize = 6;
const FIELD_ACT_SELL_AMT: usize = 7;
const FIELD_ACT_BUY_VOL: usize = 8;
const FIELD_ACT_SELL_VOL: usize = 9;
const FIELD_ACT_BUY_CNT: usize = 10;
const FIELD_ACT_SELL_CNT: usize = 11;
const FIELD_UP_TICK: usize = 12;
const FIELD_DOWN_TICK: usize = 13;
const FIELD_BID_SIZE1: usize = 14;
const FIELD_ASK_SIZE1: usize = 15;
const FIELD_BID_SIZE6: usize = 16;
const FIELD_ASK_SIZE6: usize = 17;
const FIELD_BID_SIZE10: usize = 18;
const FIELD_ASK_SIZE10: usize = 19;
const FIELD_BID_VWAP_MEAN: usize = 20;
const FIELD_ASK_VWAP_MEAN: usize = 21;
const FIELD_ASK_VWAP3: usize = 22;
const FIELD_ASK_VWAP5: usize = 23;
const FIELD_ASK_VWAP10: usize = 24;
const FIELD_BID_VWAP3: usize = 25;
const FIELD_BID_VWAP5: usize = 26;
const FIELD_BID_VWAP10: usize = 27;
const FIELD_BID_VOL1: usize = 28;
const FIELD_ASK_VOL1: usize = 29;
const FIELD_BID_VOL2: usize = 30;
const FIELD_ASK_VOL2: usize = 31;
const FIELD_BID_VOL3: usize = 32;
const FIELD_ASK_VOL3: usize = 33;

const ALL_DAY_BINS: usize = 8;
const SHORT_BINS: usize = 6;
const EPS: f64 = 1e-12;

#[derive(Clone)]
struct SegmentDayFeatures {
    ret: Vec<f64>,
    intraday_vol: Vec<f64>,
    raw_amt: Vec<f64>,
    log_amt: Vec<f64>,
    buy_ratio: Vec<f64>,
    net_flow: Vec<f64>,
    log_avg_trade_amt: Vec<f64>,
    range_ratio: Vec<f64>,
    close_loc: Vec<f64>,
    tick_balance: Vec<f64>,
    depth1: Vec<f64>,
    depth6: Vec<f64>,
    bid10: Vec<f64>,
    vwap_gap: Vec<f64>,
    vwap_ladder_gap: Vec<f64>,
    queue1: Vec<f64>,
    queue123: Vec<f64>,
    flow_eff: Vec<f64>,
    big_order_ratio: Vec<f64>,
    book_pressure: Vec<f64>,
    summary: Vec<f64>,
    ret_path: Vec<f64>,
    cumret_path: Vec<f64>,
    rv_path: Vec<f64>,
    amt_profile: Vec<f64>,
    net_flow_path: Vec<f64>,
    buy_ratio_path: Vec<f64>,
    trade_size_path: Vec<f64>,
    tick_balance_path: Vec<f64>,
    depth1_path: Vec<f64>,
    depth6_path: Vec<f64>,
    queue_path: Vec<f64>,
    quote_gap_path: Vec<f64>,
    path_bins: usize,
    n_stocks: usize,
    summary_dim: usize,
}

#[derive(Clone)]
struct ClusterDayResult {
    labels: Vec<usize>,
    centers: Vec<f64>,
    k: usize,
    mean_ret: Vec<f64>,
    std_ret: Vec<f64>,
    mean_buy_ratio: Vec<f64>,
    mean_net_flow: Vec<f64>,
    mean_log_amt: Vec<f64>,
    mean_intraday_vol: Vec<f64>,
    mean_range: Vec<f64>,
    mean_close_loc: Vec<f64>,
    mean_flow_eff: Vec<f64>,
    mean_book_pressure: Vec<f64>,
    total_amt: Vec<f64>,
    breadth: Vec<f64>,
    leader_gap: Vec<f64>,
    skew_ret: Vec<f64>,
    theme_heat: Vec<f64>,
    ret_path_mean: Vec<f64>,
    flow_path_mean: Vec<f64>,
    amt_path_mean: Vec<f64>,
    depth_path_mean: Vec<f64>,
    return_rank_pct: Vec<f64>,
    flow_rank_pct: Vec<f64>,
    buy_rank_pct: Vec<f64>,
    dists_to_centers: Vec<f64>,
}

fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

fn mean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += v;
            cnt += 1;
        }
    }
    if cnt == 0 {
        f64::NAN
    } else {
        sum / cnt as f64
    }
}

fn std(values: &[f64]) -> f64 {
    let m = mean(values);
    if !m.is_finite() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += (v - m).powi(2);
            cnt += 1;
        }
    }
    if cnt <= 1 {
        0.0
    } else {
        (sum / cnt as f64).sqrt()
    }
}

fn skew(values: &[f64]) -> f64 {
    let m = mean(values);
    if !m.is_finite() {
        return f64::NAN;
    }
    let s = std(values);
    if !s.is_finite() || s <= EPS {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += ((v - m) / s).powi(3);
            cnt += 1;
        }
    }
    if cnt == 0 {
        f64::NAN
    } else {
        sum / cnt as f64
    }
}

fn slope(values: &[f64]) -> f64 {
    let mut xs = Vec::with_capacity(values.len());
    let mut ys = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() {
            xs.push(i as f64);
            ys.push(v);
        }
    }
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let sx2: f64 = xs.iter().map(|x| x * x).sum();
    let denom = n as f64 * sx2 - sx * sx;
    if denom.abs() <= EPS {
        f64::NAN
    } else {
        (n as f64 * sxy - sx * sy) / denom
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) {
        let x = a[i];
        let y = b[i];
        if x.is_finite() && y.is_finite() {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
    }
    if na <= EPS || nb <= EPS {
        f64::NAN
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for i in 0..a.len().min(b.len()) {
        let x = a[i];
        let y = b[i];
        if x.is_finite() && y.is_finite() {
            sum += (x - y).powi(2);
            cnt += 1;
        }
    }
    if cnt == 0 {
        f64::NAN
    } else {
        sum.sqrt()
    }
}

fn hhi_from_abs(values: &[f64]) -> f64 {
    let total: f64 = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .sum();
    if total <= EPS {
        return f64::NAN;
    }
    values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| {
            let p = v.abs() / total;
            p * p
        })
        .sum()
}

fn dct_first_k(values: &[f64], k: usize) -> Vec<f64> {
    let n = values.len() as f64;
    (0..k)
        .map(|kk| {
            let mut sum = 0.0;
            for (i, &v) in values.iter().enumerate() {
                if v.is_finite() {
                    let angle = std::f64::consts::PI * (i as f64 + 0.5) * kk as f64 / n.max(1.0);
                    sum += v * angle.cos();
                }
            }
            sum
        })
        .collect()
}

fn rank_percentile(values: &[f64], labels: &[usize], k: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    let mut groups: Vec<Vec<(usize, f64)>> = (0..k).map(|_| Vec::new()).collect();
    for i in 0..n {
        let label = labels[i];
        let v = values[i];
        if label < k && v.is_finite() {
            groups[label].push((i, v));
        }
    }
    for group in groups.iter_mut() {
        group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let m = group.len();
        if m == 1 {
            result[group[0].0] = 0.5;
            continue;
        }
        for (rank, (idx, _)) in group.iter().enumerate() {
            result[*idx] = rank as f64 / (m - 1) as f64;
        }
    }
    result
}

fn zscore_standardize_flat(data: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n * d];
    for j in 0..d {
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let v = data[i * d + j];
            if v.is_finite() {
                vals.push(v);
            }
        }
        if vals.is_empty() {
            continue;
        }
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let s = (vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / vals.len() as f64)
            .sqrt()
            .max(EPS);
        for i in 0..n {
            let v = data[i * d + j];
            if v.is_finite() {
                out[i * d + j] = (v - m) / s;
            }
        }
    }
    out
}

fn kmeans_pp_init_flat(data: &[f64], n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut centers = vec![0.0; k * d];
    if n == 0 || k == 0 {
        return centers;
    }
    let mut rng_state = seed;
    let mut next_rand = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };
    let first = (next_rand() * n as f64) as usize % n;
    centers[..d].copy_from_slice(&data[first * d..(first + 1) * d]);
    let mut dists = vec![f64::MAX; n];
    for c in 1..k {
        let prev_center = &centers[(c - 1) * d..c * d];
        for i in 0..n {
            let mut dist = 0.0;
            for j in 0..d {
                let diff = data[i * d + j] - prev_center[j];
                dist += diff * diff;
            }
            dists[i] = dists[i].min(dist);
        }
        let total: f64 = dists.iter().sum();
        if total <= EPS {
            centers[c * d..(c + 1) * d].copy_from_slice(&data[..d]);
            continue;
        }
        let target = next_rand() * total;
        let mut acc = 0.0;
        let mut chosen = 0usize;
        for (i, &dist) in dists.iter().enumerate() {
            acc += dist;
            if acc >= target {
                chosen = i;
                break;
            }
        }
        centers[c * d..(c + 1) * d].copy_from_slice(&data[chosen * d..(chosen + 1) * d]);
    }
    centers
}

fn kmeans_flat(data: &[f64], n: usize, d: usize, k: usize) -> (Vec<usize>, Vec<f64>) {
    if n == 0 || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut best_labels = vec![0usize; n];
    let mut best_centers = vec![0.0; k * d];
    let mut best_inertia = f64::MAX;
    for init in 0..8 {
        let mut centers = kmeans_pp_init_flat(data, n, d, k, 42 + init as u64 * 997);
        let mut labels = vec![0usize; n];
        for _ in 0..80 {
            let mut changed = false;
            for i in 0..n {
                let row = &data[i * d..(i + 1) * d];
                let mut best_c = 0usize;
                let mut best_dist = f64::MAX;
                for c in 0..k {
                    let center = &centers[c * d..(c + 1) * d];
                    let mut dist = 0.0;
                    for j in 0..d {
                        let diff = row[j] - center[j];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                if labels[i] != best_c {
                    labels[i] = best_c;
                    changed = true;
                }
            }
            let mut new_centers = vec![0.0; k * d];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let label = labels[i];
                counts[label] += 1;
                for j in 0..d {
                    new_centers[label * d + j] += data[i * d + j];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..d {
                        new_centers[c * d + j] /= counts[c] as f64;
                    }
                } else {
                    new_centers[c * d..(c + 1) * d].copy_from_slice(&centers[c * d..(c + 1) * d]);
                }
            }
            centers = new_centers;
            if !changed {
                break;
            }
        }
        let mut inertia = 0.0;
        for i in 0..n {
            let label = labels[i];
            for j in 0..d {
                let diff = data[i * d + j] - centers[label * d + j];
                inertia += diff * diff;
            }
        }
        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = labels;
            best_centers = centers;
        }
    }
    (best_labels, best_centers)
}

fn extract_segment_day(
    day_data: &[f64],
    n_minutes: usize,
    n_stocks: usize,
    n_fields: usize,
    start: usize,
    end: usize,
    path_bins: usize,
    summary_dim: usize,
) -> SegmentDayFeatures {
    let mut ret = vec![f64::NAN; n_stocks];
    let mut intraday_vol = vec![f64::NAN; n_stocks];
    let mut raw_amt = vec![f64::NAN; n_stocks];
    let mut log_amt = vec![f64::NAN; n_stocks];
    let mut buy_ratio = vec![f64::NAN; n_stocks];
    let mut net_flow = vec![f64::NAN; n_stocks];
    let mut log_avg_trade_amt = vec![f64::NAN; n_stocks];
    let mut range_ratio = vec![f64::NAN; n_stocks];
    let mut close_loc = vec![f64::NAN; n_stocks];
    let mut tick_balance = vec![f64::NAN; n_stocks];
    let mut depth1 = vec![f64::NAN; n_stocks];
    let mut depth6 = vec![f64::NAN; n_stocks];
    let mut bid10 = vec![f64::NAN; n_stocks];
    let mut vwap_gap = vec![f64::NAN; n_stocks];
    let mut vwap_ladder_gap = vec![f64::NAN; n_stocks];
    let mut queue1 = vec![f64::NAN; n_stocks];
    let mut queue123 = vec![f64::NAN; n_stocks];
    let mut flow_eff = vec![f64::NAN; n_stocks];
    let mut big_order_ratio = vec![f64::NAN; n_stocks];
    let mut book_pressure = vec![f64::NAN; n_stocks];
    let mut summary = vec![f64::NAN; n_stocks * summary_dim];

    let mut ret_path = vec![f64::NAN; n_stocks * path_bins];
    let mut cumret_path = vec![f64::NAN; n_stocks * path_bins];
    let mut rv_path = vec![f64::NAN; n_stocks * path_bins];
    let mut amt_profile = vec![f64::NAN; n_stocks * path_bins];
    let mut net_flow_path = vec![f64::NAN; n_stocks * path_bins];
    let mut buy_ratio_path = vec![f64::NAN; n_stocks * path_bins];
    let mut trade_size_path = vec![f64::NAN; n_stocks * path_bins];
    let mut tick_balance_path = vec![f64::NAN; n_stocks * path_bins];
    let mut depth1_path = vec![f64::NAN; n_stocks * path_bins];
    let mut depth6_path = vec![f64::NAN; n_stocks * path_bins];
    let mut queue_path = vec![f64::NAN; n_stocks * path_bins];
    let mut quote_gap_path = vec![f64::NAN; n_stocks * path_bins];

    let get = |minute: usize, stock: usize, field: usize| -> f64 {
        day_data[minute * n_stocks * n_fields + stock * n_fields + field]
    };

    for s in 0..n_stocks {
        let mut first_open = f64::NAN;
        let mut last_close = f64::NAN;
        let mut high_max = f64::NEG_INFINITY;
        let mut low_min = f64::INFINITY;
        let mut amt_sum = 0.0;
        let mut buy_amt_sum = 0.0;
        let mut sell_amt_sum = 0.0;
        let mut buy_vol_sum = 0.0;
        let mut sell_vol_sum = 0.0;
        let mut buy_cnt_sum = 0.0;
        let mut sell_cnt_sum = 0.0;
        let mut up_tick_sum = 0.0;
        let mut down_tick_sum = 0.0;
        let mut depth1_vals = Vec::with_capacity(end - start);
        let mut depth6_vals = Vec::with_capacity(end - start);
        let mut bid10_vals = Vec::with_capacity(end - start);
        let mut vwap_gap_vals = Vec::with_capacity(end - start);
        let mut vwap_ladder_vals = Vec::with_capacity(end - start);
        let mut queue1_vals = Vec::with_capacity(end - start);
        let mut queue123_vals = Vec::with_capacity(end - start);
        let mut minute_amounts = Vec::with_capacity(end - start);
        let mut log_rets = Vec::with_capacity(end - start);
        let mut prev_close = f64::NAN;

        for m in start..end.min(n_minutes) {
            let op = get(m, s, FIELD_OPEN);
            let cl = get(m, s, FIELD_CLOSE);
            if !first_open.is_finite() && op > 0.0 {
                first_open = op;
            }
            if cl > 0.0 {
                last_close = cl;
            }
            let hi = get(m, s, FIELD_HIGH);
            let lo = get(m, s, FIELD_LOW);
            if hi.is_finite() {
                high_max = high_max.max(hi);
            }
            if lo.is_finite() {
                low_min = low_min.min(lo);
            }

            let amt = get(m, s, FIELD_AMOUNT);
            if amt.is_finite() {
                amt_sum += amt.max(0.0);
                minute_amounts.push(amt.max(0.0));
            }
            let buy_amt = get(m, s, FIELD_ACT_BUY_AMT);
            let sell_amt = get(m, s, FIELD_ACT_SELL_AMT);
            let buy_vol = get(m, s, FIELD_ACT_BUY_VOL);
            let sell_vol = get(m, s, FIELD_ACT_SELL_VOL);
            let buy_cnt = get(m, s, FIELD_ACT_BUY_CNT);
            let sell_cnt = get(m, s, FIELD_ACT_SELL_CNT);
            let upt = get(m, s, FIELD_UP_TICK);
            let dnt = get(m, s, FIELD_DOWN_TICK);
            if buy_amt.is_finite() {
                buy_amt_sum += buy_amt.max(0.0);
            }
            if sell_amt.is_finite() {
                sell_amt_sum += sell_amt.max(0.0);
            }
            if buy_vol.is_finite() {
                buy_vol_sum += buy_vol.max(0.0);
            }
            if sell_vol.is_finite() {
                sell_vol_sum += sell_vol.max(0.0);
            }
            if buy_cnt.is_finite() {
                buy_cnt_sum += buy_cnt.max(0.0);
            }
            if sell_cnt.is_finite() {
                sell_cnt_sum += sell_cnt.max(0.0);
            }
            if upt.is_finite() {
                up_tick_sum += upt.max(0.0);
            }
            if dnt.is_finite() {
                down_tick_sum += dnt.max(0.0);
            }

            let bid1 = get(m, s, FIELD_BID_SIZE1);
            let ask1 = get(m, s, FIELD_ASK_SIZE1);
            let bid6 = get(m, s, FIELD_BID_SIZE6);
            let ask6 = get(m, s, FIELD_ASK_SIZE6);
            let bid10v = get(m, s, FIELD_BID_SIZE10);
            let ask10v = get(m, s, FIELD_ASK_SIZE10);
            if bid1.is_finite() && ask1.is_finite() {
                depth1_vals.push((bid1 - ask1) / (bid1 + ask1 + EPS));
            }
            if bid6.is_finite() && ask6.is_finite() {
                depth6_vals.push((bid6 - ask6) / (bid6 + ask6 + EPS));
            }
            if bid10v.is_finite() && ask10v.is_finite() {
                bid10_vals.push((bid10v - ask10v) / (bid10v + ask10v + EPS));
            }

            let bg = get(m, s, FIELD_BID_VWAP_MEAN);
            let ag = get(m, s, FIELD_ASK_VWAP_MEAN);
            if bg.is_finite() && ag.is_finite() {
                vwap_gap_vals.push(bg - ag);
            }

            let ladder = [
                get(m, s, FIELD_BID_VWAP3) - get(m, s, FIELD_ASK_VWAP3),
                get(m, s, FIELD_BID_VWAP5) - get(m, s, FIELD_ASK_VWAP5),
                get(m, s, FIELD_BID_VWAP10) - get(m, s, FIELD_ASK_VWAP10),
            ];
            let ladder_mean = mean(&ladder);
            if ladder_mean.is_finite() {
                vwap_ladder_vals.push(ladder_mean);
            }

            let b1 = get(m, s, FIELD_BID_VOL1);
            let a1 = get(m, s, FIELD_ASK_VOL1);
            let b2 = get(m, s, FIELD_BID_VOL2);
            let a2 = get(m, s, FIELD_ASK_VOL2);
            let b3 = get(m, s, FIELD_BID_VOL3);
            let a3 = get(m, s, FIELD_ASK_VOL3);
            if b1.is_finite() && a1.is_finite() {
                queue1_vals.push((b1 - a1) / (b1 + a1 + EPS));
            }
            let q123 = mean(&[
                (b1 - a1) / (b1 + a1 + EPS),
                (b2 - a2) / (b2 + a2 + EPS),
                (b3 - a3) / (b3 + a3 + EPS),
            ]);
            if q123.is_finite() {
                queue123_vals.push(q123);
            }

            if prev_close > 0.0 && cl > 0.0 {
                log_rets.push((cl / prev_close).ln());
            }
            if cl > 0.0 {
                prev_close = cl;
            }
        }

        ret[s] = if first_open > 0.0 && last_close > 0.0 {
            last_close / first_open - 1.0
        } else {
            f64::NAN
        };
        intraday_vol[s] = std(&log_rets);
        raw_amt[s] = finite_or_nan(amt_sum);
        log_amt[s] = if amt_sum > 0.0 {
            amt_sum.ln()
        } else {
            f64::NAN
        };
        buy_ratio[s] = if amt_sum > 0.0 {
            buy_amt_sum / amt_sum
        } else {
            f64::NAN
        };
        net_flow[s] = if amt_sum > 0.0 {
            (buy_amt_sum - sell_amt_sum) / amt_sum
        } else {
            f64::NAN
        };
        let total_cnt = buy_cnt_sum + sell_cnt_sum;
        log_avg_trade_amt[s] = if amt_sum > 0.0 && total_cnt > 0.0 {
            (amt_sum / total_cnt).ln()
        } else {
            f64::NAN
        };
        range_ratio[s] = if first_open > 0.0 && high_max.is_finite() && low_min.is_finite() {
            (high_max - low_min) / first_open
        } else {
            f64::NAN
        };
        close_loc[s] = if high_max.is_finite() && low_min.is_finite() && last_close.is_finite() {
            (last_close - low_min) / (high_max - low_min + EPS)
        } else {
            f64::NAN
        };
        tick_balance[s] = (up_tick_sum - down_tick_sum) / (up_tick_sum + down_tick_sum + EPS);
        depth1[s] = mean(&depth1_vals);
        depth6[s] = mean(&depth6_vals);
        bid10[s] = mean(&bid10_vals);
        vwap_gap[s] = mean(&vwap_gap_vals);
        vwap_ladder_gap[s] = mean(&vwap_ladder_vals);
        queue1[s] = mean(&queue1_vals);
        queue123[s] = mean(&queue123_vals);
        flow_eff[s] = if net_flow[s].is_finite() {
            ret[s] / (net_flow[s].abs() + 1e-6)
        } else {
            f64::NAN
        };
        book_pressure[s] = mean(&[depth1[s], depth6[s], bid10[s], queue123[s]]);

        if !minute_amounts.is_empty() && amt_sum > 0.0 {
            let mut sorted = minute_amounts.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let median = sorted[sorted.len() / 2];
            let big_sum: f64 = minute_amounts.iter().filter(|&&v| v > median).sum();
            big_order_ratio[s] = big_sum / amt_sum;
        }

        for b in 0..path_bins {
            let bs = start + (end - start) * b / path_bins;
            let be = start + (end - start) * (b + 1) / path_bins;
            let mut bin_first_open = f64::NAN;
            let mut bin_last_close = f64::NAN;
            let mut bin_amt = 0.0;
            let mut bin_buy_amt = 0.0;
            let mut bin_sell_amt = 0.0;
            let mut bin_cnt = 0.0;
            let mut bin_up = 0.0;
            let mut bin_down = 0.0;
            let mut bin_log_rets = Vec::with_capacity(be.saturating_sub(bs));
            let mut bin_prev_close = f64::NAN;
            let mut bin_depth1_vals = Vec::with_capacity(be.saturating_sub(bs));
            let mut bin_depth6_vals = Vec::with_capacity(be.saturating_sub(bs));
            let mut bin_queue_vals = Vec::with_capacity(be.saturating_sub(bs));
            let mut bin_quote_vals = Vec::with_capacity(be.saturating_sub(bs));

            for m in bs..be.min(n_minutes) {
                let op = get(m, s, FIELD_OPEN);
                let cl = get(m, s, FIELD_CLOSE);
                if !bin_first_open.is_finite() && op > 0.0 {
                    bin_first_open = op;
                }
                if cl > 0.0 {
                    bin_last_close = cl;
                }
                let amt = get(m, s, FIELD_AMOUNT);
                if amt.is_finite() {
                    bin_amt += amt.max(0.0);
                }
                let buy_amt = get(m, s, FIELD_ACT_BUY_AMT);
                let sell_amt = get(m, s, FIELD_ACT_SELL_AMT);
                if buy_amt.is_finite() {
                    bin_buy_amt += buy_amt.max(0.0);
                }
                if sell_amt.is_finite() {
                    bin_sell_amt += sell_amt.max(0.0);
                }
                let buy_cnt = get(m, s, FIELD_ACT_BUY_CNT);
                let sell_cnt = get(m, s, FIELD_ACT_SELL_CNT);
                if buy_cnt.is_finite() {
                    bin_cnt += buy_cnt.max(0.0);
                }
                if sell_cnt.is_finite() {
                    bin_cnt += sell_cnt.max(0.0);
                }
                let upt = get(m, s, FIELD_UP_TICK);
                let dnt = get(m, s, FIELD_DOWN_TICK);
                if upt.is_finite() {
                    bin_up += upt.max(0.0);
                }
                if dnt.is_finite() {
                    bin_down += dnt.max(0.0);
                }

                let bid1 = get(m, s, FIELD_BID_SIZE1);
                let ask1 = get(m, s, FIELD_ASK_SIZE1);
                let bid6 = get(m, s, FIELD_BID_SIZE6);
                let ask6 = get(m, s, FIELD_ASK_SIZE6);
                if bid1.is_finite() && ask1.is_finite() {
                    bin_depth1_vals.push((bid1 - ask1) / (bid1 + ask1 + EPS));
                }
                if bid6.is_finite() && ask6.is_finite() {
                    bin_depth6_vals.push((bid6 - ask6) / (bid6 + ask6 + EPS));
                }
                let b1 = get(m, s, FIELD_BID_VOL1);
                let a1 = get(m, s, FIELD_ASK_VOL1);
                if b1.is_finite() && a1.is_finite() {
                    bin_queue_vals.push((b1 - a1) / (b1 + a1 + EPS));
                }
                let bg = get(m, s, FIELD_BID_VWAP_MEAN);
                let ag = get(m, s, FIELD_ASK_VWAP_MEAN);
                if bg.is_finite() && ag.is_finite() {
                    bin_quote_vals.push(bg - ag);
                }

                if bin_prev_close > 0.0 && cl > 0.0 {
                    bin_log_rets.push((cl / bin_prev_close).ln());
                }
                if cl > 0.0 {
                    bin_prev_close = cl;
                }
            }

            let idx = s * path_bins + b;
            ret_path[idx] = if bin_first_open > 0.0 && bin_last_close > 0.0 {
                bin_last_close / bin_first_open - 1.0
            } else {
                f64::NAN
            };
            cumret_path[idx] = if first_open > 0.0 && bin_last_close > 0.0 {
                bin_last_close / first_open - 1.0
            } else {
                f64::NAN
            };
            rv_path[idx] = std(&bin_log_rets);
            amt_profile[idx] = if amt_sum > 0.0 {
                bin_amt / amt_sum
            } else {
                f64::NAN
            };
            net_flow_path[idx] = if bin_amt > 0.0 {
                (bin_buy_amt - bin_sell_amt) / bin_amt
            } else {
                f64::NAN
            };
            buy_ratio_path[idx] = if bin_amt > 0.0 {
                bin_buy_amt / bin_amt
            } else {
                f64::NAN
            };
            trade_size_path[idx] = if bin_amt > 0.0 && bin_cnt > 0.0 {
                (bin_amt / bin_cnt).ln()
            } else {
                f64::NAN
            };
            tick_balance_path[idx] = (bin_up - bin_down) / (bin_up + bin_down + EPS);
            depth1_path[idx] = mean(&bin_depth1_vals);
            depth6_path[idx] = mean(&bin_depth6_vals);
            queue_path[idx] = mean(&bin_queue_vals);
            quote_gap_path[idx] = mean(&bin_quote_vals);
        }

        // 根据 summary_dim 选择聚类特征:
        // 3维: ret, log_amt, net_flow
        // 7维: ret, intraday_vol, log_amt, buy_ratio, range_ratio, close_loc, depth1
        // 17维: 全部
        let all_fields = [
            ret[s],
            intraday_vol[s],
            log_amt[s],
            buy_ratio[s],
            net_flow[s],
            log_avg_trade_amt[s],
            range_ratio[s],
            close_loc[s],
            tick_balance[s],
            depth1[s],
            depth6[s],
            bid10[s],
            vwap_gap[s],
            vwap_ladder_gap[s],
            queue123[s],
            flow_eff[s],
            big_order_ratio[s],
        ];
        let selected: Vec<f64> = match summary_dim {
            3 => vec![ret[s], log_amt[s], net_flow[s]],
            7 => vec![
                ret[s],
                intraday_vol[s],
                log_amt[s],
                buy_ratio[s],
                range_ratio[s],
                close_loc[s],
                depth1[s],
            ],
            _ => all_fields.to_vec(),
        };
        for (j, &v) in selected.iter().enumerate() {
            summary[s * summary_dim + j] = v;
        }
    }

    SegmentDayFeatures {
        ret,
        intraday_vol,
        raw_amt,
        log_amt,
        buy_ratio,
        net_flow,
        log_avg_trade_amt,
        range_ratio,
        close_loc,
        tick_balance,
        depth1,
        depth6,
        bid10,
        vwap_gap,
        vwap_ladder_gap,
        queue1,
        queue123,
        flow_eff,
        big_order_ratio,
        book_pressure,
        summary,
        ret_path,
        cumret_path,
        rv_path,
        amt_profile,
        net_flow_path,
        buy_ratio_path,
        trade_size_path,
        tick_balance_path,
        depth1_path,
        depth6_path,
        queue_path,
        quote_gap_path,
        path_bins,
        n_stocks,
        summary_dim,
    }
}

fn cluster_segment_day(features: &SegmentDayFeatures, k: usize) -> ClusterDayResult {
    let n = features.n_stocks;
    let d = features.summary_dim;
    let mut valid_indices = Vec::with_capacity(n);
    for i in 0..n {
        let mut ok = true;
        for j in 0..d {
            if !features.summary[i * d + j].is_finite() {
                ok = false;
                break;
            }
        }
        if ok {
            valid_indices.push(i);
        }
    }
    if valid_indices.is_empty() {
        return ClusterDayResult {
            labels: vec![usize::MAX; n],
            centers: Vec::new(),
            k: 0,
            mean_ret: Vec::new(),
            std_ret: Vec::new(),
            mean_buy_ratio: Vec::new(),
            mean_net_flow: Vec::new(),
            mean_log_amt: Vec::new(),
            mean_intraday_vol: Vec::new(),
            mean_range: Vec::new(),
            mean_close_loc: Vec::new(),
            mean_flow_eff: Vec::new(),
            mean_book_pressure: Vec::new(),
            total_amt: Vec::new(),
            breadth: Vec::new(),
            leader_gap: Vec::new(),
            skew_ret: Vec::new(),
            theme_heat: Vec::new(),
            ret_path_mean: Vec::new(),
            flow_path_mean: Vec::new(),
            amt_path_mean: Vec::new(),
            depth_path_mean: Vec::new(),
            return_rank_pct: vec![f64::NAN; n],
            flow_rank_pct: vec![f64::NAN; n],
            buy_rank_pct: vec![f64::NAN; n],
            dists_to_centers: Vec::new(),
        };
    }

    let n_valid = valid_indices.len();
    let actual_k = k.min(n_valid).max(1);
    let mut clean = vec![0.0; n_valid * d];
    for (idx, &orig) in valid_indices.iter().enumerate() {
        for j in 0..d {
            clean[idx * d + j] = features.summary[orig * d + j];
        }
    }
    let scaled = zscore_standardize_flat(&clean, n_valid, d);
    let (labels_clean, centers) = kmeans_flat(&scaled, n_valid, d, actual_k);
    let mut labels = vec![usize::MAX; n];
    for (idx, &orig) in valid_indices.iter().enumerate() {
        labels[orig] = labels_clean[idx];
    }

    let mut dists_to_centers = vec![f64::INFINITY; n * actual_k];
    for (idx, &orig) in valid_indices.iter().enumerate() {
        for c in 0..actual_k {
            let mut dist = 0.0;
            for j in 0..d {
                let diff = scaled[idx * d + j] - centers[c * d + j];
                dist += diff * diff;
            }
            dists_to_centers[orig * actual_k + c] = dist.sqrt();
        }
    }

    let mut mean_ret = vec![0.0; actual_k];
    let mut std_ret = vec![0.0; actual_k];
    let mut mean_buy_ratio = vec![0.0; actual_k];
    let mut mean_net_flow = vec![0.0; actual_k];
    let mut mean_log_amt = vec![0.0; actual_k];
    let mut mean_intraday_vol = vec![0.0; actual_k];
    let mut mean_range = vec![0.0; actual_k];
    let mut mean_close_loc = vec![0.0; actual_k];
    let mut mean_flow_eff = vec![0.0; actual_k];
    let mut mean_book_pressure = vec![0.0; actual_k];
    let mut total_amt = vec![0.0; actual_k];
    let mut breadth = vec![0.0; actual_k];
    let mut leader_gap = vec![0.0; actual_k];
    let mut skew_ret = vec![0.0; actual_k];
    let mut theme_heat = vec![0.0; actual_k];
    let mut ret_path_mean = vec![0.0; actual_k * features.path_bins];
    let mut flow_path_mean = vec![0.0; actual_k * features.path_bins];
    let mut amt_path_mean = vec![0.0; actual_k * features.path_bins];
    let mut depth_path_mean = vec![0.0; actual_k * features.path_bins];
    let mut counts = vec![0usize; actual_k];
    let mut ret_lists: Vec<Vec<f64>> = (0..actual_k).map(|_| Vec::new()).collect();
    let market_total_amt: f64 = features.raw_amt.iter().filter(|v| v.is_finite()).sum();

    for i in 0..n {
        let label = labels[i];
        if label >= actual_k {
            continue;
        }
        counts[label] += 1;
        let r = features.ret[i];
        if r.is_finite() {
            mean_ret[label] += r;
            ret_lists[label].push(r);
            if r > 0.0 {
                breadth[label] += 1.0;
            }
        }
        if features.buy_ratio[i].is_finite() {
            mean_buy_ratio[label] += features.buy_ratio[i];
        }
        if features.net_flow[i].is_finite() {
            mean_net_flow[label] += features.net_flow[i];
        }
        if features.log_amt[i].is_finite() {
            mean_log_amt[label] += features.log_amt[i];
        }
        if features.intraday_vol[i].is_finite() {
            mean_intraday_vol[label] += features.intraday_vol[i];
        }
        if features.range_ratio[i].is_finite() {
            mean_range[label] += features.range_ratio[i];
        }
        if features.close_loc[i].is_finite() {
            mean_close_loc[label] += features.close_loc[i];
        }
        if features.flow_eff[i].is_finite() {
            mean_flow_eff[label] += features.flow_eff[i];
        }
        if features.book_pressure[i].is_finite() {
            mean_book_pressure[label] += features.book_pressure[i];
        }
        if features.raw_amt[i].is_finite() {
            total_amt[label] += features.raw_amt[i];
        }
        for b in 0..features.path_bins {
            let idx = i * features.path_bins + b;
            let base = label * features.path_bins + b;
            if features.ret_path[idx].is_finite() {
                ret_path_mean[base] += features.ret_path[idx];
            }
            if features.net_flow_path[idx].is_finite() {
                flow_path_mean[base] += features.net_flow_path[idx];
            }
            if features.amt_profile[idx].is_finite() {
                amt_path_mean[base] += features.amt_profile[idx];
            }
            if features.depth1_path[idx].is_finite() {
                depth_path_mean[base] += features.depth1_path[idx];
            }
        }
    }

    for c in 0..actual_k {
        if counts[c] > 0 {
            let cntf = counts[c] as f64;
            mean_ret[c] /= cntf;
            mean_buy_ratio[c] /= cntf;
            mean_net_flow[c] /= cntf;
            mean_log_amt[c] /= cntf;
            mean_intraday_vol[c] /= cntf;
            mean_range[c] /= cntf;
            mean_close_loc[c] /= cntf;
            mean_flow_eff[c] /= cntf;
            mean_book_pressure[c] /= cntf;
            breadth[c] /= cntf;
            for b in 0..features.path_bins {
                let base = c * features.path_bins + b;
                ret_path_mean[base] /= cntf;
                flow_path_mean[base] /= cntf;
                amt_path_mean[base] /= cntf;
                depth_path_mean[base] /= cntf;
            }
        }
        std_ret[c] = std(&ret_lists[c]);
        skew_ret[c] = skew(&ret_lists[c]);
        if !ret_lists[c].is_empty() {
            let mut sorted = ret_lists[c].clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            leader_gap[c] = if sorted.len() >= 2 {
                sorted[0] - sorted[1]
            } else {
                0.0
            };
        }
        theme_heat[c] = if market_total_amt > 0.0 {
            total_amt[c] / market_total_amt
        } else {
            f64::NAN
        };
    }

    let return_rank_pct = rank_percentile(&features.ret, &labels, actual_k);
    let flow_rank_pct = rank_percentile(&features.net_flow, &labels, actual_k);
    let buy_rank_pct = rank_percentile(&features.buy_ratio, &labels, actual_k);

    ClusterDayResult {
        labels,
        centers,
        k: actual_k,
        mean_ret,
        std_ret,
        mean_buy_ratio,
        mean_net_flow,
        mean_log_amt,
        mean_intraday_vol,
        mean_range,
        mean_close_loc,
        mean_flow_eff,
        mean_book_pressure,
        total_amt,
        breadth,
        leader_gap,
        skew_ret,
        theme_heat,
        ret_path_mean,
        flow_path_mean,
        amt_path_mean,
        depth_path_mean,
        return_rank_pct,
        flow_rank_pct,
        buy_rank_pct,
        dists_to_centers,
    }
}

fn theme_value_by_label(values: &[f64], labels: &[usize], stock: usize) -> f64 {
    let label = labels[stock];
    if label < values.len() {
        values[label]
    } else {
        f64::NAN
    }
}

fn path_slice(path: &[f64], idx: usize, bins: usize) -> &[f64] {
    &path[idx * bins..(idx + 1) * bins]
}

fn cluster_path_slice(path: &[f64], label: usize, bins: usize) -> &[f64] {
    &path[label * bins..(label + 1) * bins]
}

/// 返回 (前5距离升序, 前5概率按距离升序排列, 全部k个聚类概率, 第2近聚类label)
fn top5_from_dists(dists: &[f64], k: usize) -> ([f64; 5], [f64; 5], Vec<f64>, usize) {
    let mut pairs: Vec<(usize, f64)> = (0..k).map(|i| (i, dists[i])).collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let mut top5_dists = [f64::NAN; 5];
    for i in 0..5.min(k) {
        top5_dists[i] = pairs[i].1;
    }
    let min_d = pairs.first().map(|x| x.1).unwrap_or(0.0);
    let mut weights = Vec::with_capacity(k);
    let mut sumw = 0.0;
    for &(_, d) in &pairs {
        let w = if d.is_finite() {
            (-(d - min_d)).exp()
        } else {
            0.0
        };
        sumw += w;
        weights.push(w);
    }
    if sumw > 0.0 {
        for w in weights.iter_mut() {
            *w /= sumw;
        }
    }
    let mut probs_by_label = vec![0.0; k];
    for ((label, _), prob) in pairs.iter().zip(weights.iter()) {
        probs_by_label[*label] = *prob;
    }
    // 前5概率，按距离升序排列对应的聚类
    let mut top5_probs = [f64::NAN; 5];
    for i in 0..5.min(k) {
        top5_probs[i] = probs_by_label[pairs[i].0];
    }
    let second_label = if pairs.len() > 1 {
        pairs[1].0
    } else {
        0
    };
    (top5_dists, top5_probs, probs_by_label, second_label)
}

fn array2_from_flat(
    py: Python,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("二维数组形状错误: {}", e)))?;
    Ok(arr.into_pyarray(py).to_owned())
}

#[pyfunction(signature = (minute_data, k=30, summary_dim=17, n_threads=8))]
pub fn compute_theme_feature_expansion_from_minute(
    py: Python,
    minute_data: PyReadonlyArrayDyn<f64>,
    k: usize,
    summary_dim: usize,
    n_threads: usize,
) -> PyResult<(
    Vec<Py<PyArray2<f64>>>,
    Vec<Py<PyArray1<f64>>>,
    Vec<String>,
    Vec<String>,
)> {
    let data = minute_data.as_array();
    let shape = data.shape();
    if shape.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "minute_data必须是4维数组 (n_days, n_minutes, n_stocks, n_fields)",
        ));
    }
    let n_days = shape[0];
    let n_minutes = shape[1];
    let n_stocks = shape[2];
    let n_fields = shape[3];
    if n_fields < 34 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "minute_data字段数不足，至少需要34列",
        ));
    }
    let actual_threads = n_threads.clamp(1, 10);
    if summary_dim != 3 && summary_dim != 7 && summary_dim != 17 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "summary_dim必须是3、7或17",
        ));
    }
    let data_vec: Vec<f64> = data.iter().cloned().collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;

    let all_day_features: Vec<SegmentDayFeatures> = pool.install(|| {
        (0..n_days)
            .into_par_iter()
            .map(|day_idx| {
                let offset = day_idx * n_minutes * n_stocks * n_fields;
                let day_data = &data_vec[offset..offset + n_minutes * n_stocks * n_fields];
                extract_segment_day(
                    day_data,
                    n_minutes,
                    n_stocks,
                    n_fields,
                    0,
                    n_minutes,
                    ALL_DAY_BINS,
                    summary_dim,
                )
            })
            .collect()
    });
    let open_features: Vec<SegmentDayFeatures> = pool.install(|| {
        (0..n_days)
            .into_par_iter()
            .map(|day_idx| {
                let offset = day_idx * n_minutes * n_stocks * n_fields;
                let day_data = &data_vec[offset..offset + n_minutes * n_stocks * n_fields];
                extract_segment_day(
                    day_data,
                    n_minutes,
                    n_stocks,
                    n_fields,
                    0,
                    30.min(n_minutes),
                    SHORT_BINS,
                    summary_dim,
                )
            })
            .collect()
    });
    let close_start = n_minutes.saturating_sub(30);
    let close_features: Vec<SegmentDayFeatures> = pool.install(|| {
        (0..n_days)
            .into_par_iter()
            .map(|day_idx| {
                let offset = day_idx * n_minutes * n_stocks * n_fields;
                let day_data = &data_vec[offset..offset + n_minutes * n_stocks * n_fields];
                extract_segment_day(
                    day_data,
                    n_minutes,
                    n_stocks,
                    n_fields,
                    close_start,
                    n_minutes,
                    SHORT_BINS,
                    summary_dim,
                )
            })
            .collect()
    });

    let all_clusters: Vec<ClusterDayResult> = pool.install(|| {
        all_day_features
            .par_iter()
            .map(|f| cluster_segment_day(f, k))
            .collect()
    });
    let open_clusters: Vec<ClusterDayResult> = pool.install(|| {
        open_features
            .par_iter()
            .map(|f| cluster_segment_day(f, k))
            .collect()
    });
    let close_clusters: Vec<ClusterDayResult> = pool.install(|| {
        close_features
            .par_iter()
            .map(|f| cluster_segment_day(f, k))
            .collect()
    });

    let target_day = n_days - 1;
    let prev_day = target_day.saturating_sub(1);

    let all_f = &all_day_features[target_day];
    let all_c = &all_clusters[target_day];
    let open_f = &open_features[target_day];
    let open_c = &open_clusters[target_day];
    let close_f = &close_features[target_day];
    let close_c = &close_clusters[target_day];

    let mut vector_names = vec![
        "ret_vs_theme_path_8".to_string(),
        "flow_vs_theme_path_8".to_string(),
        "amt_vs_theme_path_8".to_string(),
        "depth_vs_theme_path_8".to_string(),
        "ret_dct_6".to_string(),
        "flow_dct_6".to_string(),
        "joint_latent_6".to_string(),
        "soft_assign_dists_5".to_string(),
        "soft_assign_probs_5".to_string(),
        "theme_gap_vector_3".to_string(),
        "open_close_ret_vs_theme_4".to_string(),
        "open_close_path_vs_theme_6".to_string(),
        "open_close_center_margin_4".to_string(),
        "flow_to_price_transmission_6".to_string(),
        "open_signal_close_confirm_4".to_string(),
        "capital_expression_pricing_confirmation_8".to_string(),
    ];

    let scalar_names = vec![
        "return_deviation",
        "amount_deviation",
        "intraday_vol_deviation",
        "range_deviation",
        "close_location_deviation",
        "flow_efficiency_deviation",
        "book_pressure_deviation",
        "buy_ratio_rank_in_theme",
        "flow_rank_in_theme",
        "amt_share_in_theme",
        "yesterday_theme_buy_ratio",
        "yesterday_theme_dispersion",
        "yesterday_theme_breadth",
        "yesterday_theme_amount",
        "yesterday_theme_flow_efficiency",
        "yesterday_theme_leader_gap",
        "yesterday_theme_skewness",
        "path_cos_to_theme",
        "path_l2_to_theme",
        "early_late_divergence",
        "trend_noise_ratio",
        "center_margin",
        "relative_center_margin",
        "soft_membership_entropy",
        "second_theme_return_gap",
        "second_theme_buy_gap",
        "open_flow_dev_x_close_ret_dev",
        "open_buy_dev_x_close_theme_ret",
        "open_flow_dev_x_close_center_margin",
        "open_ret_dev_x_close_ret_dev",
        "open_flow_rank_x_close_rank",
        "open_strong_close_weak",
        "open_strong_close_fade",
        "open_flow_unconfirmed",
        "open_theme_hot_close_cold",
        "close_recovery_after_weak_open",
        "close_flow_recovery_after_weak_open",
        "late_confirmation_strength",
        "late_capital_takeover",
        "theme_core_upgrade",
        "theme_entropy_drop",
        "edge_to_core_signal",
        "cross_theme_upgrade",
        "expression_confirmation_cos",
        "expression_confirmation_l2",
        "signal_consistency_score",
        "capital_price_agreement",
        "open_act_buy_dev_x_yesterday_close_theme_ret",
        "open_flow_dev_x_yesterday_theme_buy_ratio",
        "open_center_margin_x_yesterday_theme_ret",
        "open_soft_entropy_x_yesterday_theme_dispersion",
        "close_confirm_x_yesterday_theme_return",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();

    let vector_dims = [
        8usize, 8, 8, 8, 6, 6, 6, 5, 5, 3, 4, 6, 4, 6, 4, 8,
    ];
    let mut vector_data: Vec<Vec<f64>> = vector_dims
        .iter()
        .map(|&dim| vec![f64::NAN; n_stocks * dim])
        .collect();
    let mut scalar_data: Vec<Vec<f64>> = scalar_names
        .iter()
        .map(|_| vec![f64::NAN; n_stocks])
        .collect();

    for s in 0..n_stocks {
        let all_label = all_c.labels[s];
        let open_label = open_c.labels[s];
        let close_label = close_c.labels[s];
        if all_label == usize::MAX || all_label >= all_c.k {
            continue;
        }
        let all_theme_ret = all_c.mean_ret[all_label];
        let all_theme_buy = all_c.mean_buy_ratio[all_label];
        let all_theme_log_amt = all_c.mean_log_amt[all_label];
        let all_theme_vol = all_c.mean_intraday_vol[all_label];
        let all_theme_range = all_c.mean_range[all_label];
        let all_theme_close_loc = all_c.mean_close_loc[all_label];
        let all_theme_flow_eff = all_c.mean_flow_eff[all_label];
        let all_theme_book = all_c.mean_book_pressure[all_label];
        let all_theme_total_amt = all_c.total_amt[all_label];

        let all_ret_path = path_slice(&all_f.ret_path, s, ALL_DAY_BINS);
        let all_flow_path = path_slice(&all_f.net_flow_path, s, ALL_DAY_BINS);
        let all_amt_path = path_slice(&all_f.amt_profile, s, ALL_DAY_BINS);
        let all_depth_path = path_slice(&all_f.depth1_path, s, ALL_DAY_BINS);
        let theme_ret_path = cluster_path_slice(&all_c.ret_path_mean, all_label, ALL_DAY_BINS);
        let theme_flow_path = cluster_path_slice(&all_c.flow_path_mean, all_label, ALL_DAY_BINS);
        let theme_amt_path = cluster_path_slice(&all_c.amt_path_mean, all_label, ALL_DAY_BINS);
        let theme_depth_path = cluster_path_slice(&all_c.depth_path_mean, all_label, ALL_DAY_BINS);

        let ret_vs_theme: Vec<f64> = all_ret_path
            .iter()
            .zip(theme_ret_path.iter())
            .map(|(a, b)| a - b)
            .collect();
        let flow_vs_theme: Vec<f64> = all_flow_path
            .iter()
            .zip(theme_flow_path.iter())
            .map(|(a, b)| a - b)
            .collect();
        let amt_vs_theme: Vec<f64> = all_amt_path
            .iter()
            .zip(theme_amt_path.iter())
            .map(|(a, b)| a - b)
            .collect();
        let depth_vs_theme: Vec<f64> = all_depth_path
            .iter()
            .zip(theme_depth_path.iter())
            .map(|(a, b)| a - b)
            .collect();

        let ret_dct = dct_first_k(all_ret_path, 6);
        let flow_dct = dct_first_k(all_flow_path, 6);
        let depth_dct = dct_first_k(all_depth_path, 2);
        let joint_latent = vec![
            ret_dct[0],
            ret_dct[1],
            flow_dct[0],
            flow_dct[1],
            depth_dct[0],
            depth_dct[1],
        ];

        for b in 0..ALL_DAY_BINS {
            vector_data[0][s * ALL_DAY_BINS + b] = ret_vs_theme[b];
            vector_data[1][s * ALL_DAY_BINS + b] = flow_vs_theme[b];
            vector_data[2][s * ALL_DAY_BINS + b] = amt_vs_theme[b];
            vector_data[3][s * ALL_DAY_BINS + b] = depth_vs_theme[b];
        }
        for j in 0..6 {
            vector_data[4][s * 6 + j] = ret_dct[j];
            vector_data[5][s * 6 + j] = flow_dct[j];
            vector_data[6][s * 6 + j] = joint_latent[j];
        }

        let dists = &all_c.dists_to_centers[s * all_c.k..(s + 1) * all_c.k];
        let (top5_dists, top5_probs, probs_all, second_label) = top5_from_dists(dists, all_c.k);
        for i in 0..5 {
            vector_data[7][s * 5 + i] = top5_dists[i];
            vector_data[8][s * 5 + i] = top5_probs[i];
        }
        let second_label = second_label.min(all_c.k.saturating_sub(1));
        vector_data[9][s * 3] = all_c.mean_ret[all_label] - all_c.mean_ret[second_label];
        vector_data[9][s * 3 + 1] =
            all_c.mean_buy_ratio[all_label] - all_c.mean_buy_ratio[second_label];
        vector_data[9][s * 3 + 2] = all_c.theme_heat[all_label] - all_c.theme_heat[second_label];

        if open_label < open_c.k && close_label < close_c.k {
            let open_ret_dev =
                open_f.ret[s] - theme_value_by_label(&open_c.mean_ret, &open_c.labels, s);
            let open_flow_dev =
                open_f.net_flow[s] - theme_value_by_label(&open_c.mean_net_flow, &open_c.labels, s);
            let close_ret_dev =
                close_f.ret[s] - theme_value_by_label(&close_c.mean_ret, &close_c.labels, s);
            let close_flow_dev = close_f.net_flow[s]
                - theme_value_by_label(&close_c.mean_net_flow, &close_c.labels, s);
            vector_data[10][s * 4] = open_ret_dev;
            vector_data[10][s * 4 + 1] = open_flow_dev;
            vector_data[10][s * 4 + 2] = close_ret_dev;
            vector_data[10][s * 4 + 3] = close_flow_dev;

            let open_path = path_slice(&open_f.ret_path, s, SHORT_BINS);
            let open_theme_path = cluster_path_slice(&open_c.ret_path_mean, open_label, SHORT_BINS);
            let close_path = path_slice(&close_f.ret_path, s, SHORT_BINS);
            let close_theme_path =
                cluster_path_slice(&close_c.ret_path_mean, close_label, SHORT_BINS);
            let open_cos = cosine_similarity(open_path, open_theme_path);
            let close_cos = cosine_similarity(close_path, close_theme_path);
            let open_l2 = l2_distance(open_path, open_theme_path);
            let close_l2 = l2_distance(close_path, close_theme_path);
            vector_data[11][s * 6] = open_cos;
            vector_data[11][s * 6 + 1] = open_l2;
            vector_data[11][s * 6 + 2] = close_cos;
            vector_data[11][s * 6 + 3] = close_l2;
            vector_data[11][s * 6 + 4] = close_cos - open_cos;
            vector_data[11][s * 6 + 5] = close_l2 - open_l2;

            let open_dists = &open_c.dists_to_centers[s * open_c.k..(s + 1) * open_c.k];
            let (open_top5_dists, _, open_probs_all, _) = top5_from_dists(open_dists, open_c.k);
            let close_dists = &close_c.dists_to_centers[s * close_c.k..(s + 1) * close_c.k];
            let (_, close_top5_dists, close_probs_all, _) = top5_from_dists(close_dists, close_c.k);
            let open_entropy: f64 = open_probs_all
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            let close_entropy: f64 = close_probs_all
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            let open_margin = open_top5_dists[1] - open_top5_dists[0];
            let close_margin = close_top5_dists[1] - close_top5_dists[0];
            vector_data[12][s * 4] = open_margin;
            vector_data[12][s * 4 + 1] = close_margin;
            vector_data[12][s * 4 + 2] = close_margin - open_margin;
            vector_data[12][s * 4 + 3] = open_entropy - close_entropy;

            vector_data[13][s * 6] = open_flow_dev;
            vector_data[13][s * 6 + 1] = open_ret_dev;
            vector_data[13][s * 6 + 2] = close_flow_dev;
            vector_data[13][s * 6 + 3] = close_ret_dev;
            vector_data[13][s * 6 + 4] = close_ret_dev - open_flow_dev;
            vector_data[13][s * 6 + 5] =
                theme_value_by_label(&close_c.mean_ret, &close_c.labels, s) - open_flow_dev;

            let open_buy_dev = open_f.buy_ratio[s]
                - theme_value_by_label(&open_c.mean_buy_ratio, &open_c.labels, s);
            vector_data[14][s * 4] = open_buy_dev;
            vector_data[14][s * 4 + 1] =
                theme_value_by_label(&close_c.mean_ret, &close_c.labels, s);
            vector_data[14][s * 4 + 2] = close_ret_dev;
            vector_data[14][s * 4 + 3] = open_buy_dev * close_ret_dev;

            let open_depth_dev = open_f.depth1[s]
                - theme_value_by_label(&open_c.mean_book_pressure, &open_c.labels, s);
            let expr_vec = [open_flow_dev, open_buy_dev, open_depth_dev, open_margin];
            let conf_vec = [
                close_ret_dev,
                theme_value_by_label(&close_c.mean_ret, &close_c.labels, s),
                close_margin,
                close_cos,
            ];
            vector_data[15][s * 8] = open_flow_dev;
            vector_data[15][s * 8 + 1] = open_buy_dev;
            vector_data[15][s * 8 + 2] = open_depth_dev;
            vector_data[15][s * 8 + 3] = open_margin;
            vector_data[15][s * 8 + 4] = close_ret_dev;
            vector_data[15][s * 8 + 5] =
                theme_value_by_label(&close_c.mean_ret, &close_c.labels, s);
            vector_data[15][s * 8 + 6] = close_margin;
            vector_data[15][s * 8 + 7] = cosine_similarity(&expr_vec, &conf_vec);
        }

        scalar_data[0][s] = all_f.ret[s] - all_theme_ret;
        scalar_data[1][s] = all_f.log_amt[s] - all_theme_log_amt;
        scalar_data[2][s] = all_f.intraday_vol[s] - all_theme_vol;
        scalar_data[3][s] = all_f.range_ratio[s] - all_theme_range;
        scalar_data[4][s] = all_f.close_loc[s] - all_theme_close_loc;
        scalar_data[5][s] = all_f.flow_eff[s] - all_theme_flow_eff;
        scalar_data[6][s] = all_f.book_pressure[s] - all_theme_book;
        scalar_data[7][s] = all_c.buy_rank_pct[s];
        scalar_data[8][s] = all_c.flow_rank_pct[s];
        scalar_data[9][s] = all_f.raw_amt[s] / (all_theme_total_amt + 1e-6);
        scalar_data[17][s] = cosine_similarity(all_ret_path, theme_ret_path);
        scalar_data[18][s] = l2_distance(all_ret_path, theme_ret_path);
        scalar_data[19][s] = mean(&ret_vs_theme[..4]) - mean(&ret_vs_theme[4..]);
        scalar_data[20][s] = {
            let dct = dct_first_k(all_ret_path, 6);
            let low: f64 = dct[..3].iter().map(|v| v.abs()).sum();
            let high: f64 = dct[3..].iter().map(|v| v.abs()).sum();
            low / (high + 1e-6)
        };
        scalar_data[21][s] = top5_dists[1] - top5_dists[0];
        scalar_data[22][s] = (top5_dists[1] - top5_dists[0]) / (top5_dists[0] + 1e-6);
        scalar_data[23][s] = probs_all
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| -p * p.ln())
            .sum();
        scalar_data[24][s] = all_c.mean_ret[all_label] - all_c.mean_ret[second_label];
        scalar_data[25][s] = all_c.mean_buy_ratio[all_label] - all_c.mean_buy_ratio[second_label];

        if prev_day < n_days && prev_day != target_day {
            let prev_close_c = &close_clusters[prev_day];
            let prev_close_f = &close_features[prev_day];
            let prev_label = prev_close_c.labels[s];
            if prev_label < prev_close_c.k {
                scalar_data[10][s] = prev_close_c.mean_buy_ratio[prev_label];
                scalar_data[11][s] = prev_close_c.std_ret[prev_label];
                scalar_data[12][s] = prev_close_c.breadth[prev_label];
                scalar_data[13][s] = prev_close_c.total_amt[prev_label];
                scalar_data[14][s] = prev_close_c.mean_flow_eff[prev_label];
                scalar_data[15][s] = prev_close_c.leader_gap[prev_label];
                scalar_data[16][s] = prev_close_c.skew_ret[prev_label];
                let _ = prev_close_f;
            }
        }

        if open_label < open_c.k && close_label < close_c.k {
            let open_ret_dev = open_f.ret[s] - open_c.mean_ret[open_label];
            let open_flow_dev = open_f.net_flow[s] - open_c.mean_net_flow[open_label];
            let open_buy_dev = open_f.buy_ratio[s] - open_c.mean_buy_ratio[open_label];
            let close_ret_dev = close_f.ret[s] - close_c.mean_ret[close_label];
            let close_flow_dev = close_f.net_flow[s] - close_c.mean_net_flow[close_label];
            let close_theme_ret = close_c.mean_ret[close_label];
            let open_dists = &open_c.dists_to_centers[s * open_c.k..(s + 1) * open_c.k];
            let close_dists = &close_c.dists_to_centers[s * close_c.k..(s + 1) * close_c.k];
            let (open_top5_dists, _, open_probs, _) = top5_from_dists(open_dists, open_c.k);
            let (close_top5_dists, _, close_probs, _) = top5_from_dists(close_dists, close_c.k);
            let open_margin = open_top5_dists[1] - open_top5_dists[0];
            let close_margin = close_top5_dists[1] - close_top5_dists[0];
            let open_entropy: f64 = open_probs
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            let close_entropy: f64 = close_probs
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            let open_theme_hot = open_c.theme_heat[open_label];
            let close_theme_hot = close_c.theme_heat[close_label];
            let open_flow_rank = open_c.flow_rank_pct[s];
            let close_rank = close_c.return_rank_pct[s];
            let close_flow_rank = close_c.flow_rank_pct[s];
            let open_path = path_slice(&open_f.ret_path, s, SHORT_BINS);
            let open_theme_path = cluster_path_slice(&open_c.ret_path_mean, open_label, SHORT_BINS);
            let close_path = path_slice(&close_f.ret_path, s, SHORT_BINS);
            let close_theme_path =
                cluster_path_slice(&close_c.ret_path_mean, close_label, SHORT_BINS);
            let open_cos = cosine_similarity(open_path, open_theme_path);
            let close_cos = cosine_similarity(close_path, close_theme_path);
            let expr_vec = [
                open_flow_dev,
                open_buy_dev,
                open_f.depth1[s] - open_c.mean_book_pressure[open_label],
                open_margin,
            ];
            let conf_vec = [close_ret_dev, close_theme_ret, close_margin, close_cos];
            scalar_data[26][s] = open_flow_dev * close_ret_dev;
            scalar_data[27][s] = open_buy_dev * close_theme_ret;
            scalar_data[28][s] = open_flow_dev * close_margin;
            scalar_data[29][s] = open_ret_dev * close_ret_dev;
            scalar_data[30][s] = open_flow_rank * close_rank;
            scalar_data[31][s] = if open_flow_dev > 0.0 && close_ret_dev < 0.0 {
                1.0
            } else {
                0.0
            };
            scalar_data[32][s] = open_ret_dev - close_ret_dev;
            scalar_data[33][s] = open_flow_dev - close_ret_dev;
            scalar_data[34][s] = open_theme_hot - close_theme_hot;
            scalar_data[35][s] = close_ret_dev - open_ret_dev;
            scalar_data[36][s] = close_flow_dev - open_flow_dev;
            scalar_data[37][s] = close_ret_dev * open_flow_dev.signum();
            scalar_data[38][s] = close_flow_rank - open_flow_rank;
            scalar_data[39][s] = close_margin - open_margin;
            scalar_data[40][s] = open_entropy - close_entropy;
            scalar_data[41][s] = if close_margin > open_margin && close_entropy < open_entropy {
                1.0
            } else {
                0.0
            };
            scalar_data[42][s] = close_theme_ret - open_c.mean_ret[open_label];
            scalar_data[43][s] = cosine_similarity(&expr_vec, &conf_vec);
            scalar_data[44][s] = l2_distance(&expr_vec, &conf_vec);
            let consistency = [
                (open_flow_dev > 0.0 && close_ret_dev > 0.0) as i32 as f64,
                (open_buy_dev > 0.0 && close_theme_ret > 0.0) as i32 as f64,
                (close_margin > open_margin) as i32 as f64,
                (close_cos > open_cos) as i32 as f64,
            ];
            scalar_data[45][s] = mean(&consistency);
            scalar_data[46][s] = 1.0 - (open_flow_rank - close_rank).abs();

            if prev_day < n_days && prev_day != target_day {
                let prev_close_c = &close_clusters[prev_day];
                let prev_label = prev_close_c.labels[s];
                if prev_label < prev_close_c.k {
                    scalar_data[47][s] = open_buy_dev * prev_close_c.mean_ret[prev_label];
                    scalar_data[48][s] = open_flow_dev * prev_close_c.mean_buy_ratio[prev_label];
                    scalar_data[49][s] = open_margin * prev_close_c.mean_ret[prev_label];
                    scalar_data[50][s] = open_entropy * prev_close_c.std_ret[prev_label];
                    scalar_data[51][s] = close_ret_dev * prev_close_c.mean_ret[prev_label];
                }
            }
        }
    }

    let mut vector_arrays = Vec::with_capacity(vector_names.len());
    for (data, &dim) in vector_data.into_iter().zip(vector_dims.iter()) {
        vector_arrays.push(array2_from_flat(py, n_stocks, dim, data)?);
    }
    let scalar_arrays = scalar_data
        .into_iter()
        .map(|v| PyArray1::from_vec(py, v).to_owned())
        .collect::<Vec<_>>();

    Ok((vector_arrays, scalar_arrays, vector_names, scalar_names))
}



/// 提取3维聚类散点数据，用于可视化。
/// 返回 (coords: n_stocks×3, labels: n_stocks, centers: k×3, n_clusters)
/// coords 列为 [ret, log_amt, net_flow]，已做 zscore 标准化
#[pyfunction(signature = (minute_data, k=30, n_threads=8))]
pub fn get_theme_cluster_scatter_3d(
    py: Python,
    minute_data: PyReadonlyArrayDyn<f64>,
    k: usize,
    n_threads: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<i64>>, Py<PyArray2<f64>>, usize)> {
    let data = minute_data.as_array();
    let shape = data.shape();
    if shape.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "minute_data必须是4维数组 (n_days, n_minutes, n_stocks, n_fields)",
        ));
    }
    let n_days = shape[0];
    let n_minutes = shape[1];
    let n_stocks = shape[2];
    let n_fields = shape[3];
    if n_fields < 34 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "minute_data字段数不足，至少需要34列",
        ));
    }
    if n_days == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "minute_data至少需要1天数据",
        ));
    }
    let actual_threads = n_threads.clamp(1, 10);
    let data_vec: Vec<f64> = data.iter().cloned().collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;

    // 只用最后一天
    let offset = (n_days - 1) * n_minutes * n_stocks * n_fields;
    let day_data = &data_vec[offset..offset + n_minutes * n_stocks * n_fields];
    let features = pool.install(|| {
        extract_segment_day(day_data, n_minutes, n_stocks, n_fields, 0, n_minutes, ALL_DAY_BINS, 3)
    });
    let cluster_result = cluster_segment_day(&features, k);

    // 构建坐标矩阵 (原始值: ret, log_amt, net_flow)
    let mut coords = vec![f64::NAN; n_stocks * 3];
    for s in 0..n_stocks {
        coords[s * 3] = features.ret[s];
        coords[s * 3 + 1] = features.log_amt[s];
        coords[s * 3 + 2] = features.net_flow[s];
    }

    // 标签 (无效标签用 -1)
    let labels: Vec<i64> = cluster_result
        .labels
        .iter()
        .map(|&l| if l == usize::MAX { -1 } else { l as i64 })
        .collect();

    // 聚类中心：对 center 在 zscore 空间做反变换得到原始空间的近似中心
    // 直接用每个聚类内股票的均值作为原始空间中心
    let actual_k = cluster_result.k;
    let mut centers_raw = vec![0.0f64; actual_k * 3];
    let mut counts = vec![0usize; actual_k];
    for s in 0..n_stocks {
        let l = cluster_result.labels[s];
        if l >= actual_k {
            continue;
        }
        for j in 0..3 {
            let v = coords[s * 3 + j];
            if v.is_finite() {
                centers_raw[l * 3 + j] += v;
            }
        }
        counts[l] += 1;
    }
    for c in 0..actual_k {
        if counts[c] > 0 {
            for j in 0..3 {
                centers_raw[c * 3 + j] /= counts[c] as f64;
            }
        }
    }

    let coords_arr = Array2::from_shape_vec((n_stocks, 3), coords)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("坐标数组形状错误: {}", e)))?;
    let centers_arr = Array2::from_shape_vec((actual_k, 3), centers_raw)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("中心数组形状错误: {}", e)))?;

    Ok((
        coords_arr.into_pyarray(py).to_owned(),
        PyArray1::from_vec(py, labels).to_owned(),
        centers_arr.into_pyarray(py).to_owned(),
        actual_k,
    ))
}
