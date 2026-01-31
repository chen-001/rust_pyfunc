use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::sync::Arc;

const OUTPUT_COLS: usize = 15;
const LEVELS: usize = 10;

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64], mean_val: f64) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let var = values
        .iter()
        .map(|v| {
            let diff = v - mean_val;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn skewness(values: &[f64], mean_val: f64, std_val: f64) -> f64 {
    if values.len() < 3 || !std_val.is_finite() || std_val == 0.0 {
        return f64::NAN;
    }
    let m3 = values
        .iter()
        .map(|v| {
            let diff = v - mean_val;
            diff * diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    m3 / (std_val * std_val * std_val)
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return f64::NAN;
    }
    let mean_x = mean(x);
    let mean_y = mean(y);
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn autocorr_lag1(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let x = &values[..values.len() - 1];
    let y = &values[1..];
    correlation(x, y)
}

fn trend_corr(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mut idx = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        idx.push((i + 1) as f64);
    }
    correlation(values, &idx)
}

#[pyfunction]
#[pyo3(signature = (ticks_array, snaps_array))]
pub fn reconstruct_limit_order_lifecycle(
    py: Python,
    ticks_array: PyReadonlyArray2<f64>,
    snaps_array: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let ticks = ticks_array.as_array();
    let snaps = snaps_array.as_array();
    let (n_ticks, tick_cols) = ticks.dim();
    let (n_snaps, snap_cols) = snaps.dim();

    if tick_cols < 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ticks_array需要至少7列",
        ));
    }
    if snap_cols < 41 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "snaps_array需要至少41列 (exchtime + bid_prc1-10 + bid_vol1-10 + ask_prc1-10 + ask_vol1-10)",
        ));
    }

    let mut start_indices = vec![0usize; n_snaps];
    let mut max_bid_ids = vec![0i64; n_snaps];
    let mut max_ask_ids = vec![0i64; n_snaps];
    let mut tick_idx = 0usize;
    let mut current_max_bid = 0i64;
    let mut current_max_ask = 0i64;

    for s_idx in 0..n_snaps {
        let snap_ts = snaps[[s_idx, 0]];
        while tick_idx < n_ticks && ticks[[tick_idx, 0]] < snap_ts {
            let ask_id = ticks[[tick_idx, 5]] as i64;
            let bid_id = ticks[[tick_idx, 6]] as i64;
            if ask_id > current_max_ask {
                current_max_ask = ask_id;
            }
            if bid_id > current_max_bid {
                current_max_bid = bid_id;
            }
            tick_idx += 1;
        }

        let mut temp_idx = tick_idx;
        while temp_idx < n_ticks && ticks[[temp_idx, 0]] == snap_ts {
            let ask_id = ticks[[temp_idx, 5]] as i64;
            let bid_id = ticks[[temp_idx, 6]] as i64;
            if ask_id > current_max_ask {
                current_max_ask = ask_id;
            }
            if bid_id > current_max_bid {
                current_max_bid = bid_id;
            }
            temp_idx += 1;
        }

        start_indices[s_idx] = tick_idx;
        max_bid_ids[s_idx] = current_max_bid;
        max_ask_ids[s_idx] = current_max_ask;
        tick_idx = temp_idx;
    }

    let ticks_arc = Arc::new(ticks.to_owned());
    let snaps_arc = Arc::new(snaps.to_owned());
    let start_indices = Arc::new(start_indices);
    let max_bid_ids = Arc::new(max_bid_ids);
    let max_ask_ids = Arc::new(max_ask_ids);

    let total_rows = n_snaps * LEVELS * 2;
    let mut output = vec![f64::NAN; total_rows * OUTPUT_COLS];

    py.allow_threads(|| {
        output
            .chunks_mut(LEVELS * 2 * OUTPUT_COLS)
            .enumerate()
            .for_each(|(s_idx, chunk)| {
                let snap_ts = snaps_arc[[s_idx, 0]];
                let start_idx = start_indices[s_idx];
                let max_bid_id = max_bid_ids[s_idx];
                let max_ask_id = max_ask_ids[s_idx];

                for side in 0..2 {
                    for level in 0..LEVELS {
                        let row = side * LEVELS + level;
                        let base = row * OUTPUT_COLS;
                        chunk[base] = snap_ts;
                        chunk[base + 1] = side as f64;
                        chunk[base + 2] = (level + 1) as f64;

                        let price_col = if side == 0 {
                            1 + level
                        } else {
                            21 + level
                        };
                        let target_price = snaps_arc[[s_idx, price_col]];
                        let id_limit = if side == 0 { max_bid_id } else { max_ask_id };

                        let mut volumes: Vec<f64> = Vec::new();
                        let mut passive_ids: Vec<f64> = Vec::new();

                        let mut idx = start_idx;
                        while idx < n_ticks {
                            let tick_price = ticks_arc[[idx, 1]];
                            if side == 0 {
                                if tick_price < target_price {
                                    break;
                                }
                            } else if tick_price > target_price {
                                break;
                            }

                            let flag = ticks_arc[[idx, 4]] as i32;
                            let ask_id = ticks_arc[[idx, 5]] as i64;
                            let bid_id = ticks_arc[[idx, 6]] as i64;

                            if tick_price == target_price {
                                if side == 0 {
                                    if flag == 83 && bid_id <= id_limit {
                                        volumes.push(ticks_arc[[idx, 2]]);
                                        passive_ids.push(bid_id as f64);
                                    }
                                } else if flag == 66 && ask_id <= id_limit {
                                    volumes.push(ticks_arc[[idx, 2]]);
                                    passive_ids.push(ask_id as f64);
                                }
                            }

                            idx += 1;
                        }

                        if volumes.is_empty() {
                            chunk[base + 3] = 0.0;
                            chunk[base + 9] = 0.0;
                            continue;
                        }

                        let vol_sum = volumes.iter().sum::<f64>();
                        let vol_mean = mean(&volumes);
                        let vol_std = std_dev(&volumes, vol_mean);
                        let vol_skew = skewness(&volumes, vol_mean, vol_std);
                        let vol_autocorr = autocorr_lag1(&volumes);
                        let vol_trend = trend_corr(&volumes);

                        chunk[base + 3] = vol_sum;
                        chunk[base + 4] = vol_mean;
                        chunk[base + 5] = vol_std;
                        chunk[base + 6] = vol_skew;
                        chunk[base + 7] = vol_autocorr;
                        chunk[base + 8] = vol_trend;

                        let mut deltas: Vec<f64> = passive_ids
                            .iter()
                            .map(|id| (id_limit as f64) - id)
                            .collect();
                        let id_count = deltas.len() as f64;
                        let id_mean = mean(&deltas);
                        let id_std = std_dev(&deltas, id_mean);
                        let id_skew = skewness(&deltas, id_mean, id_std);
                        let id_trend = trend_corr(&passive_ids);

                        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let id_span = if deltas.len() >= 2 {
                            deltas[deltas.len() - 1] - deltas[0]
                        } else {
                            0.0
                        };

                        chunk[base + 9] = id_count;
                        chunk[base + 10] = id_span;
                        chunk[base + 11] = id_mean;
                        chunk[base + 12] = id_std;
                        chunk[base + 13] = id_skew;
                        chunk[base + 14] = id_trend;
                    }
                }
            });
    });

    let result = Array2::from_shape_vec((total_rows, OUTPUT_COLS), output).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("输出数组形状构建失败")
    })?;
    Ok(result.into_pyarray(py).to_owned())
}
