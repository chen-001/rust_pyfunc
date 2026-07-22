//! 两阶段分钟 CAPM 横截面因子。
//!
//! 1. 用过去 20 个交易日、截至当前分钟之前的数据，滚动估计每只股票的
//!    时间序列 CAPM 暴露；
//! 2. 每分钟用全市场股票的滚动 beta 解释当分钟收益，做横截面 CAPM 回归；
//! 3. pipeline 输出每只股票各分钟统计量的日内均值；指定分钟 Python 入口用于诊断。
//!
//! 分钟风险利率近似为 0；市场收益使用当分钟全市场有效股票的等权收益代理。

use crate::minute_data_reader::read_minute_field_multi_day;
use ndarray::Array3;
use pyo3::prelude::*;
use std::fs;
use std::io;

const DATA_DIR: &str = "/ssd_data/data/1min_factor_text";
const MINUTES_PER_DAY: usize = 240;
const LOOKBACK_DAYS: usize = 20;
const ROLLING_WINDOW: usize = LOOKBACK_DAYS * MINUTES_PER_DAY;
const MIN_HISTORY_OBS: u32 = 500;
const MAX_ABS_MINUTE_RETURN: f64 = 0.30;

const ROLLING_NAMES: [&str; 7] = [
    "rolling_beta",
    "rolling_alpha",
    "rolling_corr_market",
    "rolling_r2",
    "rolling_residual_std",
    "rolling_beta_se",
    "rolling_beta_t",
];

const STOCK_NAMES: [&str; 15] = [
    "capm_fitted_return",
    "capm_residual",
    "capm_residual_zscore",
    "capm_residual_rank",
    "capm_abs_residual",
    "capm_squared_residual",
    "capm_positive_residual",
    "capm_negative_residual",
    "capm_residual_contribution",
    "capm_leverage",
    "capm_studentized_residual",
    "capm_cooks_distance",
    "capm_dffits",
    "capm_expected_return_rank",
    "capm_beta_rank",
];

const MARKET_NAMES: [&str; 19] = [
    "capm_intercept",
    "capm_beta_premium",
    "capm_intercept_se_hc3",
    "capm_beta_premium_se_hc3",
    "capm_intercept_t_hc3",
    "capm_beta_premium_t_hc3",
    "capm_intercept_pvalue_normal",
    "capm_beta_premium_pvalue_normal",
    "capm_r2",
    "capm_adj_r2",
    "capm_residual_std",
    "capm_rmse",
    "capm_sse",
    "capm_f_stat",
    "capm_n_obs",
    "capm_beta_mean",
    "capm_beta_std",
    "capm_return_dispersion",
    "capm_market_return_equal_weight",
];

const N_ALL_METRICS: usize = ROLLING_NAMES.len() + STOCK_NAMES.len() + MARKET_NAMES.len();

/// 生产横截面因子：在 2025-12-18..2025-12-31 的 10 个交易日上，按逐股日均值
/// 做 Spearman 去重（|rho| >= 0.95），并排除同一分钟对所有股票相同的市场状态量。
const SELECTED_INDICES: [usize; 10] = [0, 1, 2, 4, 8, 9, 10, 11, 16, 18];
pub const N_FACTORS: usize = SELECTED_INDICES.len();

#[derive(Clone, Copy, Default)]
struct RollingSums {
    n: u32,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

#[derive(Clone, Copy)]
struct Exposure {
    beta: f64,
    alpha: f64,
    corr: f64,
    r2: f64,
    residual_std: f64,
    beta_se: f64,
    beta_t: f64,
}

impl Exposure {
    fn values(self) -> [f64; 7] {
        [
            self.beta,
            self.alpha,
            self.corr,
            self.r2,
            self.residual_std,
            self.beta_se,
            self.beta_t,
        ]
    }
}

struct CrossSectionFit {
    valid: Vec<usize>,
    fitted: Vec<f64>,
    residual: Vec<f64>,
    residual_rank: Vec<f64>,
    expected_rank: Vec<f64>,
    beta_rank: Vec<f64>,
    leverage: Vec<f64>,
    market: [f64; MARKET_NAMES.len()],
}

fn load_window_dates(date: i64) -> io::Result<Vec<i64>> {
    let path = format!("{DATA_DIR}/calendar_map.csv");
    let content = fs::read_to_string(&path)?;
    let dates: Vec<i64> = content
        .lines()
        .filter_map(|line| line.trim().parse::<i64>().ok())
        .collect();
    let pos = dates.binary_search(&date).map_err(|_| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("日期 {date} 不在分钟日历中"),
        )
    })?;
    if pos < LOOKBACK_DAYS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("日期 {date} 之前不足 {LOOKBACK_DAYS} 个交易日"),
        ));
    }
    Ok(dates[pos - LOOKBACK_DAYS..=pos].to_vec())
}

#[inline]
fn valid_return(v: f64) -> bool {
    v.is_finite() && v.abs() <= MAX_ABS_MINUTE_RETURN
}

/// 将 close 转为按 (time, stock) 连续排列的分钟收益，并同步生成等权市场收益。
fn build_returns(close: &Array3<f64>) -> (Vec<f32>, Vec<f64>) {
    let n_days = close.shape()[0];
    let n_rows = close.shape()[1];
    let n_stocks = close.shape()[2];
    let n_times = n_days * n_rows;
    let mut returns = vec![f32::NAN; n_times * n_stocks];
    let mut market = vec![f64::NAN; n_times];

    for d in 0..n_days {
        for minute in 1..n_rows {
            let t = d * n_rows + minute;
            let base = t * n_stocks;
            let mut sum = 0.0;
            let mut count = 0usize;
            for stock in 0..n_stocks {
                let prev = close[(d, minute - 1, stock)];
                let current = close[(d, minute, stock)];
                if prev.is_finite() && current.is_finite() && prev > 0.0 {
                    let ret = current / prev - 1.0;
                    if valid_return(ret) {
                        returns[base + stock] = ret as f32;
                        sum += ret;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                market[t] = sum / count as f64;
            }
        }
    }
    (returns, market)
}

#[inline]
fn add_pair(s: &mut RollingSums, x: f64, y: f64, sign: f64) {
    if !x.is_finite() || !valid_return(y) {
        return;
    }
    if sign > 0.0 {
        s.n += 1;
    } else {
        s.n = s.n.saturating_sub(1);
    }
    s.sx += sign * x;
    s.sy += sign * y;
    s.sxx += sign * x * x;
    s.syy += sign * y * y;
    s.sxy += sign * x * y;
}

fn exposure(s: RollingSums) -> Option<Exposure> {
    if s.n < MIN_HISTORY_OBS {
        return None;
    }
    let n = s.n as f64;
    let sxx = s.sxx - s.sx * s.sx / n;
    let syy = s.syy - s.sy * s.sy / n;
    let sxy = s.sxy - s.sx * s.sy / n;
    if sxx <= 1e-18 || syy <= 1e-18 {
        return None;
    }
    let beta = sxy / sxx;
    let alpha = s.sy / n - beta * s.sx / n;
    let corr = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let r2 = corr * corr;
    let sse = (syy - beta * sxy).max(0.0);
    let residual_std = (sse / (n - 2.0)).sqrt();
    let beta_se = residual_std / sxx.sqrt();
    let beta_t = if beta_se > 0.0 {
        beta / beta_se
    } else {
        f64::NAN
    };
    Some(Exposure {
        beta,
        alpha,
        corr,
        r2,
        residual_std,
        beta_se,
        beta_t,
    })
}

/// Abramowitz-Stegun 近似；p 值采用大样本正态近似，避免在热路径引入分布对象。
fn normal_two_sided_pvalue(t: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    let z = t.abs();
    let p = 0.231_641_9;
    let b1 = 0.319_381_530;
    let b2 = -0.356_563_782;
    let b3 = 1.781_477_937;
    let b4 = -1.821_255_978;
    let b5 = 1.330_274_429;
    let u = 1.0 / (1.0 + p * z);
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let upper = phi * (b1 * u + b2 * u.powi(2) + b3 * u.powi(3) + b4 * u.powi(4) + b5 * u.powi(5));
    (2.0 * upper).clamp(0.0, 1.0)
}

fn percentile_ranks(values: &[(usize, f64)], n_stocks: usize) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut out = vec![f64::NAN; n_stocks];
    if sorted.is_empty() {
        return out;
    }
    let denom = (sorted.len().saturating_sub(1)).max(1) as f64;
    let mut start = 0usize;
    while start < sorted.len() {
        let mut end = start + 1;
        while end < sorted.len() && sorted[end].1 == sorted[start].1 {
            end += 1;
        }
        let rank = ((start + end - 1) as f64 * 0.5) / denom;
        for &(idx, _) in &sorted[start..end] {
            out[idx] = rank;
        }
        start = end;
    }
    out
}

fn fit_cross_section(
    current_returns: &[f32],
    exposures: &[Option<Exposure>],
    market_return: f64,
) -> Option<CrossSectionFit> {
    let n_stocks = current_returns.len();
    let mut valid = Vec::with_capacity(n_stocks);
    let mut sx = 0.0;
    let mut sy = 0.0;
    for stock in 0..n_stocks {
        let y = current_returns[stock] as f64;
        if valid_return(y) {
            if let Some(e) = exposures[stock] {
                if e.beta.is_finite() {
                    valid.push(stock);
                    sx += e.beta;
                    sy += y;
                }
            }
        }
    }
    if valid.len() < 30 {
        return None;
    }

    let n = valid.len() as f64;
    let mean_x = sx / n;
    let mean_y = sy / n;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for &stock in &valid {
        let x = exposures[stock].unwrap().beta;
        let y = current_returns[stock] as f64;
        let dx = x - mean_x;
        let dy = y - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 1e-18 || syy <= 1e-18 {
        return None;
    }

    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let mut fitted = vec![f64::NAN; n_stocks];
    let mut residual = vec![f64::NAN; n_stocks];
    let mut leverage = vec![f64::NAN; n_stocks];
    let mut sse = 0.0;
    for &stock in &valid {
        let x = exposures[stock].unwrap().beta;
        let yhat = intercept + slope * x;
        let resid = current_returns[stock] as f64 - yhat;
        let h = 1.0 / n + (x - mean_x).powi(2) / sxx;
        fitted[stock] = yhat;
        residual[stock] = resid;
        leverage[stock] = h;
        sse += resid * resid;
    }

    let residual_std = (sse / (n - 2.0)).sqrt();
    let rmse = (sse / n).sqrt();
    let r2 = (1.0 - sse / syy).clamp(0.0, 1.0);
    let adj_r2 = 1.0 - (1.0 - r2) * (n - 1.0) / (n - 2.0);
    let f_stat = if 1.0 - r2 > 1e-18 {
        r2 / (1.0 - r2) * (n - 2.0)
    } else {
        f64::INFINITY
    };

    // HC3 sandwich covariance: (X'X)^-1 X' diag(e²/(1-h)²) X (X'X)^-1。
    let inv00 = 1.0 / n + mean_x * mean_x / sxx;
    let inv01 = -mean_x / sxx;
    let inv11 = 1.0 / sxx;
    let mut meat00 = 0.0;
    let mut meat01 = 0.0;
    let mut meat11 = 0.0;
    for &stock in &valid {
        let x = exposures[stock].unwrap().beta;
        let one_minus_h = (1.0 - leverage[stock]).max(1e-12);
        let w = residual[stock].powi(2) / one_minus_h.powi(2);
        meat00 += w;
        meat01 += w * x;
        meat11 += w * x * x;
    }
    let cov00 = inv00 * inv00 * meat00 + 2.0 * inv00 * inv01 * meat01 + inv01 * inv01 * meat11;
    let cov11 = inv01 * inv01 * meat00 + 2.0 * inv01 * inv11 * meat01 + inv11 * inv11 * meat11;
    let intercept_se = cov00.max(0.0).sqrt();
    let slope_se = cov11.max(0.0).sqrt();
    let intercept_t = if intercept_se > 0.0 {
        intercept / intercept_se
    } else {
        f64::NAN
    };
    let slope_t = if slope_se > 0.0 {
        slope / slope_se
    } else {
        f64::NAN
    };

    let residual_values: Vec<(usize, f64)> = valid.iter().map(|&i| (i, residual[i])).collect();
    let expected_values: Vec<(usize, f64)> = valid.iter().map(|&i| (i, fitted[i])).collect();
    let beta_values: Vec<(usize, f64)> = valid
        .iter()
        .map(|&i| (i, exposures[i].unwrap().beta))
        .collect();

    let market = [
        intercept,
        slope,
        intercept_se,
        slope_se,
        intercept_t,
        slope_t,
        normal_two_sided_pvalue(intercept_t),
        normal_two_sided_pvalue(slope_t),
        r2,
        adj_r2,
        residual_std,
        rmse,
        sse,
        f_stat,
        n,
        mean_x,
        (sxx / (n - 1.0)).sqrt(),
        (syy / (n - 1.0)).sqrt(),
        market_return,
    ];

    Some(CrossSectionFit {
        valid,
        fitted,
        residual,
        residual_rank: percentile_ranks(&residual_values, n_stocks),
        expected_rank: percentile_ranks(&expected_values, n_stocks),
        beta_rank: percentile_ranks(&beta_values, n_stocks),
        leverage,
        market,
    })
}

#[inline]
fn accumulate(sum: &mut [f64], count: &mut [u16], base: usize, offset: usize, value: f64) {
    if value.is_finite() {
        sum[base + offset] += value;
        count[base + offset] += 1;
    }
}

fn compute_internal(
    date: i64,
    capture_minute: Option<usize>,
) -> io::Result<(Vec<String>, Vec<f32>, Option<Vec<f32>>)> {
    if let Some(minute) = capture_minute {
        if minute >= MINUTES_PER_DAY {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("minute_index 必须小于 {MINUTES_PER_DAY}"),
            ));
        }
    }

    let dates = load_window_dates(date)?;
    let (codes, close) = read_minute_field_multi_day("close", &dates)?;
    let n_stocks = close.shape()[2];
    let n_times = close.shape()[0] * close.shape()[1];
    let target_start = (dates.len() - 1) * MINUTES_PER_DAY;
    let (returns, market_returns) = build_returns(&close);
    drop(close);

    let mut rolling = vec![RollingSums::default(); n_stocks];
    let mut exposures = vec![None; n_stocks];
    let mut sums = vec![0.0f64; n_stocks * N_ALL_METRICS];
    let mut counts = vec![0u16; n_stocks * N_ALL_METRICS];
    let mut captured = capture_minute.map(|_| vec![f32::NAN; n_stocks * N_ALL_METRICS]);

    for t in 0..n_times {
        if t >= ROLLING_WINDOW {
            let old_t = t - ROLLING_WINDOW;
            let old_x = market_returns[old_t];
            let old_base = old_t * n_stocks;
            for stock in 0..n_stocks {
                add_pair(
                    &mut rolling[stock],
                    old_x,
                    returns[old_base + stock] as f64,
                    -1.0,
                );
            }
        }

        if t >= target_start {
            for stock in 0..n_stocks {
                exposures[stock] = exposure(rolling[stock]);
            }
            let minute = t - target_start;
            let base = t * n_stocks;
            let current = &returns[base..base + n_stocks];
            if let Some(fit) = fit_cross_section(current, &exposures, market_returns[t]) {
                let capture_this = capture_minute == Some(minute);
                let residual_std = fit.market[10];
                let sse = fit.market[12];
                for &stock in &fit.valid {
                    let out_base = stock * N_ALL_METRICS;
                    let exp = exposures[stock].unwrap();
                    let rolling_values = exp.values();
                    let resid = fit.residual[stock];
                    let fitted = fit.fitted[stock];
                    let h = fit.leverage[stock];
                    let one_minus_h = (1.0 - h).max(1e-12);
                    let studentized = if residual_std > 0.0 {
                        resid / (residual_std * one_minus_h.sqrt())
                    } else {
                        f64::NAN
                    };
                    let stock_values = [
                        fitted,
                        resid,
                        if residual_std > 0.0 {
                            resid / residual_std
                        } else {
                            f64::NAN
                        },
                        fit.residual_rank[stock],
                        resid.abs(),
                        resid * resid,
                        resid.max(0.0),
                        resid.min(0.0),
                        if sse > 0.0 {
                            resid * resid / sse
                        } else {
                            f64::NAN
                        },
                        h,
                        studentized,
                        studentized * studentized * h / (2.0 * one_minus_h),
                        studentized * (h / one_minus_h).sqrt(),
                        fit.expected_rank[stock],
                        fit.beta_rank[stock],
                    ];

                    for (k, &value) in rolling_values.iter().enumerate() {
                        accumulate(&mut sums, &mut counts, out_base, k, value);
                    }
                    for (k, &value) in stock_values.iter().enumerate() {
                        accumulate(
                            &mut sums,
                            &mut counts,
                            out_base,
                            ROLLING_NAMES.len() + k,
                            value,
                        );
                    }
                    for (k, &value) in fit.market.iter().enumerate() {
                        accumulate(
                            &mut sums,
                            &mut counts,
                            out_base,
                            ROLLING_NAMES.len() + STOCK_NAMES.len() + k,
                            value,
                        );
                    }

                    if capture_this {
                        if let Some(ref mut raw) = captured {
                            for (k, &value) in rolling_values.iter().enumerate() {
                                raw[out_base + k] = value as f32;
                            }
                            for (k, &value) in stock_values.iter().enumerate() {
                                raw[out_base + ROLLING_NAMES.len() + k] = value as f32;
                            }
                            for (k, &value) in fit.market.iter().enumerate() {
                                raw[out_base + ROLLING_NAMES.len() + STOCK_NAMES.len() + k] =
                                    value as f32;
                            }
                        }
                    }
                }
            }
        }

        let x = market_returns[t];
        let base = t * n_stocks;
        for stock in 0..n_stocks {
            add_pair(&mut rolling[stock], x, returns[base + stock] as f64, 1.0);
        }
    }

    let mut out_codes = Vec::new();
    let mut out_vals = Vec::new();
    let mut out_raw = capture_minute.map(|_| Vec::new());
    for stock in 0..n_stocks {
        let base = stock * N_ALL_METRICS;
        if counts[base] == 0 || codes[stock].is_empty() {
            continue;
        }
        out_codes.push(codes[stock].clone());
        for k in 0..N_ALL_METRICS {
            let n = counts[base + k];
            out_vals.push(if n > 0 {
                (sums[base + k] / n as f64) as f32
            } else {
                f32::NAN
            });
        }
        if let (Some(raw), Some(filtered)) = (&captured, &mut out_raw) {
            filtered.extend_from_slice(&raw[base..base + N_ALL_METRICS]);
        }
    }

    Ok((out_codes, out_vals, out_raw))
}

/// 完整研究结果：返回每只股票 41 项分钟统计量的日内均值。
pub fn compute_minute_capm_all_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    let (codes, vals, _) = compute_internal(date, None)?;
    Ok((codes, vals))
}

/// pipeline 核心：返回经真实横截面相关性去重后的 10 项代表性因子。
pub fn compute_minute_capm_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    let (codes, all_vals) = compute_minute_capm_all_full(date)?;
    let mut selected = Vec::with_capacity(codes.len() * N_FACTORS);
    for row in all_vals.chunks_exact(N_ALL_METRICS) {
        selected.extend(SELECTED_INDICES.iter().map(|&idx| row[idx]));
    }
    Ok((codes, selected))
}

pub fn minute_capm_names() -> Vec<String> {
    let all = minute_capm_all_names();
    SELECTED_INDICES
        .iter()
        .map(|&idx| all[idx].clone())
        .collect()
}

pub fn minute_capm_all_names() -> Vec<String> {
    ROLLING_NAMES
        .iter()
        .chain(STOCK_NAMES.iter())
        .chain(MARKET_NAMES.iter())
        .map(|name| format!("minute_{name}_mean"))
        .collect()
}

pub fn minute_capm_at_names() -> Vec<String> {
    ROLLING_NAMES
        .iter()
        .chain(STOCK_NAMES.iter())
        .chain(MARKET_NAMES.iter())
        .map(|name| (*name).to_string())
        .collect()
}

#[pyfunction]
pub fn py_minute_capm(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_minute_capm_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn py_minute_capm_all(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_minute_capm_all_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// 返回指定分钟的逐股原始统计量；minute_index 为 0..239。
#[pyfunction]
pub fn py_minute_capm_at(date: i64, minute_index: usize) -> PyResult<(Vec<String>, Vec<f32>)> {
    let (codes, _, raw) = compute_internal(date, Some(minute_index))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok((codes, raw.unwrap_or_default()))
}

#[pyfunction]
pub fn py_minute_capm_names() -> Vec<String> {
    minute_capm_names()
}

#[pyfunction]
pub fn py_minute_capm_all_names() -> Vec<String> {
    minute_capm_all_names()
}

#[pyfunction]
pub fn py_minute_capm_at_names() -> Vec<String> {
    minute_capm_at_names()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_exposure_recovers_linear_model() {
        let mut sums = RollingSums::default();
        for i in 0..1000 {
            let x = (i as f64 - 500.0) / 100_000.0;
            let y = 0.000_02 + 1.5 * x;
            add_pair(&mut sums, x, y, 1.0);
        }
        let got = exposure(sums).unwrap();
        assert!((got.beta - 1.5).abs() < 1e-10);
        assert!((got.alpha - 0.000_02).abs() < 1e-10);
        assert!((got.r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cross_section_recovers_slope_and_identity_relations() {
        let n = 100usize;
        let mut rets = Vec::with_capacity(n);
        let mut exposures = Vec::with_capacity(n);
        for i in 0..n {
            let beta = 0.5 + i as f64 / 100.0;
            let noise = ((i % 7) as f64 - 3.0) * 1e-5;
            rets.push((0.000_1 + 0.002 * beta + noise) as f32);
            exposures.push(Some(Exposure {
                beta,
                alpha: 0.0,
                corr: 0.0,
                r2: 0.0,
                residual_std: 0.0,
                beta_se: 0.0,
                beta_t: 0.0,
            }));
        }
        let fit = fit_cross_section(&rets, &exposures, 0.001).unwrap();
        assert!((fit.market[1] - 0.002).abs() < 1e-5);
        for i in fit.valid {
            assert!((fit.fitted[i] + fit.residual[i] - rets[i] as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn names_match_factor_count() {
        assert_eq!(N_ALL_METRICS, 41);
        assert_eq!(N_FACTORS, 10);
        assert_eq!(minute_capm_names().len(), N_FACTORS);
        assert_eq!(minute_capm_all_names().len(), N_ALL_METRICS);
        assert_eq!(minute_capm_at_names().len(), N_ALL_METRICS);
    }
}
