//! Agent交互验证因子计算 — 核心逻辑
//!
//! 计算六类交互验证因子，将Agent的虚拟交易与真实市场前向数据做交叉验证。
//! 因子顺序与 Python _build_names() 严格一致。
pub mod factors_extra;
pub mod py_bindings;

use std::f64;

// ============================================================
// 常量
// ============================================================

pub const NS_PER_SEC: i64 = 1_000_000_000;
pub const NS_PER_MS: i64 = 1_000_000;

pub const FV_PER_HORIZON: usize = 13;
pub const MC_PER_AGENT: usize = 9;
pub const OB_PER_AGENT: usize = 11;
pub const PT_PER_HORIZON: usize = 4;
pub const AG_PER_AGENT: usize = 11;
pub const CA_PER_PAIR: usize = 4;

// ============================================================
// 因子索引计算
// ============================================================

#[inline]
pub fn per_agent_size(n_fwd_h: usize, n_pt_h: usize) -> usize {
    n_fwd_h * FV_PER_HORIZON + MC_PER_AGENT + OB_PER_AGENT + n_pt_h * PT_PER_HORIZON + AG_PER_AGENT
}

#[inline]
pub fn total_factor_count(n_agents: usize, n_fwd_h: usize, n_pt_h: usize) -> usize {
    let n_pairs = n_agents * (n_agents - 1) / 2;
    n_agents * per_agent_size(n_fwd_h, n_pt_h) + n_pairs * CA_PER_PAIR
}

#[inline]
fn fv_base(a: usize, n_fwd_h: usize, n_pt_h: usize) -> usize {
    a * per_agent_size(n_fwd_h, n_pt_h)
}

#[inline]
fn mc_base(a: usize, n_fwd_h: usize, n_pt_h: usize) -> usize {
    a * per_agent_size(n_fwd_h, n_pt_h) + n_fwd_h * FV_PER_HORIZON
}

#[inline]
fn ob_base(a: usize, n_fwd_h: usize, n_pt_h: usize) -> usize {
    a * per_agent_size(n_fwd_h, n_pt_h) + n_fwd_h * FV_PER_HORIZON + MC_PER_AGENT
}

#[inline]
fn pt_base(a: usize, n_fwd_h: usize, n_pt_h: usize) -> usize {
    a * per_agent_size(n_fwd_h, n_pt_h) + n_fwd_h * FV_PER_HORIZON + MC_PER_AGENT + OB_PER_AGENT
}

#[inline]
fn ag_base(a: usize, n_fwd_h: usize, n_pt_h: usize) -> usize {
    a * per_agent_size(n_fwd_h, n_pt_h)
        + n_fwd_h * FV_PER_HORIZON
        + MC_PER_AGENT
        + OB_PER_AGENT
        + n_pt_h * PT_PER_HORIZON
}

#[inline]
fn ca_base(n_agents: usize, n_fwd_h: usize, n_pt_h: usize) -> usize {
    n_agents * per_agent_size(n_fwd_h, n_pt_h)
}

#[inline]
fn pair_idx(a: usize, b: usize, n: usize) -> usize {
    // a < b
    a * (2 * n - a - 1) / 2 + (b - a - 1)
}

// ============================================================
// 工具函数
// ============================================================

#[inline]
fn dir_sign(d: i32) -> f64 {
    match d {
        66 => 1.0,
        83 => -1.0,
        _ => 0.0,
    }
}

#[inline]
fn safe_div(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-15 || !b.is_finite() {
        0.0
    } else {
        a / b
    }
}

#[inline]
fn cap(v: f64, limit: f64) -> f64 {
    if !v.is_finite() {
        return 0.0;
    }
    v.max(-limit).min(limit)
}

fn safe_mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let sum: f64 = x.iter().filter(|v| v.is_finite()).sum();
    let count = x.iter().filter(|v| v.is_finite()).count();
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn safe_std(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let valid: Vec<f64> = x.iter().filter(|v| v.is_finite()).copied().collect();
    if valid.len() < 2 {
        return 0.0;
    }
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

fn safe_skew(x: &[f64]) -> f64 {
    let valid: Vec<f64> = x.iter().filter(|v| v.is_finite()).copied().collect();
    if valid.len() < 3 {
        return 0.0;
    }
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-15 {
        return 0.0;
    }
    valid.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f64>() / n
}

/// 从 start_idx 开始，找第一个时间戳 >= ts[start_idx] + horizon_ns 的索引
fn find_future_idx(ts: &[i64], start_idx: usize, horizon_ns: i64) -> Option<usize> {
    let target = ts[start_idx] + horizon_ns;
    let later = &ts[start_idx..];
    match later.binary_search(&target) {
        Ok(i) => Some(start_idx + i),
        Err(i) => {
            if i < later.len() {
                Some(start_idx + i)
            } else {
                None
            }
        }
    }
}

// ============================================================
// 主计算函数
// ============================================================

pub fn compute_validation_factors(
    mkt_ts: &[i64],
    mkt_pr: &[f64],
    mkt_vo: &[f64],
    mkt_fl: &[i32],
    // Order book
    ob_ts: &[i64],
    ob_bid1: &[f64],
    ob_ask1: &[f64],
    ob_bid_vol1: &[f64],
    ob_ask_vol1: &[f64],
    // Order IDs (for state computation)
    bid_order_ids: &[i64],
    ask_order_ids: &[i64],
    // Agent data
    per_agent_idx: &[Vec<i64>],
    per_agent_dir: &[Vec<i32>],
    per_agent_vol: &[Vec<f64>],
    // Config
    fwd_horizons_sec: &[f64],
    pt_horizons_sec: &[f64],
) -> Vec<f64> {
    let n_agents = per_agent_idx.len();
    let n_fwd_h = fwd_horizons_sec.len();
    let n_pt_h = pt_horizons_sec.len();
    let n_mkt = mkt_ts.len();
    let n_ob = ob_ts.len();

    if n_mkt < 10 {
        return vec![0.0; total_factor_count(n_agents, n_fwd_h, n_pt_h)];
    }

    let n_total = total_factor_count(n_agents, n_fwd_h, n_pt_h);
    let mut result = vec![0.0f64; n_total];
    let pa_size = per_agent_size(n_fwd_h, n_pt_h);

    // --- 预计算 Agent 方向符号 ---
    let per_agent_sign: Vec<Vec<f64>> = per_agent_dir
        .iter()
        .map(|dirs| dirs.iter().map(|&d| dir_sign(d)).collect())
        .collect();

    // --- 盘口衍生数组 ---
    let ob_spread: Vec<f64> = ob_ask1.iter().zip(ob_bid1.iter()).map(|(a, b)| a - b).collect();
    let ob_imbalance: Vec<f64> = ob_bid_vol1
        .iter()
        .zip(ob_ask_vol1.iter())
        .map(|(b, a)| if b + a > 0.0 { (b - a) / (b + a) } else { 0.0 })
        .collect();
    let ob_depth: Vec<f64> = ob_bid_vol1
        .iter()
        .zip(ob_ask_vol1.iter())
        .map(|(b, a)| b + a)
        .collect();

    // 每个逐笔成交 → 最近盘口快照索引
    let ob_idx_for_trade: Vec<usize> = mkt_ts
        .iter()
        .map(|&t| match ob_ts.binary_search(&t) {
            Ok(i) => i,
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        })
        .collect();

    // ===========================================================
    // 前向价格预计算（两指针法 O(n_mkt × n_horizons)）
    // ===========================================================
    let mut fwd_prices: Vec<Vec<f64>> = vec![vec![f64::NAN; n_mkt]; n_fwd_h];
    let mut fwd_indices: Vec<Vec<i64>> = vec![vec![-1i64; n_mkt]; n_fwd_h];

    for (h_idx, &h_sec) in fwd_horizons_sec.iter().enumerate() {
        let h_ns = (h_sec * NS_PER_SEC as f64) as i64;
        let mut j: usize = 0;
        for i in 0..n_mkt {
            let target = mkt_ts[i].wrapping_add(h_ns);
            while j < n_mkt && mkt_ts[j] < target {
                j += 1;
            }
            if j < n_mkt {
                fwd_prices[h_idx][i] = mkt_pr[j];
                fwd_indices[h_idx][i] = j as i64;
            }
        }
    }

    // ===========================================================
    // PILLAR 1: 前向验证 (Forward Validation)
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let vols_a = &per_agent_vol[a];
        let n_trades = idxs_a.len();
        if n_trades < 2 {
            continue;
        }

        // 预取价格
        let entry_price: Vec<f64> = idxs_a.iter().map(|&i| mkt_pr[i as usize]).collect();

        for (h_idx, &_h_sec) in fwd_horizons_sec.iter().enumerate() {
            // 收集有效数据
            let mut aligned_ret: Vec<f64> = Vec::with_capacity(n_trades);
            let mut vol_v: Vec<f64> = Vec::with_capacity(n_trades);
            let mut mfe_arr: Vec<f64> = Vec::with_capacity(n_trades);
            let mut mae_arr: Vec<f64> = Vec::with_capacity(n_trades);

            for k in 0..n_trades {
                let i = idxs_a[k] as usize;
                let fp = fwd_prices[h_idx][i];
                if !fp.is_finite() || !entry_price[k].is_finite() || entry_price[k] <= 0.0 {
                    continue;
                }
                let raw_ret = (fp - entry_price[k]) / entry_price[k];
                let ar = raw_ret * sign_a[k];
                aligned_ret.push(ar);
                vol_v.push(vols_a[k]);

                // MFE/MAE
                let fwd_i = fwd_indices[h_idx][i];
                if fwd_i >= 0 && fwd_i as usize > i {
                    let end = fwd_i as usize;
                    let seg = &mkt_pr[i..=end];
                    let seg_max = seg.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let seg_min = seg.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let ep = entry_price[k];
                    if sign_a[k] > 0.0 {
                        mfe_arr.push(((seg_max - ep) / ep).max(0.0));
                        mae_arr.push(((ep - seg_min) / ep).max(0.0));
                    } else {
                        mfe_arr.push(((ep - seg_min) / ep).max(0.0));
                        mae_arr.push(((seg_max - ep) / ep).max(0.0));
                    }
                } else {
                    mfe_arr.push(0.0);
                    mae_arr.push(0.0);
                }
            }

            let n_valid = aligned_ret.len();
            if n_valid < 2 {
                continue;
            }

            let base = fv_base(a, n_fwd_h, n_pt_h) + h_idx * FV_PER_HORIZON;

            result[base] = safe_mean(&aligned_ret);
            result[base + 1] = safe_std(&aligned_ret);
            result[base + 2] = safe_skew(&aligned_ret);

            let wins: Vec<f64> = aligned_ret.iter().filter(|&&v| v > 0.0).copied().collect();
            let losses: Vec<f64> = aligned_ret.iter().filter(|&&v| v <= 0.0).copied().collect();
            let hitrate = if n_valid > 0 {
                wins.len() as f64 / n_valid as f64
            } else {
                0.0
            };
            result[base + 3] = hitrate;

            let win_mean = safe_mean(&wins);
            let loss_mean = safe_mean(&losses);
            let abs_loss = loss_mean.abs();
            result[base + 4] = win_mean;
            result[base + 5] = abs_loss;
            result[base + 6] = cap(safe_div(win_mean, abs_loss), 100.0);
            result[base + 7] = if !losses.is_empty() {
                cap(safe_div(wins.iter().sum(), losses.iter().map(|v| v.abs()).sum()), 100.0)
            } else {
                0.0
            };

            result[base + 8] = safe_mean(&mfe_arr);
            result[base + 9] = safe_mean(&mae_arr);
            result[base + 10] = cap(safe_div(safe_mean(&mfe_arr), safe_mean(&mae_arr)), 100.0);

            // VWAP return
            let vol_sum: f64 = vol_v.iter().sum();
            if vol_sum > 1e-15 {
                let vwap: f64 = aligned_ret
                    .iter()
                    .zip(vol_v.iter())
                    .map(|(r, v)| r * v)
                    .sum::<f64>()
                    / vol_sum;
                result[base + 11] = vwap;
            }

            // 收益自相关
            if n_valid >= 4 {
                let n = aligned_ret.len();
                let mean_all = safe_mean(&aligned_ret);
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for k in 0..n - 1 {
                    let d1 = aligned_ret[k] - mean_all;
                    let d2 = aligned_ret[k + 1] - mean_all;
                    num += d1 * d2;
                    den += d1 * d1;
                }
                let den2: f64 = (1..n).map(|k| {
                    let d = aligned_ret[k] - mean_all;
                    d * d
                }).sum();
                let den_prod = (den * den2).sqrt();
                if den_prod > 1e-15 {
                    result[base + 12] = num / den_prod;
                }
            }
        }
    }

    // ===========================================================
    // PILLAR 2: 市场确认 (Market Confirmation)
    // ===========================================================
    let window_ns = 2000 * NS_PER_MS;
    let half_ns = window_ns / 2;

    // 全局基线：每第N个采样
    let sample_step = (n_mkt / 1000).max(1);
    let mut glob_buy_ratios: Vec<f64> = Vec::with_capacity(1000);
    let mut glob_vol_densities: Vec<f64> = Vec::with_capacity(1000);
    for i in (0..n_mkt).step_by(sample_step).take(1000) {
        let t = mkt_ts[i];
        let lo = match mkt_ts.binary_search(&(t - half_ns)) {
            Ok(x) => x,
            Err(x) => x,
        };
        let hi = match mkt_ts.binary_search(&(t + half_ns)) {
            Ok(x) => x + 1,
            Err(x) => x,
        };
        if hi > lo {
            let n_real = hi - lo;
            let n_buy = mkt_fl[lo..hi].iter().filter(|&&f| f == 66).count();
            glob_buy_ratios.push(n_buy as f64 / n_real as f64);
            glob_vol_densities.push(mkt_vo[lo..hi].iter().sum::<f64>() / n_real as f64);
        }
    }
    let glob_buy_ratio = safe_mean(&glob_buy_ratios);
    let glob_vol_density = safe_mean(&glob_vol_densities);

    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 2 {
            continue;
        }

        let mut flow_align = vec![0.0f64; n_trades];
        let mut buy_ratios = vec![0.0f64; n_trades];
        let mut vol_densities = vec![0.0f64; n_trades];
        let mut real_vol_buy_ratios = vec![0.0f64; n_trades];

        for k in 0..n_trades {
            let i = idxs_a[k] as usize;
            let t = mkt_ts[i];
            let lo = match mkt_ts.binary_search(&(t - half_ns)) {
                Ok(x) => x,
                Err(x) => x,
            };
            let hi = match mkt_ts.binary_search(&(t + half_ns)) {
                Ok(x) => x + 1,
                Err(x) => x,
            };
            if hi <= lo {
                continue;
            }
            let n_real = hi - lo;
            let n_buy = mkt_fl[lo..hi].iter().filter(|&&f| f == 66).count();
            let buy_ratio = n_buy as f64 / n_real as f64;
            buy_ratios[k] = buy_ratio;
            vol_densities[k] = mkt_vo[lo..hi].iter().sum::<f64>() / n_real as f64;
            let buy_vol: f64 = mkt_fl[lo..hi]
                .iter()
                .zip(mkt_vo[lo..hi].iter())
                .filter(|(&f, _)| f == 66)
                .map(|(_, &v)| v)
                .sum();
            let sell_vol: f64 = mkt_fl[lo..hi]
                .iter()
                .zip(mkt_vo[lo..hi].iter())
                .filter(|(&f, _)| f == 83)
                .map(|(_, &v)| v)
                .sum();
            let total_vol = buy_vol + sell_vol;
            real_vol_buy_ratios[k] = if total_vol > 0.0 {
                buy_vol / total_vol
            } else {
                0.5
            };
            flow_align[k] = (buy_ratio - 0.5) * 2.0 * sign_a[k];
        }

        let buy_mask: Vec<bool> = sign_a.iter().map(|&s| s > 0.0).collect();
        let sell_mask: Vec<bool> = sign_a.iter().map(|&s| s < 0.0).collect();
        let buy_ratios_buy: Vec<f64> = buy_ratios
            .iter()
            .zip(buy_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();
        let buy_ratios_sell: Vec<f64> = buy_ratios
            .iter()
            .zip(sell_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();
        let vol_dens_buy: Vec<f64> = vol_densities
            .iter()
            .zip(buy_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();
        let vol_dens_sell: Vec<f64> = vol_densities
            .iter()
            .zip(sell_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();

        let base = mc_base(a, n_fwd_h, n_pt_h);
        result[base] = safe_mean(&flow_align);
        result[base + 1] = safe_std(&flow_align);
        result[base + 2] = safe_mean(&buy_ratios);
        result[base + 3] = safe_mean(&vol_densities) - glob_vol_density;
        result[base + 4] = safe_mean(&real_vol_buy_ratios);
        result[base + 5] = if !buy_ratios_buy.is_empty() {
            safe_mean(&buy_ratios_buy)
        } else {
            0.0
        };
        result[base + 6] = if !buy_ratios_sell.is_empty() {
            safe_mean(&buy_ratios_sell)
        } else {
            0.0
        };
        result[base + 7] = if !vol_dens_buy.is_empty() {
            safe_mean(&vol_dens_buy)
        } else {
            0.0
        };
        result[base + 8] = if !vol_dens_sell.is_empty() {
            safe_mean(&vol_dens_sell)
        } else {
            0.0
        };
    }

    // ===========================================================
    // PILLAR 3: 盘口交互 (Order Book)
    // ===========================================================
    let ob_sample_step = (n_ob / 1000).max(1);
    let mut ob_baseline_spreads: Vec<f64> = Vec::with_capacity(1000);
    let mut ob_baseline_depths: Vec<f64> = Vec::with_capacity(1000);
    for i in (0..n_ob).step_by(ob_sample_step).take(1000) {
        ob_baseline_spreads.push(ob_spread[i]);
        ob_baseline_depths.push(ob_depth[i]);
    }
    let baseline_spread = safe_mean(&ob_baseline_spreads);
    let baseline_depth = safe_mean(&ob_baseline_depths);

    for a in 0..n_agents {
        if n_ob == 0 {
            continue;
        }
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 2 {
            continue;
        }

        let ob_i: Vec<usize> = idxs_a.iter().map(|&i| ob_idx_for_trade[i as usize]).collect();
        let trade_spread: Vec<f64> = ob_i.iter().map(|&i| ob_spread[i]).collect();
        let trade_imbalance: Vec<f64> = ob_i.iter().map(|&i| ob_imbalance[i]).collect();
        let trade_depth: Vec<f64> = ob_i.iter().map(|&i| ob_depth[i]).collect();
        let buy_mask: Vec<bool> = sign_a.iter().map(|&s| s > 0.0).collect();
        let sell_mask: Vec<bool> = sign_a.iter().map(|&s| s < 0.0).collect();
        let ob_align: Vec<f64> = trade_imbalance
            .iter()
            .zip(sign_a.iter())
            .map(|(&imb, &s)| imb * s)
            .collect();

        let imb_buy: Vec<f64> = trade_imbalance
            .iter()
            .zip(buy_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();
        let imb_sell: Vec<f64> = trade_imbalance
            .iter()
            .zip(sell_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();
        let spread_buy: Vec<f64> = trade_spread
            .iter()
            .zip(buy_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();
        let spread_sell: Vec<f64> = trade_spread
            .iter()
            .zip(sell_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();

        let base = ob_base(a, n_fwd_h, n_pt_h);
        result[base] = safe_mean(&trade_spread);
        result[base + 1] = safe_mean(&trade_spread) - baseline_spread;
        result[base + 2] = safe_mean(&trade_imbalance);
        result[base + 3] = safe_mean(&trade_depth);
        result[base + 4] = safe_mean(&trade_depth) - baseline_depth;
        result[base + 5] = if !imb_buy.is_empty() {
            safe_mean(&imb_buy)
        } else {
            0.0
        };
        result[base + 6] = if !imb_sell.is_empty() {
            safe_mean(&imb_sell)
        } else {
            0.0
        };
        result[base + 7] = if !spread_buy.is_empty() {
            safe_mean(&spread_buy)
        } else {
            0.0
        };
        result[base + 8] = if !spread_sell.is_empty() {
            safe_mean(&spread_sell)
        } else {
            0.0
        };
        result[base + 9] = safe_mean(&ob_align);
        result[base + 10] = safe_std(&ob_align);
    }

    // ===========================================================
    // PILLAR 4: 交易后演化 (Post-Trade)
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 2 {
            continue;
        }

        for (h_idx, &h_sec) in pt_horizons_sec.iter().enumerate() {
            let h_ns = (h_sec * NS_PER_SEC as f64) as i64;
            let mut price_impact = vec![0.0f64; n_trades];
            let mut spread_change = vec![0.0f64; n_trades];
            let mut imb_change = vec![0.0f64; n_trades];
            let mut vol_ratio_arr = vec![0.0f64; n_trades];

            for k in 0..n_trades {
                let i = idxs_a[k] as usize;
                let t = mkt_ts[i];

                // 前向价格变化
                if let Some(j) = find_future_idx(mkt_ts, i, h_ns) {
                    if j > i {
                        price_impact[k] =
                            (mkt_pr[j] - mkt_pr[i]) / mkt_pr[i] * sign_a[k];
                    }
                }

                // 盘口变化
                let ob_i = ob_idx_for_trade[i];
                let ob_j = match ob_ts.binary_search(&(t.wrapping_add(h_ns))) {
                    Ok(x) => x,
                    Err(x) => x,
                };
                if ob_j < n_ob && ob_j > ob_i {
                    spread_change[k] = ob_spread[ob_j] - ob_spread[ob_i];
                    imb_change[k] = ob_imbalance[ob_j] - ob_imbalance[ob_i];
                }

                // 成交量比率
                let lo_pre = match mkt_ts.binary_search(&(t - h_ns)) {
                    Ok(x) => x,
                    Err(x) => x,
                };
                let vol_pre: f64 = mkt_vo[lo_pre..i].iter().sum();
                let hi_post = match mkt_ts.binary_search(&(t + h_ns)) {
                    Ok(x) => x + 1,
                    Err(x) => x,
                };
                let vol_post: f64 = mkt_vo[i + 1..hi_post].iter().sum();
                vol_ratio_arr[k] = safe_div(vol_post, vol_pre);
            }

            let base = pt_base(a, n_fwd_h, n_pt_h) + h_idx * PT_PER_HORIZON;
            result[base] = safe_mean(&price_impact);
            result[base + 1] = safe_mean(&spread_change);
            result[base + 2] = safe_mean(&imb_change);
            result[base + 3] = cap(safe_mean(&vol_ratio_arr), 100.0);
        }
    }

    // ===========================================================
    // PILLAR 5: 跨Agent交互 (Cross-Agent)
    // ===========================================================
    for a in 0..n_agents {
        for b in (a + 1)..n_agents {
            let idx_a = &per_agent_idx[a];
            let idx_b = &per_agent_idx[b];
            let sign_a = &per_agent_sign[a];
            let sign_b = &per_agent_sign[b];
            if idx_a.len() < 2 || idx_b.len() < 2 {
                continue;
            }

            let b_ts: Vec<i64> = idx_b.iter().map(|&i| mkt_ts[i as usize]).collect();
            let mut b_dir_at_a = vec![0.0f64; idx_a.len()];

            for k in 0..idx_a.len() {
                let t = mkt_ts[idx_a[k] as usize];
                let nearest = match b_ts.binary_search(&t) {
                    Ok(x) => x,
                    Err(x) => x,
                };
                if nearest > 0 && nearest < b_ts.len() {
                    let b_idx = if (b_ts[nearest - 1] - t).abs() < (b_ts[nearest] - t).abs() {
                        nearest - 1
                    } else {
                        nearest
                    };
                    if b_idx < sign_b.len() {
                        b_dir_at_a[k] = sign_b[b_idx];
                    }
                }
            }

            let agreement: Vec<bool> = sign_a
                .iter()
                .zip(b_dir_at_a.iter())
                .map(|(&sa, &sb)| (sa - sb).abs() < 1e-12 && sb != 0.0)
                .collect();
            let agree_rate = if !agreement.is_empty() {
                agreement.iter().filter(|&&x| x).count() as f64 / agreement.len() as f64
            } else {
                0.0
            };

            // 共识时前向1秒收益
            let agree_mask: Vec<bool> = agreement
                .iter()
                .zip(sign_a.iter())
                .map(|(&ag, &sa)| ag && sa != 0.0)
                .collect();
            let mut consensus_ret = 0.0f64;
            if agree_mask.iter().any(|&x| x) {
                let mut rets: Vec<f64> = Vec::new();
                for k in 0..idx_a.len() {
                    if agree_mask[k] {
                        let i = idx_a[k] as usize;
                        let fp = fwd_prices[3][i]; // h=1s → index 3 in FWD_HORIZONS_SEC
                        if fp.is_finite() {
                            let raw = (fp - mkt_pr[i]) / mkt_pr[i];
                            rets.push(raw * sign_a[k]);
                        }
                    }
                }
                if rets.len() >= 2 {
                    consensus_ret = safe_mean(&rets);
                }
            }

            // 分歧时
            let disagree_mask: Vec<bool> = agreement
                .iter()
                .zip(sign_a.iter())
                .zip(b_dir_at_a.iter())
                .map(|((&ag, &sa), &sb)| !ag && sa != 0.0 && sb != 0.0)
                .collect();
            let mut disagree_ret = 0.0f64;
            if disagree_mask.iter().any(|&x| x) {
                let mut rets: Vec<f64> = Vec::new();
                for k in 0..idx_a.len() {
                    if disagree_mask[k] {
                        let i = idx_a[k] as usize;
                        let fp = fwd_prices[3][i];
                        if fp.is_finite() {
                            let raw = (fp - mkt_pr[i]) / mkt_pr[i];
                            rets.push(raw * sign_a[k]);
                        }
                    }
                }
                if rets.len() >= 2 {
                    disagree_ret = safe_mean(&rets);
                }
            }

            let p_idx = pair_idx(a, b, n_agents);
            let base = ca_base(n_agents, n_fwd_h, n_pt_h) + p_idx * CA_PER_PAIR;
            result[base] = agree_rate;
            result[base + 1] = consensus_ret;
            result[base + 2] = disagree_ret;
            result[base + 3] = cap(consensus_ret - disagree_ret, 100.0);
        }
    }

    // ===========================================================
    // PILLAR 6: Agent 基础统计
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let dirs_a = &per_agent_dir[a];
        let vols_a = &per_agent_vol[a];

        let n = idxs_a.len();
        let n_buy = dirs_a.iter().filter(|&&d| d == 66).count();
        let n_sell = dirs_a.iter().filter(|&&d| d == 83).count();
        let buy_vol: f64 = dirs_a
            .iter()
            .zip(vols_a.iter())
            .filter(|(&d, _)| d == 66)
            .map(|(_, &v)| v)
            .sum();
        let sell_vol: f64 = dirs_a
            .iter()
            .zip(vols_a.iter())
            .filter(|(&d, _)| d == 83)
            .map(|(_, &v)| v)
            .sum();

        let (time_mean, time_std) = if !idxs_a.is_empty() {
            let t0 = mkt_ts[0];
            let trade_secs: Vec<f64> = idxs_a
                .iter()
                .map(|&i| (mkt_ts[i as usize] - t0) as f64 / NS_PER_SEC as f64)
                .collect();
            (safe_mean(&trade_secs), safe_std(&trade_secs))
        } else {
            (0.0, 0.0)
        };

        let base = ag_base(a, n_fwd_h, n_pt_h);
        result[base] = n as f64;
        result[base + 1] = n_buy as f64;
        result[base + 2] = n_sell as f64;
        result[base + 3] = safe_div(n_buy as f64, n as f64);
        result[base + 4] = buy_vol;
        result[base + 5] = sell_vol;
        // final_position: total_buy_vol - total_sell_vol (approximation)
        result[base + 6] = buy_vol - sell_vol;
        result[base + 7] = time_mean;
        result[base + 8] = time_std;
        result[base + 9] = safe_mean(vols_a);
        result[base + 10] = safe_std(vols_a);
    }

    // ============================================================
    // PILLARS 7-13: Extra factors (TQ, PC, SA, RI, ET, CD, LF)
    // ============================================================
    let extra = factors_extra::compute_extra_factors(
        mkt_ts,
        mkt_pr,
        mkt_vo,
        mkt_fl,
        &ob_spread,
        &ob_imbalance,
        &ob_depth,
        &ob_idx_for_trade,
        ob_bid_vol1,
        ob_ask_vol1,
        ob_bid1,
        ob_ask1,
        bid_order_ids,
        ask_order_ids,
        per_agent_idx,
        per_agent_dir,
        per_agent_vol,
        &per_agent_sign,
        &fwd_prices,
        &fwd_indices,
        fwd_horizons_sec,
    );
    result.extend(extra);

    result
}
