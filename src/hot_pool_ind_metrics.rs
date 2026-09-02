//! 同热点股票池「行业维度拓展」——4 组补充因子（Multica COR-36 评论 396a1460 设计执行）。
//!
//! 组 1 hotpool_ext_ind_rel_*   (840): 全市场识别后, 每股每次入选记 4 个原始量
//!      (buy/vol/ba/z), 收盘后按"同行业当日全部入选记录"回填行业截面排名百分位与
//!      z 标准化 + 频率比(截至第 k 条累计次数/行业人均), 走 21 统计量降维。
//! 组 2 hotpool_ext_ind_cooc_*  (720): 初版全市场共现 top-10 按同行/跨行拆分聚合
//!      (11 特征 × mean/std × 4 口径 + 2 结构量, 热/冷各一遍)。
//! 组 3 hotpool_ext_ind_heat_*  ( 24): 行业层热度聚合映射到每股
//!      (sum/nratio/share/z/own/spread)。
//! 组 4 hotpool_ext_indpool_*  (7072): 识别层行业内化——组=当秒同行业入选者,
//!      40 维组特征全部在行业内桶上计算 (840×8 降维) + 同行共现 (352)。
//!
//! 设计报告: research_ext_ideas/design_report_hotpool_ind.md
//! sandbox 验证: sandbox_hotpool_ind/（逻辑 + 速度基准均达标）

use crate::fast_csv_reader::read_trade_fast_inner;
use crate::features::get_features_factors_rust_full;
use crate::hot_stock_pool_metrics::{
    build_group_features_arr, build_stock_data, corr_fast, cst_midnight_epoch, fill_per_stock_arr,
    herfindahl, kurtosis, list_codes, mean, mean_std, neighbor_mean_inline, percentile_sorted,
    rank_pct_in, sec_to_idx, skew, std, top_k_concentration, RollingCache, SecStat, StockData,
    ADJUSTED_SECONDS, BASIC_FEAT_N, FEAT_PER_INCLUSION, N_PARAM_COMBOS, PARAM_CONFIGS,
    SECOND_STEP,
};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// 申万一级行业静态映射（每股最后有效行业归属，6 位码 // 10000 → 1..31）。
const IND_CSV_PATH: &str = "/hdd/user_home_unsafe/chenzongwei/sw_ind_l1.csv";

const HIST_WIN: usize = 120;
const Z_THRESH: f64 = 1.5;

const COMBO_LABELS: [&str; 4] = ["x60y3_buy", "x60y3_ba", "x15y10_buy", "x15y10_ba"];

// 因子块大小
const G1_PER: usize = 5 * 21; // 5 特征 × 21 统计
const G2_PER: usize = BASIC_FEAT_N * 2 * 4 + 2; // 11×2×4 + 2 结构量 = 90
const G3_PER: usize = 6;
const G4_REDUCED_PER: usize = FEAT_PER_INCLUSION * 21; // 40×21 = 840
const G4_COOC_PER: usize = BASIC_FEAT_N * 2 * 4; // 88
pub const N_FACTORS: usize = 4 * 2 * G1_PER
    + 4 * 2 * G2_PER
    + 4 * G3_PER
    + 4 * 2 * G4_REDUCED_PER
    + 4 * G4_COOC_PER; // 840+720+24+6720+352 = 8656

fn load_industry() -> std::collections::HashMap<String, u8> {
    let mut m = std::collections::HashMap::new();
    if let Ok(content) = std::fs::read_to_string(IND_CSV_PATH) {
        for line in content.lines().skip(1) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((code, ind)) = line.split_once(',') {
                if let Ok(v) = ind.trim().parse::<i32>() {
                    m.insert(code.trim().to_string(), v.clamp(0, 31) as u8);
                }
            }
        }
    }
    m
}

type Feat40 = [f32; FEAT_PER_INCLUSION];
type Rel5 = [f32; 5]; // [rk_buy, rk_vol, rk_ba, z_norm, freq]

struct PiOut {
    rel_hot: Vec<Vec<[f32; 4]>>,
    rel_cold: Vec<Vec<[f32; 4]>>,
    pool_log_hot: Vec<Vec<usize>>,
    pool_log_cold: Vec<Vec<usize>>,
    hot_cnt: Vec<u32>,
    cold_cnt: Vec<u32>,
    hot_sum_ind: [u32; 32],
    cold_sum_ind: [u32; 32],
    hot_nstock_ind: [u32; 32],
    cold_nstock_ind: [u32; 32],
    mkt_hot_total: u32,
    mkt_cold_total: u32,
    indpool_hot: Vec<Vec<Feat40>>,
    indpool_cold: Vec<Vec<Feat40>>,
    ind_pool_log_hot: [Vec<Vec<usize>>; 32],
    ind_pool_log_cold: [Vec<Vec<usize>>; 32],
}

#[allow(clippy::too_many_arguments)]
fn compute_pi(
    x: usize,
    d_type: usize,
    min_trades: u32,
    caches: &[RollingCache],
    stocks: &[StockData],
) -> PiOut {
    let n_valid = stocks.len();
    let d_field: u8 = if d_type == 0 { 0 } else { 1 };

    let mut rel_hot: Vec<Vec<[f32; 4]>> = vec![Vec::new(); n_valid];
    let mut rel_cold: Vec<Vec<[f32; 4]>> = vec![Vec::new(); n_valid];
    let mut pool_log_hot: Vec<Vec<usize>> = Vec::new();
    let mut pool_log_cold: Vec<Vec<usize>> = Vec::new();
    let mut hot_cnt = vec![0u32; n_valid];
    let mut cold_cnt = vec![0u32; n_valid];
    let mut hot_sum_ind = [0u32; 32];
    let mut cold_sum_ind = [0u32; 32];
    let mut hot_nstock_ind = [0u32; 32];
    let mut cold_nstock_ind = [0u32; 32];
    let mut mkt_hot_total: u32 = 0;
    let mut mkt_cold_total: u32 = 0;
    let mut indpool_hot: Vec<Vec<Feat40>> = vec![Vec::new(); n_valid];
    let mut indpool_cold: Vec<Vec<Feat40>> = vec![Vec::new(); n_valid];
    let mut ind_pool_log_hot: [Vec<Vec<usize>>; 32] = std::array::from_fn(|_| Vec::new());
    let mut ind_pool_log_cold: [Vec<Vec<usize>>; 32] = std::array::from_fn(|_| Vec::new());

    // 组 4 状态：per gt per 行业
    let mut prev_mean_br = [[f32::NAN; 32]; 2];
    let mut prev2_mean_br = [[f32::NAN; 32]; 2];
    let mut prev_mean_ba = [[f32::NAN; 32]; 2];
    let mut prev2_mean_ba = [[f32::NAN; 32]; 2];
    let mut prev_sec = [[usize::MAX; 32]; 2];
    let mut prev_bucket: [[Vec<usize>; 32]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    let mut stay_seconds = vec![[0u32; 2]; n_valid];
    let mut stay_seen = vec![[false; 2]; n_valid];

    let mut d_hist: Vec<std::collections::VecDeque<f32>> = (0..n_valid)
        .map(|_| std::collections::VecDeque::with_capacity(HIST_WIN / SECOND_STEP + 1))
        .collect();

    let mut buf_all_vals_d = vec![f32::NAN; n_valid];
    let mut buf_all_ba_vals = vec![f32::NAN; n_valid];
    let mut ind_buf: [Vec<usize>; 32] = std::array::from_fn(|_| Vec::new());

    for sec in (15..ADJUSTED_SECONDS).step_by(SECOND_STEP) {
        if sec < x - 1 {
            continue;
        }
        for (si, cache) in caches.iter().enumerate() {
            let dv = cache.get_by_x(x, d_field, sec);
            let bv = cache.get_by_x(x, 1, sec);
            buf_all_vals_d[si] = dv;
            buf_all_ba_vals[si] = bv;
            if dv.is_finite() {
                let h = &mut d_hist[si];
                h.push_back(dv);
                while h.len() > HIST_WIN / SECOND_STEP {
                    h.pop_front();
                }
            }
        }
        let mut top_pairs: Vec<(usize, f32, f32)> = Vec::new(); // (stock, D, z)
        let mut bot_pairs: Vec<(usize, f32, f32)> = Vec::new();
        for si in 0..n_valid {
            let dv = buf_all_vals_d[si];
            if !dv.is_finite() {
                continue;
            }
            let trades = caches[si].get_by_x(x, 4, sec);
            if trades < min_trades as f32 {
                continue;
            }
            let h = &d_hist[si];
            if h.len() < 5 {
                continue;
            }
            let hmean: f64 = h.iter().map(|v| *v as f64).sum::<f64>() / h.len() as f64;
            let hvar: f64 =
                h.iter().map(|v| (*v as f64 - hmean).powi(2)).sum::<f64>() / h.len() as f64;
            let hstd = hvar.sqrt();
            if hstd < 1e-8 {
                continue;
            }
            let zscore = (dv as f64 - hmean) / hstd;
            if zscore > Z_THRESH {
                top_pairs.push((si, dv, zscore as f32));
            } else if zscore < -Z_THRESH {
                bot_pairs.push((si, dv, zscore as f32));
            }
        }
        if top_pairs.is_empty() && bot_pairs.is_empty() {
            continue;
        }
        top_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        bot_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mkt_total_vol: f32 = (0..n_valid).map(|i| caches[i].get_by_x(x, 3, sec)).sum();
        let mkt_mean_d = {
            let vf: Vec<f32> = buf_all_vals_d.iter().copied().filter(|v| v.is_finite()).collect();
            mean(&vf)
        };
        let mkt_mean_ba = {
            let vf: Vec<f32> = buf_all_ba_vals
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            mean(&vf)
        };

        // ---- 全市场组（初版口径）：只为组 1 记录 rel + 组 2 pool_log + 组 3 计数 ----
        for (gt, pairs) in [0usize, 1usize].iter().zip([&top_pairs, &bot_pairs].iter()) {
            let gt = *gt;
            let pool_idx: Vec<usize> = pairs.iter().map(|(i, _, _)| *i).collect();
            if pool_idx.is_empty() {
                continue;
            }
            if gt == 0 {
                pool_log_hot.push(pool_idx.clone());
            } else {
                pool_log_cold.push(pool_idx.clone());
            }
            for &(si, _dv, z) in pairs.iter() {
                if stocks[si].ind < 1 {
                    continue;
                }
                let buy = caches[si].get_by_x(x, 0, sec);
                let vol = caches[si].get_by_x(x, 3, sec);
                let ba = caches[si].get_by_x(x, 1, sec);
                let rec = [buy, vol, ba, z];
                if gt == 0 {
                    rel_hot[si].push(rec);
                } else {
                    rel_cold[si].push(rec);
                }
            }
            for &si in pool_idx.iter() {
                if gt == 0 {
                    hot_cnt[si] += 1;
                } else {
                    cold_cnt[si] += 1;
                }
            }
        }

        // ---- 组 4：行业内桶 ----
        for (gt, pairs) in [0usize, 1usize].iter().zip([&top_pairs, &bot_pairs].iter()) {
            let gt = *gt;
            let pool_idx: Vec<usize> = pairs.iter().map(|(i, _, _)| *i).collect();
            if pool_idx.is_empty() {
                continue;
            }
            for b in ind_buf.iter_mut() {
                b.clear();
            }
            for &si in pool_idx.iter() {
                let ind = stocks[si].ind as usize;
                if ind >= 1 && ind <= 31 {
                    ind_buf[ind].push(si);
                }
            }
            for ind in 1..=31usize {
                let bucket = &ind_buf[ind];
                if bucket.is_empty() {
                    continue;
                }
                let n_b = bucket.len();
                let sto_buy: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 0, sec)).collect();
                let sto_ba: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 1, sec)).collect();
                let sto_ret: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 2, sec)).collect();
                let sto_vol: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 3, sec)).collect();
                let br_finite: Vec<f32> = sto_buy.iter().copied().filter(|v| v.is_finite()).collect();
                let ba_finite: Vec<f32> = sto_ba.iter().copied().filter(|v| v.is_finite()).collect();
                let ret_finite: Vec<f32> = sto_ret.iter().copied().filter(|v| v.is_finite()).collect();
                let vol_finite: Vec<f32> = sto_vol
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .collect();
                let mean_br = mean(&br_finite);
                let mean_ba = mean(&ba_finite);
                let pool_total_vol: f32 = sto_vol.iter().sum();
                let c04_val = if mkt_total_vol > 0.0 {
                    pool_total_vol / mkt_total_vol
                } else {
                    f32::NAN
                };
                let mut br_sorted: Vec<f32> = br_finite.clone();
                br_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mut ba_sorted: Vec<f32> = ba_finite.clone();
                ba_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let vol_sorted: Vec<f32> = {
                    let mut v = vol_finite.clone();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    v
                };
                let (br_m, br_s) = mean_std(&br_sorted);
                let (ba_m, ba_s) = mean_std(&ba_sorted);
                let mut group_feats: [f32; 34] = build_group_features_arr(
                    &br_finite, &ba_finite, &ret_finite, &vol_finite, &sto_buy, &sto_ba,
                    &sto_ret, &br_sorted, &ba_sorted, mean_br, mean_ba, mkt_mean_d, mkt_mean_ba,
                );
                // 桶级差分（A06/A07/B06/B07）
                group_feats[5] = if prev_mean_br[gt][ind].is_finite() && mean_br.is_finite() {
                    mean_br - prev_mean_br[gt][ind]
                } else {
                    f32::NAN
                };
                group_feats[6] =
                    if prev_mean_br[gt][ind].is_finite()
                        && prev2_mean_br[gt][ind].is_finite()
                        && mean_br.is_finite()
                    {
                        (mean_br - prev_mean_br[gt][ind])
                            - (prev_mean_br[gt][ind] - prev2_mean_br[gt][ind])
                    } else {
                        f32::NAN
                    };
                group_feats[18] = if prev_mean_ba[gt][ind].is_finite() && mean_ba.is_finite() {
                    mean_ba - prev_mean_ba[gt][ind]
                } else {
                    f32::NAN
                };
                group_feats[19] =
                    if prev_mean_ba[gt][ind].is_finite()
                        && prev2_mean_ba[gt][ind].is_finite()
                        && mean_ba.is_finite()
                    {
                        (mean_ba - prev_mean_ba[gt][ind])
                            - (prev_mean_ba[gt][ind] - prev2_mean_ba[gt][ind])
                    } else {
                        f32::NAN
                    };

                // 与上一秒该行业桶的连续性（双指针交集）
                let (grp_overlap, grp_gap, grp_prev_avail) = if prev_sec[gt][ind] != usize::MAX {
                    let mut sorted_bucket = bucket.clone();
                    sorted_bucket.sort_unstable();
                    let prev = &prev_bucket[gt][ind];
                    let (mut i, mut j, mut ov) = (0usize, 0usize, 0usize);
                    while i < sorted_bucket.len() && j < prev.len() {
                        match sorted_bucket[i].cmp(&prev[j]) {
                            std::cmp::Ordering::Equal => {
                                ov += 1;
                                i += 1;
                                j += 1;
                            }
                            std::cmp::Ordering::Less => i += 1,
                            std::cmp::Ordering::Greater => j += 1,
                        }
                    }
                    (ov, sec - prev_sec[gt][ind], true)
                } else {
                    (0usize, 0usize, false)
                };

                // stay_seconds 更新（行业内桶口径）
                let contiguous = grp_prev_avail && grp_gap == SECOND_STEP;
                for &si in bucket.iter() {
                    if !stay_seen[si][gt] {
                        stay_seen[si][gt] = true;
                        let in_prev = if contiguous {
                            prev_bucket[gt][ind].binary_search(&si).is_ok()
                        } else {
                            false
                        };
                        if contiguous && in_prev {
                            stay_seconds[si][gt] += 1;
                        } else {
                            stay_seconds[si][gt] = if contiguous { 0 } else { 1 };
                        }
                    }
                }
                for &si in bucket.iter() {
                    stay_seen[si][gt] = false;
                }
                let stay_ge3 = bucket.iter().filter(|&&si| stay_seconds[si][gt] >= 3).count();
                group_feats[33] = stay_ge3 as f32 / n_b.max(1) as f32;

                // per-stock 组装 40 维
                for (rank_i, &stock_i) in bucket.iter().enumerate() {
                    let rank_pct = rank_i as f32 / n_b.max(1) as f32;
                    let mut per_stock = group_feats;
                    fill_per_stock_arr(
                        &mut per_stock, &br_sorted, &ba_sorted, &vol_sorted, sto_buy[rank_i],
                        sto_ba[rank_i], sto_vol[rank_i], br_m, br_s, ba_m, ba_s,
                    );
                    per_stock[12] = neighbor_mean_inline(&sto_buy, rank_i, 3);
                    per_stock[25] = neighbor_mean_inline(&sto_ba, rank_i, 3);
                    per_stock[29] = c04_val;
                    let (cf01, cf02, cf03, cf04, cf05) = if grp_prev_avail {
                        (
                            grp_overlap as f32,
                            grp_overlap as f32
                                / std::cmp::min(n_b, prev_bucket[gt][ind].len()).max(1) as f32,
                            rank_pct,
                            if grp_gap == SECOND_STEP { 2.0 } else { 1.0 },
                            grp_gap as f32,
                        )
                    } else {
                        (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN)
                    };
                    let mut all_feats: Feat40 = [f32::NAN; FEAT_PER_INCLUSION];
                    all_feats[..34].copy_from_slice(&per_stock);
                    all_feats[34] = cf01;
                    all_feats[35] = cf02;
                    all_feats[36] = cf03;
                    all_feats[37] = cf04;
                    all_feats[38] = cf05;
                    all_feats[39] = f32::NAN;
                    if gt == 0 {
                        indpool_hot[stock_i].push(all_feats);
                    } else {
                        indpool_cold[stock_i].push(all_feats);
                    }
                }

                if gt == 0 {
                    ind_pool_log_hot[ind].push(bucket.clone());
                } else {
                    ind_pool_log_cold[ind].push(bucket.clone());
                }

                prev2_mean_br[gt][ind] = prev_mean_br[gt][ind];
                prev_mean_br[gt][ind] = mean_br;
                prev2_mean_ba[gt][ind] = prev_mean_ba[gt][ind];
                prev_mean_ba[gt][ind] = mean_ba;
                prev_sec[gt][ind] = sec;
                let mut s = bucket.clone();
                s.sort_unstable();
                prev_bucket[gt][ind] = s;
            }
        }
    }

    for si in 0..n_valid {
        let ind = stocks[si].ind as usize;
        if ind >= 1 && ind <= 31 {
            if hot_cnt[si] > 0 {
                hot_sum_ind[ind] += hot_cnt[si];
                hot_nstock_ind[ind] += 1;
            }
            if cold_cnt[si] > 0 {
                cold_sum_ind[ind] += cold_cnt[si];
                cold_nstock_ind[ind] += 1;
            }
        }
    }
    mkt_hot_total = hot_sum_ind.iter().sum();
    mkt_cold_total = cold_sum_ind.iter().sum();

    PiOut {
        rel_hot,
        rel_cold,
        pool_log_hot,
        pool_log_cold,
        hot_cnt,
        cold_cnt,
        hot_sum_ind,
        cold_sum_ind,
        hot_nstock_ind,
        cold_nstock_ind,
        mkt_hot_total,
        mkt_cold_total,
        indpool_hot,
        indpool_cold,
        ind_pool_log_hot,
        ind_pool_log_cold,
    }
}

/// 组 1 回填：行业内当日全部入选记录的排名百分位 / 截面 z / 频率比
fn backfill_rel(rel: &mut [Vec<[f32; 4]>], stocks: &[StockData]) -> Vec<Vec<[f32; 5]>> {
    let n = rel.len();
    let mut ind_inc_cnt = [0u32; 32];
    let mut ind_nstock = [0u32; 32];
    for si in 0..n {
        let ind = stocks[si].ind as usize;
        if ind >= 1 && ind <= 31 {
            ind_inc_cnt[ind] += rel[si].len() as u32;
            ind_nstock[ind] += 1;
        }
    }
    for ind in 1..=31usize {
        if ind_inc_cnt[ind] < 2 {
            continue;
        }
        for kind in 0..4usize {
            let mut vals: Vec<(f32, usize, usize)> = Vec::with_capacity(ind_inc_cnt[ind] as usize);
            for si in 0..n {
                if stocks[si].ind as usize != ind {
                    continue;
                }
                for (k, rec) in rel[si].iter().enumerate() {
                    if rec[kind].is_finite() {
                        vals.push((rec[kind], si, k));
                    }
                }
            }
            if vals.len() < 2 {
                continue;
            }
            vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let sorted: Vec<f32> = vals.iter().map(|v| v.0).collect();
            if kind == 3 {
                let (m, s) = mean_std(&sorted);
                if s > 1e-12 {
                    for &(v, si, k) in vals.iter() {
                        rel[si][k][3] = ((v - m) / s) as f32;
                    }
                } else {
                    for &(_, si, k) in vals.iter() {
                        rel[si][k][3] = f32::NAN;
                    }
                }
            } else {
                let mut seq_pos = 0usize;
                for &(v, si, k) in vals.iter() {
                    while seq_pos < sorted.len() && sorted[seq_pos] < v {
                        seq_pos += 1;
                    }
                    rel[si][k][kind] = seq_pos as f32 / sorted.len() as f32;
                }
            }
        }
    }
    // freq: 截至第 k 条的累计次数 / 行业人均入选次数
    let mut out: Vec<Vec<[f32; 5]>> = vec![Vec::new(); n];
    for si in 0..n {
        let ind = stocks[si].ind as usize;
        let denom = if ind >= 1 && ind <= 31 && ind_nstock[ind] > 0 {
            ind_inc_cnt[ind] as f32 / ind_nstock[ind] as f32
        } else {
            f32::NAN
        };
        for (k, rec) in rel[si].iter().enumerate() {
            let mut r5 = [f32::NAN; 5];
            r5[..4].copy_from_slice(rec);
            r5[4] = if denom.is_finite() && denom > 0.0 {
                (k + 1) as f32 / denom
            } else {
                f32::NAN
            };
            out[si].push(r5);
        }
    }
    out
}

/// 21 统计量降维（无 corr）：用正式库 get_features_factors_rust_full 并过滤 corr 名。
fn reduce_21(seq_flat: &[f32], n_rows: usize, n_cols: usize, col_names: &[String]) -> Vec<f32> {
    if n_rows == 0 || n_cols == 0 {
        return vec![f32::NAN; 21 * n_cols];
    }
    let arr = Array2::from_shape_vec((n_rows, n_cols), seq_flat.to_vec())
        .unwrap_or_else(|_| Array2::zeros((0, n_cols)));
    let (vals, names) = get_features_factors_rust_full(&arr.view(), col_names, false);
    vals.into_iter()
        .zip(names.iter())
        .filter(|(_, n)| !n.contains("_corr_"))
        .map(|(v, _)| v)
        .collect()
}

fn reduce_seq(seq: &[[f32; 5]], col_names: &[String]) -> Vec<f32> {
    let z = seq.len();
    if z == 0 {
        return vec![f32::NAN; 21 * 5];
    }
    let mut flat = Vec::with_capacity(z * 5);
    for r in seq.iter() {
        flat.extend_from_slice(r);
    }
    reduce_21(&flat, z, 5, col_names)
}

fn reduce_seq40(seq: &[Feat40], col_names: &[String]) -> Vec<f32> {
    let z = seq.len();
    if z == 0 {
        return vec![f32::NAN; 21 * FEAT_PER_INCLUSION];
    }
    let mut flat = Vec::with_capacity(z * FEAT_PER_INCLUSION);
    for r in seq.iter() {
        flat.extend_from_slice(r);
    }
    reduce_21(&flat, z, FEAT_PER_INCLUSION, col_names)
}

/// 组 2：初版共现 top-10 同行/跨行拆分（返回 hot 块 + cold 块）
fn g2_cooc(stock_i: usize, pi_out: &PiOut, stocks: &[StockData]) -> [f32; G2_PER * 2] {
    let n_valid = stocks.len();
    let mut out = [f32::NAN; G2_PER * 2];
    let mut buf = vec![0u32; n_valid];
    let self_ind = stocks[stock_i].ind;
    let mut pairs_h: Vec<(usize, u32)> = Vec::new();
    for pool in pi_out.pool_log_hot.iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    for (si, &cnt) in buf.iter().enumerate() {
        if cnt > 0 && si != stock_i {
            pairs_h.push((si, cnt));
        }
    }
    pairs_h.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_h: Vec<(usize, u32)> = pairs_h.iter().take(10).copied().collect();
    for &(si, _) in pairs_h.iter() {
        buf[si] = 0;
    }
    let mut pairs_c: Vec<(usize, u32)> = Vec::new();
    for pool in pi_out.pool_log_cold.iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    for (si, &cnt) in buf.iter().enumerate() {
        if cnt > 0 && si != stock_i {
            pairs_c.push((si, cnt));
        }
    }
    pairs_c.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_c: Vec<(usize, u32)> = pairs_c.iter().take(10).copied().collect();

    for (gt, top10) in [0usize, 1usize].iter().zip([&top10_h, &top10_c].iter()) {
        let gt = *gt;
        let same: Vec<usize> = top10
            .iter()
            .filter(|(si, _)| stocks[*si].ind == self_ind && self_ind >= 1)
            .map(|(si, _)| *si)
            .collect();
        let cross: Vec<usize> = top10
            .iter()
            .filter(|(si, _)| stocks[*si].ind != self_ind || self_ind < 1)
            .map(|(si, _)| *si)
            .collect();
        let ratio = same.len() as f32 / 10.0;
        let cntsum: f32 = top10
            .iter()
            .filter(|(si, _)| stocks[*si].ind == self_ind && self_ind >= 1)
            .map(|(_, c)| *c as f32)
            .sum();
        let base = gt * G2_PER;
        out[base + BASIC_FEAT_N * 2 * 4] = ratio;
        out[base + BASIC_FEAT_N * 2 * 4 + 1] = cntsum;
        for j in 0..BASIC_FEAT_N {
            let same_ms = ms_of(&same, j, stocks);
            let cross_ms = ms_of(&cross, j, stocks);
            for s in 0..2usize {
                let (sv, cv) = (same_ms[s], cross_ms[s]);
                let idx = base + j * 8 + s * 4;
                out[idx] = sv;
                out[idx + 1] = cv;
                out[idx + 2] = if sv.is_finite() && cv.is_finite() {
                    sv - cv
                } else {
                    f32::NAN
                };
                out[idx + 3] = if sv.is_finite() && cv.is_finite() {
                    (sv - cv).abs()
                } else {
                    f32::NAN
                };
            }
        }
    }
    out
}

fn ms_of(peers: &[usize], j: usize, stocks: &[StockData]) -> [f32; 2] {
    let col: Vec<f32> = peers
        .iter()
        .map(|&si| stocks[si].basic_feats[j])
        .filter(|v| v.is_finite())
        .collect();
    if col.len() >= 2 {
        let m = col.iter().sum::<f32>() / col.len() as f32;
        let var = col.iter().map(|v| (v - m).powi(2)).sum::<f32>() / col.len() as f32;
        [m, var.sqrt()]
    } else {
        [f32::NAN, f32::NAN]
    }
}

/// 组 4 共现：同行共现 top-10（初版 4 口径）
fn g4_cooc(stock_i: usize, pi_out: &PiOut, stocks: &[StockData]) -> [f32; G4_COOC_PER] {
    let n_valid = stocks.len();
    let mut out = [f32::NAN; G4_COOC_PER];
    let self_ind = stocks[stock_i].ind as usize;
    if self_ind < 1 || self_ind > 31 {
        return out;
    }
    let mut buf = vec![0u32; n_valid];
    for pool in pi_out.ind_pool_log_hot[self_ind].iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    let mut pairs_h: Vec<(usize, u32)> = buf
        .iter()
        .enumerate()
        .filter(|&(si, &cnt)| si != stock_i && cnt > 0)
        .map(|(si, &cnt)| (si, cnt))
        .collect();
    pairs_h.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_h: Vec<usize> = pairs_h.iter().take(10).map(|(si, _)| *si).collect();
    for &(si, _) in pairs_h.iter() {
        buf[si] = 0;
    }
    for pool in pi_out.ind_pool_log_cold[self_ind].iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    let mut pairs_c: Vec<(usize, u32)> = buf
        .iter()
        .enumerate()
        .filter(|&(si, &cnt)| si != stock_i && cnt > 0)
        .map(|(si, &cnt)| (si, cnt))
        .collect();
    pairs_c.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_c: Vec<usize> = pairs_c.iter().take(10).map(|(si, _)| *si).collect();

    let hot_ms = ms_all(&top10_h, stocks);
    let cold_ms = ms_all(&top10_c, stocks);
    for j in 0..BASIC_FEAT_N {
        for s in 0..2usize {
            let hv = hot_ms[s * BASIC_FEAT_N + j];
            let cv = cold_ms[s * BASIC_FEAT_N + j];
            let idx = (j * 2 + s) * 4;
            out[idx] = hv;
            out[idx + 1] = cv;
            out[idx + 2] = if hv.is_finite() && cv.is_finite() {
                hv - cv
            } else {
                f32::NAN
            };
            out[idx + 3] = if hv.is_finite() && cv.is_finite() {
                (hv - cv).abs()
            } else {
                f32::NAN
            };
        }
    }
    out
}

fn ms_all(peers: &[usize], stocks: &[StockData]) -> [f32; BASIC_FEAT_N * 2] {
    let mut out = [f32::NAN; BASIC_FEAT_N * 2];
    for j in 0..BASIC_FEAT_N {
        let col: Vec<f32> = peers
            .iter()
            .map(|&si| stocks[si].basic_feats[j])
            .filter(|v| v.is_finite())
            .collect();
        if col.len() >= 2 {
            let m = col.iter().sum::<f32>() / col.len() as f32;
            let var = col.iter().map(|v| (v - m).powi(2)).sum::<f32>() / col.len() as f32;
            out[j] = m;
            out[BASIC_FEAT_N + j] = var.sqrt();
        }
    }
    out
}

/// 组 3：行业热度聚合（每股 [sum, nratio, share, z, own, NaN]，spread 由调用方回填）
fn g3_heat(
    si: usize,
    pi_out: &PiOut,
    stocks: &[StockData],
    ind_nstock: &[u32; 32],
    sum_m: f32,
    sum_sd: f32,
) -> [f32; G3_PER] {
    let mut out = [f32::NAN; G3_PER];
    let ind = stocks[si].ind as usize;
    if ind < 1 || ind > 31 {
        return out;
    }
    let hs = pi_out.hot_sum_ind[ind] as f32;
    let nratio = if ind_nstock[ind] > 0 {
        pi_out.hot_nstock_ind[ind] as f32 / ind_nstock[ind] as f32
    } else {
        f32::NAN
    };
    let share = if pi_out.mkt_hot_total > 0 {
        hs / pi_out.mkt_hot_total as f32
    } else {
        f32::NAN
    };
    let z = if sum_sd > 1e-12 {
        (hs - sum_m) / sum_sd
    } else {
        f32::NAN
    };
    let own = if hs > 0.0 {
        pi_out.hot_cnt[si] as f32 / hs
    } else {
        f32::NAN
    };
    out[0] = hs;
    out[1] = nratio;
    out[2] = share;
    out[3] = z;
    out[4] = own;
    out
}

/// 主入口：读盘 → 计算 8656 个补充因子 → (codes, vals row-major)
pub fn compute_hot_pool_ind_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let ind_map = load_industry();
    let codes = list_codes(date, "transaction");
    if codes.is_empty() {
        return Ok((vec![], vec![]));
    }

    // ① rayon 并行读全市场逐笔 + per-stock 构建数据（含行业）
    let stocks: Vec<Option<StockData>> = codes
        .par_iter()
        .map(|code| {
            let ind = *ind_map.get(code).unwrap_or(&0);
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            build_stock_data(code, date, ind, &trades)
        })
        .collect();

    let mut valid_stocks: Vec<StockData> = Vec::new();
    for s in stocks.into_iter() {
        if let Some(sd) = s {
            if sd.secs.iter().any(|s| s.has_data) {
                valid_stocks.push(sd);
            }
        }
    }
    let n_valid = valid_stocks.len();
    if n_valid == 0 {
        return Ok((vec![], vec![]));
    }

    // ② 滚动缓存
    let rolling_caches: Vec<RollingCache> = valid_stocks
        .par_iter()
        .map(|sd| RollingCache::compute(&sd.secs))
        .collect();

    // ③ 主循环（4 参数组合并行）
    let pi_outs: Vec<PiOut> = PARAM_CONFIGS
        .par_iter()
        .map(|&(x, _y, d_type, _d_threshold, min_trades)| {
            compute_pi(x, d_type, min_trades, &rolling_caches, &valid_stocks)
        })
        .collect();

    // ④ 组 1 回填
    let mut rel_back: Vec<(Vec<Vec<[f32; 5]>>, Vec<Vec<[f32; 5]>>)> = Vec::new();
    for po in pi_outs.iter() {
        let mut rh = po.rel_hot.clone();
        let mut rc = po.rel_cold.clone();
        let bfh = backfill_rel(&mut rh, &valid_stocks);
        let bfc = backfill_rel(&mut rc, &valid_stocks);
        rel_back.push((bfh, bfc));
    }

    // ⑤ 组 3 预计算
    let mut ind_nstock = [0u32; 32];
    for s in valid_stocks.iter() {
        let ind = s.ind as usize;
        if ind >= 1 && ind <= 31 {
            ind_nstock[ind] += 1;
        }
    }
    let g3_meta: Vec<(f32, f32, [f32; 32])> = pi_outs
        .iter()
        .map(|po| {
            let sums: Vec<f32> = (1..=31).map(|i| po.hot_sum_ind[i] as f32).collect();
            let (m, sd) = mean_std(&sums);
            let mut spread_ind = [f32::NAN; 32];
            for i in 1..=31usize {
                spread_ind[i] = po.hot_sum_ind[i] as f32 - po.cold_sum_ind[i] as f32;
            }
            (m, sd, spread_ind)
        })
        .collect();

    // 降维列名
    let rel_col_names: Vec<String> =
        ["rk_buy", "rk_vol", "rk_ba", "z", "freq"].iter().map(|s| s.to_string()).collect();
    let f40_col_names: Vec<String> = (0..FEAT_PER_INCLUSION).map(|i| format!("f{i:02}")).collect();

    // ⑥ 组装每股因子
    let all_factors: Vec<Vec<f32>> = (0..n_valid)
        .into_par_iter()
        .map(|stock_i| {
            let mut facs = vec![f32::NAN; N_FACTORS];
            let mut off = 0usize;
            for pi in 0..N_PARAM_COMBOS {
                for seq in [&rel_back[pi].0[stock_i], &rel_back[pi].1[stock_i]] {
                    let vals = reduce_seq(seq, &rel_col_names);
                    facs[off..off + vals.len()].copy_from_slice(&vals);
                    off += vals.len();
                }
            }
            for pi in 0..N_PARAM_COMBOS {
                let vals = g2_cooc(stock_i, &pi_outs[pi], &valid_stocks);
                facs[off..off + vals.len()].copy_from_slice(&vals);
                off += vals.len();
            }
            for pi in 0..N_PARAM_COMBOS {
                let (m, sd, spread_ind) = &g3_meta[pi];
                let mut vals = g3_heat(stock_i, &pi_outs[pi], &valid_stocks, &ind_nstock, *m, *sd);
                let ind = valid_stocks[stock_i].ind as usize;
                if ind >= 1 && ind <= 31 {
                    vals[5] = spread_ind[ind];
                }
                facs[off..off + vals.len()].copy_from_slice(&vals);
                off += vals.len();
            }
            for pi in 0..N_PARAM_COMBOS {
                for seq in
                    [&pi_outs[pi].indpool_hot[stock_i], &pi_outs[pi].indpool_cold[stock_i]]
                {
                    let vals = reduce_seq40(seq, &f40_col_names);
                    facs[off..off + vals.len()].copy_from_slice(&vals);
                    off += vals.len();
                }
            }
            for pi in 0..N_PARAM_COMBOS {
                let vals = g4_cooc(stock_i, &pi_outs[pi], &valid_stocks);
                facs[off..off + vals.len()].copy_from_slice(&vals);
                off += vals.len();
            }
            facs
        })
        .collect();

    // ⑦ fan-out
    let mut out_codes = Vec::with_capacity(n_valid);
    let mut out_vals = Vec::with_capacity(n_valid * N_FACTORS);
    for (stock_i, facs) in all_factors.iter().enumerate() {
        out_codes.push(valid_stocks[stock_i].code.clone());
        out_vals.extend_from_slice(facs);
    }
    Ok((out_codes, out_vals))
}

/// 因子名（与 compute_hot_pool_ind_full 输出顺序严格一致）。
pub fn hot_pool_ind_names() -> Vec<String> {
    let rel_feats = ["rk_buy", "rk_vol", "rk_ba", "z", "freq"];
    let basic_names = [
        "total_buy_ratio",
        "total_return",
        "ret_15s_std",
        "ret_60s_std",
        "buy_ratio_15s_std",
        "buy_ratio_60s_std",
        "bid_ask_15s_std",
        "bid_ask_60s_std",
        "total_volume",
        "vol_15s_std",
        "vol_60s_std",
    ];
    let mut names = Vec::with_capacity(N_FACTORS);
    // 组 1/组 4 降维名：stat-major（与 get_features_factors_rust_full 的 push 顺序一致）
    let dummy1 = Array2::zeros((2, 5));
    let rel_cols: Vec<String> = rel_feats.iter().map(|s| s.to_string()).collect();
    let (_, rel_reduced) = get_features_factors_rust_full(&dummy1.view(), &rel_cols, false);
    let rel_reduced: Vec<&str> = rel_reduced
        .iter()
        .filter(|n| !n.contains("_corr_"))
        .map(|s| s.as_str())
        .collect();
    let dummy4 = Array2::zeros((2, FEAT_PER_INCLUSION));
    let f40_cols: Vec<String> = (0..FEAT_PER_INCLUSION).map(|i| format!("f{i:02}")).collect();
    let (_, f40_reduced) = get_features_factors_rust_full(&dummy4.view(), &f40_cols, false);
    let f40_reduced: Vec<&str> = f40_reduced
        .iter()
        .filter(|n| !n.contains("_corr_"))
        .map(|s| s.as_str())
        .collect();

    for combo in COMBO_LABELS.iter() {
        for grp in ["hot", "cold"].iter() {
            for n in rel_reduced.iter() {
                names.push(format!("hotpool_ext_ind_rel_{combo}_{grp}_{n}"));
            }
        }
    }
    for combo in COMBO_LABELS.iter() {
        for grp in ["hot", "cold"].iter() {
            for b in basic_names.iter() {
                for s in ["mean", "std"].iter() {
                    for kind in ["same", "cross", "diff", "absdiff"].iter() {
                        names.push(format!(
                            "hotpool_ext_ind_cooc_{combo}_{grp}_{b}_{s}_{kind}"
                        ));
                    }
                }
            }
            names.push(format!("hotpool_ext_ind_cooc_{combo}_{grp}_struct_ratio"));
            names.push(format!("hotpool_ext_ind_cooc_{combo}_{grp}_struct_cntsum"));
        }
    }
    for combo in COMBO_LABELS.iter() {
        for metric in ["sum", "nratio", "share", "z", "own", "spread"].iter() {
            names.push(format!("hotpool_ext_ind_heat_{combo}_{metric}"));
        }
    }
    for combo in COMBO_LABELS.iter() {
        for grp in ["hot", "cold"].iter() {
            for n in f40_reduced.iter() {
                names.push(format!("hotpool_ext_indpool_{combo}_{grp}_{n}"));
            }
        }
    }
    for combo in COMBO_LABELS.iter() {
        for b in basic_names.iter() {
            for s in ["mean", "std"].iter() {
                for t in ["hot", "cold", "diff", "abs_diff"].iter() {
                    names.push(format!("hotpool_ext_indpool_cooc_{combo}_{b}_{s}_{t}"));
                }
            }
        }
    }
    assert_eq!(names.len(), N_FACTORS);
    names
}

// ============================================================
// PyO3 入口
// ============================================================

#[pyfunction]
pub fn py_hot_pool_ind(_py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_hot_pool_ind_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_hot_pool_ind_names() -> Vec<String> {
    hot_pool_ind_names()
}
