//! 同热点股票池因子：读一天全市场 Level2 逐笔 → 每秒识别同热/同冰点股票组 →
//! per-stock 算 40 维入选特征 → 降维 → 共现因子 → fan-out。
//!
//! # 参数
//! - x ∈ {15, 60} 秒窗口，y ∈ {3%, 10%}
//! - D ∈ {"buy_ratio", "bid_ask_diff"} 判别指标
//! - 共 4 组 (x,y,D) 参数组合: (60,3%,buy),(60,3%,bid),(15,10%,buy),(15,10%,bid)
//!
//! # 计算流程
//! 1. rayon 并行读全市场逐笔 → per-stock 按秒降采样
//! 2. 每秒计算全市场滚动窗口指标 → 排序取 top/bottom y% → 构建组
//! 3. 每组算 40 个入选特征 → 每只股票累积 z×40 矩阵
//! 4. get_features_factors_rust_full 降维每个矩阵
//! 5. 共现统计 + 基础特征均值/标准差
//! 6. 热点-冰点差分
//!
//! # 因子数
//! 8 矩阵 × features_per_group(40) = 8 × 1620 = 12960
//! + 共现 352 = 13312

use crate::fast_csv_reader::read_trade_fast_inner;
use crate::features::get_features_factors_rust_full;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::fs;

// ============================================================
// 常量
// ============================================================

/// 每只股票每次入选时计算的特征数（步骤2: 34 + 步骤3: 6 = 40）
const FEAT_PER_INCLUSION: usize = 40;

/// 参数组合数：2 x × 2 D = 4
const N_PARAM_COMBOS: usize = 4;

/// 每参数组合的降维特征数：features_per_group(40) = 21*40 + C(40,2) = 840 + 780 = 1620
const REDUCED_PER_COMBO: usize = 21 * FEAT_PER_INCLUSION + FEAT_PER_INCLUSION * (FEAT_PER_INCLUSION - 1) / 2; // 1620

/// 共现因子数：4 组合 × 11基础特征 × 2(mean/std) × 4(热/冷/差/绝对值差) = 352
const COOCCUR_FACTORS: usize = 4 * 11 * 2 * 4;

/// 总因子数
pub const N_FACTORS: usize = 8 * REDUCED_PER_COMBO + COOCCUR_FACTORS; // 12960 + 352 = 13312

/// 交易时段秒数（adjust_afternoon 后）：上午 09:30-11:30 (7200秒) + 下午 13:00-14:57 → 前移90分 → 7200秒 = 14400秒
const TOTAL_SECONDS: usize = 14400;

/// 秒槽位起始偏移（epoch秒对应当天零点）
/// 上午 epoch = 09:30:00 相对当天零点 = 34200
/// 但 adjust_afternoon 后，下午前移90分，实际交易时间映射到 [34200, 48600) 连续区间
const SEC_OFFSET: i64 = 9 * 3600 + 30 * 60; // 34200

/// 上午结束 epoch（调整后）
const MORNING_END: i64 = 11 * 3600 + 30 * 60; // 41400
/// 下午开始 epoch（调整后，原13:00前移90分）
const AFTERNOON_START: i64 = MORNING_END + 1; // 41401
/// 下午结束 epoch（调整后，原14:57前移90分）
const AFTERNOON_END: i64 = MORNING_END + (14 * 3600 + 57 * 60 - 13 * 3600); // 48420
/// 总共调整后的交易秒数
const ADJUSTED_SECONDS: usize = ((MORNING_END - SEC_OFFSET) + (AFTERNOON_END - AFTERNOON_START + 1)) as usize;

/// 参数配置：(x秒, y百分比, 判别指标类型: 0=buy_ratio, 1=bid_ask_diff)
const PARAM_CONFIGS: [(usize, f64, usize); N_PARAM_COMBOS] = [
    (60, 0.03, 0),  // x=60, y=3%, buy_ratio
    (60, 0.03, 1),  // x=60, y=3%, bid_ask_diff
    (15, 0.10, 0),  // x=15, y=10%, buy_ratio
    (15, 0.10, 1),  // x=15, y=10%, bid_ask_diff
];

/// 共现基础特征数
const BASIC_FEAT_N: usize = 11;

// ============================================================
// 数据结构
// ============================================================

/// 单股每秒的统计指标
#[derive(Clone, Copy, Default)]
struct SecStat {
    buy_ratio: f32,      // 该秒主买成交量占比
    bid_ask_mean: f32,   // 该秒 bid_order - ask_order 均值
    ret_val: f32,        // 该秒收益率
    volume: f32,         // 该秒总成交量
    last_price: f32,     // 该秒末价格
    first_price: f32,    // 该秒初价格
    has_data: bool,      // 该秒是否有成交
}

/// 单股预计算数据
struct StockData {
    code: String,
    secs: Vec<SecStat>,               // 14400 长度，索引 = epoch_sec - SEC_OFFSET
    // 共现用的 11 个基础特征
    basic_feats: [f32; BASIC_FEAT_N],
}

/// 某次入选时该股的特征快照（用于后续步骤3连续性计算）
#[derive(Clone)]
struct InclusionInfo {
    second_idx: usize,    // 秒索引
    rank_pct: f32,        // 该股在组内的排名百分位
    pool_size: usize,     // 该组总股票数
    pool_codes: Vec<String>, // 该组股票代码列表
}

// ============================================================
// 工具函数
// ============================================================

/// 列出某天某子目录下所有股票代码
pub fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/{subdir}");
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// epoch秒 → 数组索引。无效时间返回 None
#[inline]
fn sec_to_idx(epoch: f32) -> Option<usize> {
    let e = epoch as i64;
    if e < SEC_OFFSET || e > AFTERNOON_END {
        return None;
    }
    if e <= MORNING_END {
        Some((e - SEC_OFFSET) as usize)
    } else if e >= AFTERNOON_START && e <= AFTERNOON_END {
        Some((MORNING_END - SEC_OFFSET + 1 + e - AFTERNOON_START) as usize)
    } else {
        None
    }
}

/// 简单移动平均（忽略 NaN）
fn rolling_mean(data: &[f32], len: usize) -> Vec<f32> {
    let n = data.len();
    let mut out = vec![f32::NAN; n];
    if n < len || len == 0 { return out; }
    let mut sum: f64 = 0.0;
    let mut cnt: usize = 0;
    for i in 0..n {
        if data[i].is_finite() { sum += data[i] as f64; cnt += 1; }
        if i >= len {
            let j = i - len;
            if data[j].is_finite() { sum -= data[j] as f64; cnt -= 1; }
        }
        if cnt > 0 { out[i] = (sum / cnt as f64) as f32; }
    }
    out
}

/// 简单移动窗口内总和
fn rolling_sum(data: &[f32], len: usize) -> Vec<f32> {
    let n = data.len();
    let mut out = vec![f32::NAN; n];
    if n < len || len == 0 { return out; }
    let mut sum: f64 = 0.0;
    for i in 0..n {
        if data[i].is_finite() { sum += data[i] as f64; }
        if i >= len {
            let j = i - len;
            if data[j].is_finite() { sum -= data[j] as f64; }
        }
        out[i] = sum as f32;
    }
    out
}

/// 简单移动标准差
fn rolling_std(data: &[f32], len: usize) -> Vec<f32> {
    let n = data.len();
    let mut out = vec![f32::NAN; n];
    if n < len || len == 0 { return out; }
    for i in (len - 1)..n {
        let window = &data[i + 1 - len..=i];
        let finite: Vec<f32> = window.iter().filter(|v| v.is_finite()).copied().collect();
        if finite.len() < 2 { continue; }
        let mean = finite.iter().sum::<f32>() / finite.len() as f32;
        let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / finite.len() as f32;
        out[i] = var.sqrt();
    }
    out
}

/// NaN 填充向量
fn nan_vec(n: usize) -> Vec<f32> {
    vec![f32::NAN; n]
}

/// 从 SecStat 数组计算在 second_idx 处、窗口长度 x 的滚动均值
/// field: 0=buy_ratio, 1=bid_ask_mean, 2=ret_val
#[inline]
fn rolling_mean_at(secs: &[SecStat], x: usize, second_idx: usize, field: u8) -> f32 {
    if second_idx < x - 1 { return f32::NAN; }
    let start = second_idx + 1 - x;
    let window = &secs[start..=second_idx];
    let mut sum: f64 = 0.0;
    let mut cnt: usize = 0;
    for s in window.iter() {
        let v = match field {
            0 => s.buy_ratio,
            1 => s.bid_ask_mean,
            _ => s.ret_val,
        };
        if v.is_finite() { sum += v as f64; cnt += 1; }
    }
    if cnt > 0 { (sum / cnt as f64) as f32 } else { f32::NAN }
}

/// 从 SecStat 数组计算在 second_idx 处、窗口长度 x 的滚动成交量总和
#[inline]
fn rolling_sum_at(secs: &[SecStat], x: usize, second_idx: usize) -> f32 {
    if second_idx < x - 1 { return f32::NAN; }
    let start = second_idx + 1 - x;
    secs[start..=second_idx].iter().map(|s| s.volume).sum()
}

/// 填充个股级别的 A08-A10, B08-B10, C03, A13, B13
fn fill_per_stock_features(
    feats: &mut [f32],
    sto_buy: &[f32], sto_ba: &[f32], sto_vol: &[f32],
    stock_rank: usize, sb: f32, sba: f32, sv: f32,
) {
    // 排序辅助
    fn sorted_finite(v: &[f32]) -> Vec<f32> {
        let mut s: Vec<f32> = v.iter().filter(|x| x.is_finite()).copied().collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s
    }
    let br_sorted = sorted_finite(sto_buy);
    let ba_sorted = sorted_finite(sto_ba);
    let vol_sorted = sorted_finite(sto_vol);

    // A08: 买占比排名百分位
    feats[7] = rank_pct_in(&br_sorted, sb);
    // A09: z-score
    let (bm, bs) = mean_std(&br_sorted);
    feats[8] = if bs > 1e-12 && sb.is_finite() { (sb - bm) / bs } else { f32::NAN };
    // A10: 与中位数差值
    feats[9] = if sb.is_finite() && !br_sorted.is_empty() { sb - br_sorted[br_sorted.len() / 2] } else { f32::NAN };
    // A13: 与该股相关性最高的3只股票买占比均值
    feats[12] = neighbor_mean(sto_buy, stock_rank, 3);

    // B08-B10, B13
    feats[7 + 13] = rank_pct_in(&ba_sorted, sba);
    let (bam, bas) = mean_std(&ba_sorted);
    feats[8 + 13] = if bas > 1e-12 && sba.is_finite() { (sba - bam) / bas } else { f32::NAN };
    feats[9 + 13] = if sba.is_finite() && !ba_sorted.is_empty() { sba - ba_sorted[ba_sorted.len() / 2] } else { f32::NAN };
    feats[12 + 13] = neighbor_mean(sto_ba, stock_rank, 3);

    // C03: 成交量排名百分位
    feats[26 + 2] = rank_pct_in(&vol_sorted, sv);
}

/// 在已排序数组中计算值的排名百分位
fn rank_pct_in(sorted: &[f32], val: f32) -> f32 {
    if sorted.is_empty() || !val.is_finite() { return f32::NAN; }
    let pos = sorted.partition_point(|&v| v < val);
    pos as f32 / sorted.len().max(1) as f32
}

/// 均值 + 标准差
fn mean_std(v: &[f32]) -> (f32, f32) {
    let n = v.len();
    if n < 2 { return (f32::NAN, f32::NAN); }
    let m = v.iter().sum::<f32>() / n as f32;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
    (m, var.sqrt())
}

/// 与某只股票相邻的 k 只股票的值均值（用于 A13/B13 简化）
fn neighbor_mean(values: &[f32], stock_rank: usize, k: usize) -> f32 {
    if values.is_empty() || values.len() < 2 { return f32::NAN; }
    let n = values.len();
    let start = if stock_rank >= k / 2 { stock_rank - k / 2 } else { 0 };
    let end = (start + k).min(n);
    let neighbors: Vec<f32> = values[start..end].iter()
        .enumerate()
        .filter(|(i, _)| start + i != stock_rank)
        .map(|(_, &v)| v)
        .filter(|v| v.is_finite())
        .collect();
    if neighbors.is_empty() { return f32::NAN; }
    neighbors.iter().sum::<f32>() / neighbors.len() as f32
}

// ============================================================
// Per-stock 数据准备
// ============================================================

/// 从逐笔成交构建 per-stock 每秒统计 + 基础特征（优化版，单 pass）
fn build_stock_data(code: &str, _date: i64, trades: &[crate::fast_csv_reader::TradeRecord]) -> Option<StockData> {
    let n_secs = ADJUSTED_SECONDS;
    let mut buy_vol = vec![0.0f64; n_secs];
    let mut total_vol = vec![0.0f64; n_secs];
    let mut bid_ask_sum = vec![0.0f64; n_secs];
    let mut bid_ask_cnt = vec![0u32; n_secs];
    let mut first_prices = vec![0.0f32; n_secs];
    let mut last_prices = vec![0.0f32; n_secs];
    let mut has_data = vec![false; n_secs];

    for t in trades {
        let idx = match sec_to_idx(t.time_sec) {
            Some(i) => i,
            None => continue,
        };
        has_data[idx] = true;
        let vol = t.volume as f64;
        total_vol[idx] += vol;
        if t.flag == 66 {
            buy_vol[idx] += vol;
        }
        bid_ask_sum[idx] += (t.bid_order - t.ask_order) as f64;
        bid_ask_cnt[idx] += 1;
        if first_prices[idx] == 0.0 { first_prices[idx] = t.price; }
        last_prices[idx] = t.price;
    }

    // 组装每秒统计
    let mut secs = Vec::with_capacity(n_secs);
    for i in 0..n_secs {
        let tv = total_vol[i];
        if tv > 0.0 {
            let br = if tv > 0.0 { (buy_vol[i] / tv) as f32 } else { f32::NAN };
            let ba = if bid_ask_cnt[i] > 0 { (bid_ask_sum[i] / bid_ask_cnt[i] as f64) as f32 } else { f32::NAN };
            let ret = if first_prices[i] > 0.0 { (last_prices[i] - first_prices[i]) / first_prices[i] } else { f32::NAN };
            secs.push(SecStat {
                buy_ratio: br,
                bid_ask_mean: ba,
                ret_val: ret,
                volume: tv as f32,
                last_price: last_prices[i],
                first_price: first_prices[i],
                has_data: true,
            });
        } else {
            secs.push(SecStat::default());
        }
    }

    // 计算共现基础特征
    let basic = compute_basic_features(&secs);
    Some(StockData { code: code.to_string(), secs, basic_feats: basic })
}

/// 计算单股的 11 个共现基础特征
fn compute_basic_features(secs: &[SecStat]) -> [f32; BASIC_FEAT_N] {
    let n = secs.len();
    // 提取各序列
    let buy_ratios: Vec<f32> = secs.iter().map(|s| if s.has_data { s.buy_ratio } else { f32::NAN }).collect();
    let rets: Vec<f32> = secs.iter().map(|s| if s.has_data { s.ret_val } else { f32::NAN }).collect();
    let bid_asks: Vec<f32> = secs.iter().map(|s| if s.has_data { s.bid_ask_mean } else { f32::NAN }).collect();
    let vols: Vec<f32> = secs.iter().map(|s| s.volume).collect();

    // 1. 总体主买占比
    let total_buy: f64 = secs.iter().map(|s| if s.buy_ratio.is_finite() { s.buy_ratio as f64 * s.volume as f64 } else { 0.0 }).sum();
    let total_vol: f64 = secs.iter().map(|s| s.volume as f64).sum();
    let g01 = if total_vol > 0.0 { (total_buy / total_vol) as f32 } else { f32::NAN };

    // 2. 总体收益率
    let first_p = secs.iter().find(|s| s.has_data && s.first_price > 0.0).map(|s| s.first_price);
    let last_p = secs.iter().rfind(|s| s.has_data && s.last_price > 0.0).map(|s| s.last_price);
    let g02 = match (first_p, last_p) {
        (Some(fp), Some(lp)) if fp > 0.0 => (lp - fp) / fp,
        _ => f32::NAN,
    };

    // 3-8: 各窗口 std
    fn safe_std(v: &[f32]) -> f32 {
        let finite: Vec<f32> = v.iter().filter(|x| x.is_finite()).copied().collect();
        if finite.len() < 2 { return f32::NAN; }
        let mean = finite.iter().sum::<f32>() / finite.len() as f32;
        (finite.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / finite.len() as f32).sqrt()
    }
    fn rolling_std_short(data: &[f32], len: usize) -> Vec<f32> {
        let n = data.len();
        let mut out = vec![f32::NAN; n];
        if n < len || len < 2 { return out; }
        for i in (len - 1)..n {
            out[i] = safe_std(&data[i + 1 - len..=i]);
        }
        out
    }

    let ret_15s_std = rolling_std_short(&rets, 15);
    let ret_60s_std = rolling_std_short(&rets, 60);
    let br_15s_std = rolling_std_short(&buy_ratios, 15);
    let br_60s_std = rolling_std_short(&buy_ratios, 60);
    let ba_15s_std = rolling_std_short(&bid_asks, 15);
    let ba_60s_std = rolling_std_short(&bid_asks, 60);
    let vol_15s_std = rolling_std_short(&vols, 15);
    let vol_60s_std = rolling_std_short(&vols, 60);

    let g03 = safe_std(&ret_15s_std);
    let g04 = safe_std(&ret_60s_std);
    let g05 = safe_std(&br_15s_std);
    let g06 = safe_std(&br_60s_std);
    let g07 = safe_std(&ba_15s_std);
    let g08 = safe_std(&ba_60s_std);
    let g09 = total_vol as f32;
    let g10 = safe_std(&vol_15s_std);
    let g11 = safe_std(&vol_60s_std);

    [g01, g02, g03, g04, g05, g06, g07, g08, g09, g10, g11]
}

// ============================================================
// 全市场横截面核心计算
// ============================================================

/// 对某一秒，计算全市场所有股票的滚动窗口判别指标值（用于排序分组）
fn compute_ranking_values(
    stocks: &[StockData],
    x: usize,
    d_type: usize, // 0=buy_ratio, 1=bid_ask_diff
    second_idx: usize,
) -> Vec<(usize, f32)> {
    // 对每只股票，取该秒前 x 秒窗口的判别指标均值
    let mut values: Vec<(usize, f32)> = stocks.iter().enumerate()
        .filter_map(|(si, sd)| {
            if second_idx < x - 1 { return None; }
            let start = second_idx + 1 - x;
            let window = &sd.secs[start..=second_idx];
            let mut sum: f64 = 0.0;
            let mut cnt: usize = 0;
            for s in window.iter() {
                let v = match d_type {
                    0 => s.buy_ratio,
                    _ => s.bid_ask_mean,
                };
                if v.is_finite() {
                    sum += v as f64;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                Some((si, (sum / cnt as f64) as f32))
            } else {
                None
            }
        })
        .collect();
    values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    values
}

/// 对某一秒，获取全市场某判别指标的滚动均值数组（用于超额热度等计算）
fn all_market_rolling_means(stocks: &[StockData], x: usize, d_type: usize, second_idx: usize) -> Vec<f32> {
    stocks.iter().map(|sd| {
        if second_idx < x - 1 { return f32::NAN; }
        let start = second_idx + 1 - x;
        let window = &sd.secs[start..=second_idx];
        let mut sum: f64 = 0.0;
        let mut cnt: usize = 0;
        for s in window.iter() {
            let v = if d_type == 0 { s.buy_ratio } else { s.bid_ask_mean };
            if v.is_finite() { sum += v as f64; cnt += 1; }
        }
        if cnt > 0 { (sum / cnt as f64) as f32 } else { f32::NAN }
    }).collect()
}

// ============================================================
// 步骤2+3：计算入选特征（40维）
// ============================================================

/// 计算一组"同热点/冰点股票"的 34 个截面特征（步骤2）
fn compute_group_features(
    pool_indices: &[usize],          // 组内股票的索引
    stocks: &[StockData],
    x: usize,
    d_type: usize,
    second_idx: usize,
    all_market_vals: &[f32],         // 全市场滚动判别均值
    prev_group_codes: Option<&Vec<String>>, // 上一秒同参数组的股票代码
    sto_buy_ratios: &[f32],          // 组内股票的滚动买占比
    sto_bid_asks: &[f32],            // 组内股票的滚动 bid_ask
    sto_rets: &[f32],                // 组内股票的滚动收益率
    sto_vols: &[f32],                // 组内股票的滚动成交量
) -> Vec<f32> {
    let n = pool_indices.len();
    if n == 0 { return vec![f32::NAN; 34]; }

    // 组内各指标
    let buy_ratios: Vec<f32> = pool_indices.iter().map(|&i| sto_buy_ratios[i]).collect();
    let bid_asks: Vec<f32> = pool_indices.iter().map(|&i| sto_bid_asks[i]).collect();
    let rets: Vec<f32> = pool_indices.iter().map(|&i| sto_rets[i]).collect();
    let vols: Vec<f32> = pool_indices.iter().map(|&i| sto_vols[i]).collect();

    let br_finite: Vec<f32> = buy_ratios.iter().filter(|v| v.is_finite()).copied().collect();
    let ba_finite: Vec<f32> = bid_asks.iter().filter(|v| v.is_finite()).copied().collect();
    let ret_finite: Vec<f32> = rets.iter().filter(|v| v.is_finite()).copied().collect();
    let vol_finite: Vec<f32> = vols.iter().copied().filter(|&v| v.is_finite() && v > 0.0).collect();

    let mut feats = Vec::with_capacity(34);

    // --- A01-A06: 主买占比基本统计 ---
    feats.push(mean(&br_finite));                          // A01
    feats.push(std(&br_finite));                           // A02
    feats.push(skew(&br_finite));                          // A03
    feats.push(kurtosis(&br_finite));                      // A04
    feats.push(q90_q10(&buy_ratios));                      // A05

    // --- A06-A07: 主买一阶/二阶差分（组均值）---
    let mean_br = mean(&br_finite);
    // 一阶差分需要上一秒的组均值，这里从调用方传入
    feats.push(f32::NAN);  // A06 占位，调用方填充
    feats.push(f32::NAN);  // A07 占位

    // --- A08-A10: 个股相对位置（每个入选股票单独算，这里只存组统计）---
    // 这些是个股级别特征，不在组特征中
    // 占位
    feats.push(f32::NAN); feats.push(f32::NAN); feats.push(f32::NAN); // A08-A10 占位

    // --- A11: 超额热度 ---
    let all_finite: Vec<f32> = all_market_vals.iter().filter(|v| v.is_finite()).copied().collect();
    let mkt_mean = mean(&all_finite);
    feats.push(if mkt_mean.is_finite() { mean_br - mkt_mean } else { f32::NAN }); // A11

    // --- A12: 组内买占比与收益率的截面 Spearman 相关 ---
    feats.push(spearman(&buy_ratios, &rets)); // A12

    // --- A13: 与该股买占比相关性最高的3只股票的买占比均值（个股特征，组特征占位）---
    feats.push(f32::NAN); // A13 占位

    // --- B01-B13: bid_order - ask_order 版本（同 A01-A13 结构）---
    feats.push(mean(&ba_finite));    feats.push(std(&ba_finite));
    feats.push(skew(&ba_finite));    feats.push(kurtosis(&ba_finite));
    feats.push(q90_q10(&bid_asks));
    feats.push(f32::NAN); feats.push(f32::NAN); // B06-B07 占位
    feats.push(f32::NAN); feats.push(f32::NAN); feats.push(f32::NAN); // B08-B10 占位
    let mean_ba = mean(&ba_finite);
    let all_ba: Vec<f32> = all_market_vals.iter().map(|_| f32::NAN).collect(); // FIXME: 需要传入全市场 bid_ask
    feats.push(f32::NAN); // B11 占位
    feats.push(spearman(&bid_asks, &rets)); // B12
    feats.push(f32::NAN); // B13 占位

    // --- C01-C04: 成交量与资金结构 ---
    feats.push(herfindahl(&vol_finite));                 // C01
    feats.push(top_k_concentration(&vol_finite, 3));     // C02
    feats.push(f32::NAN); // C03 个股特征占位
    let mkt_total_vol: f32 = vols.iter().sum();
    feats.push(f32::NAN); // C04 占位（需要全市场总成交）

    // --- D01-D03: 收益率统计 ---
    feats.push(mean(&ret_finite));
    feats.push(std(&ret_finite));
    feats.push(skew(&ret_finite));

    // --- E01: 连续停留 ≥3秒的股票占比 ---
    feats.push(f32::NAN); // E01 占位，需要历史状态

    assert_eq!(feats.len(), 34);
    feats
}

/// 计算个股入选连续性特征（步骤3，6个）
fn compute_continuity_features(
    current_info: &InclusionInfo,
    prev_info: Option<&InclusionInfo>,
    prev_N_infos: &[InclusionInfo], // 前 N 次入选信息
) -> Vec<f32> {
    let mut feats = Vec::with_capacity(6);
    if let Some(prev) = prev_info {
        // F01: 重合数量
        let overlap: usize = current_info.pool_codes.iter()
            .filter(|c| prev.pool_codes.contains(c))
            .count();
        feats.push(overlap as f32);

        // F02: 重合率
        let denom = std::cmp::min(current_info.pool_size, prev.pool_size).max(1);
        feats.push(overlap as f32 / denom as f32);

        // F03: 排名变化
        feats.push(current_info.rank_pct - prev.rank_pct);

        // F04: 连续入选持续秒数
        let gap = current_info.second_idx - prev.second_idx;
        feats.push(if gap == 1 { 2.0_f32 } else { 1.0_f32 }); // 简化：相邻则持续

        // F05: 距离上次入选的时间间隔
        feats.push(gap as f32);

        // F06: 前N次平均重合率
        let avg_overlap: f32 = if prev_N_infos.len() >= 2 {
            let mut total = 0.0f32;
            let mut cnt = 0;
            for i in 1..prev_N_infos.len() {
                let o = prev_N_infos[i].pool_codes.iter()
                    .filter(|c| prev_N_infos[i-1].pool_codes.contains(c))
                    .count();
                total += o as f32 / std::cmp::min(prev_N_infos[i].pool_size, prev_N_infos[i-1].pool_size).max(1) as f32;
                cnt += 1;
            }
            if cnt > 0 { total / cnt as f32 } else { f32::NAN }
        } else {
            f32::NAN
        };
        feats.push(avg_overlap);
    } else {
        feats.extend(&[f32::NAN; 6]);
    }
    assert_eq!(feats.len(), 6);
    feats
}

// ============================================================
// 统计工具函数
// ============================================================

fn mean(v: &[f32]) -> f32 {
    let n = v.len();
    if n == 0 { return f32::NAN; }
    v.iter().sum::<f32>() / n as f32
}

fn std(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 { return f32::NAN; }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32).sqrt()
}

fn skew(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 3 { return f32::NAN; }
    let m = mean(v);
    let s = std(v);
    if s < 1e-12 { return 0.0; }
    let m3 = v.iter().map(|x| (x - m).powi(3)).sum::<f32>() / n as f32;
    m3 / s.powi(3)
}

fn kurtosis(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 4 { return f32::NAN; }
    let m = mean(v);
    let s = std(v);
    if s < 1e-12 { return 0.0; }
    let m4 = v.iter().map(|x| (x - m).powi(4)).sum::<f32>() / n as f32;
    m4 / s.powi(4) - 3.0
}

fn percentile_sorted(sorted: &[f32], p: f32) -> f32 {
    let n = sorted.len();
    if n == 0 { return f32::NAN; }
    let idx = (p * (n - 1) as f32) as usize;
    sorted[idx.min(n - 1)]
}

fn q90_q10(v: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = v.iter().filter(|x| x.is_finite()).copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() < 2 { return f32::NAN; }
    percentile_sorted(&sorted, 0.90) - percentile_sorted(&sorted, 0.10)
}

fn herfindahl(v: &[f32]) -> f32 {
    let total: f32 = v.iter().sum();
    if total <= 0.0 { return f32::NAN; }
    v.iter().map(|x| (x / total).powi(2)).sum()
}

fn top_k_concentration(v: &[f32], k: usize) -> f32 {
    let total: f32 = v.iter().sum();
    if total <= 0.0 { return f32::NAN; }
    let mut sorted: Vec<f32> = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top_sum: f32 = sorted.iter().take(k).sum();
    top_sum / total
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let pairs: Vec<(f32, f32)> = a.iter().zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();
    let n = pairs.len();
    if n < 3 { return f32::NAN; }

    // 排名
    fn rank(v: &[f32]) -> Vec<f32> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0f32; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j < idx.len() && v[idx[j]] == v[idx[i]] { j += 1; }
            let avg = (i + j - 1) as f32 / 2.0;
            for k in i..j { ranks[idx[k]] = avg; }
            i = j;
        }
        ranks
    }
    let av: Vec<f32> = pairs.iter().map(|&(x, _)| x).collect();
    let bv: Vec<f32> = pairs.iter().map(|&(_, y)| y).collect();
    let ra = rank(&av);
    let rb = rank(&bv);
    let n_f = n as f32;
    let d2: f32 = ra.iter().zip(rb.iter()).map(|(ra, rb)| (ra - rb).powi(2)).sum();
    1.0 - 6.0 * d2 / (n_f * (n_f * n_f - 1.0))
}

// ============================================================
// v1 核心入口：读盘 → 计算 → (codes, vals)
// ============================================================

pub fn compute_hot_stock_pool_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction");
    if codes.is_empty() {
        return Ok((vec![], vec![]));
    }

    // ① rayon 并行读全市场逐笔 + per-stock 构建数据
    let stocks: Vec<Option<StockData>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            build_stock_data(code, date, &trades)
        })
        .collect();

    // 过滤有效股票
    let mut valid_stocks: Vec<StockData> = Vec::new();
    for (_code, s) in codes.iter().zip(stocks.into_iter()) {
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

    // ② 主循环：每秒计算分组、特征
    // 热/冰点分开累积：accum_{hot,cold}[pi][stock_i] = Vec<Vec<f32>>（z 组 40 维特征）
    type StockAccum = Vec<Vec<Vec<Vec<f32>>>>; // [N_PARAM_COMBOS][n_valid] → Vec<Vec<f32>>
    type InfoAccum = Vec<Vec<Vec<InclusionInfo>>>;
    type CoocAccum = Vec<Vec<FxHashMap<usize, u32>>>;

    let mut accum_hot: StockAccum = (0..N_PARAM_COMBOS).map(|_| vec![Vec::new(); n_valid]).collect();
    let mut accum_cold: StockAccum = (0..N_PARAM_COMBOS).map(|_| vec![Vec::new(); n_valid]).collect();
    let mut infos_hot: InfoAccum = (0..N_PARAM_COMBOS).map(|_| vec![Vec::new(); n_valid]).collect();
    let mut infos_cold: InfoAccum = (0..N_PARAM_COMBOS).map(|_| vec![Vec::new(); n_valid]).collect();
    let mut cooc_hot: CoocAccum = (0..N_PARAM_COMBOS).map(|_| vec![FxHashMap::default(); n_valid]).collect();
    let mut cooc_cold: CoocAccum = (0..N_PARAM_COMBOS).map(|_| vec![FxHashMap::default(); n_valid]).collect();

    // 历史状态：每个参数组合的上两秒组均值（用于 A06/A07, B06/B07）
    let mut prev_mean_br: [f32; N_PARAM_COMBOS] = [f32::NAN; N_PARAM_COMBOS];
    let mut prev2_mean_br: [f32; N_PARAM_COMBOS] = [f32::NAN; N_PARAM_COMBOS];
    let mut prev_mean_ba: [f32; N_PARAM_COMBOS] = [f32::NAN; N_PARAM_COMBOS];
    let mut prev2_mean_ba: [f32; N_PARAM_COMBOS] = [f32::NAN; N_PARAM_COMBOS];
    let mut prev_pool_codes: [Vec<String>; N_PARAM_COMBOS] = Default::default();
    let mut prev_sec: [usize; N_PARAM_COMBOS] = [usize::MAX; N_PARAM_COMBOS];

    // 用于 E01：维护每只股票每参数组合的连续在组内秒数
    let mut stay_seconds: Vec<[u32; N_PARAM_COMBOS]> = vec![[0u32; N_PARAM_COMBOS]; n_valid];

    // 每秒处理
    for sec in 15..ADJUSTED_SECONDS {
        for (pi, &(x, y, d_type)) in PARAM_CONFIGS.iter().enumerate() {
            if sec < x - 1 { continue; }
            let n_top = ((n_valid as f64) * y).ceil() as usize;
            if n_top < 2 { continue; }

            let ranking = compute_ranking_values(&valid_stocks, x, d_type, sec);
            if ranking.len() < n_top { continue; }

            let top_indices: Vec<usize> = ranking.iter().take(n_top).map(|&(i, _)| i).collect();
            let bottom_indices: Vec<usize> = ranking.iter().rev().take(n_top).map(|&(i, _)| i).collect();

            for (group_type, pool_idx) in [0usize, 1].iter().zip([&top_indices, &bottom_indices].iter()) {
                let pool_codes: Vec<String> = pool_idx.iter().map(|&i| valid_stocks[i].code.clone()).collect();
                let n_pool = pool_idx.len();

                // 组内滚动均值
                let sto_buy: Vec<f32> = pool_idx.iter().map(|&i| rolling_mean_at(&valid_stocks[i].secs, x, sec, 0)).collect();
                let sto_ba: Vec<f32> = pool_idx.iter().map(|&i| rolling_mean_at(&valid_stocks[i].secs, x, sec, 1)).collect();
                let sto_ret: Vec<f32> = pool_idx.iter().map(|&i| rolling_mean_at(&valid_stocks[i].secs, x, sec, 2)).collect();
                let sto_vol: Vec<f32> = pool_idx.iter().map(|&i| rolling_sum_at(&valid_stocks[i].secs, x, sec)).collect();

                let all_market = all_market_rolling_means(&valid_stocks, x, d_type, sec);
                let all_market_ba = all_market_rolling_means(&valid_stocks, x, 1, sec);

                // 计算组特征
                let mut group_feats = compute_group_features(
                    pool_idx, &valid_stocks, x, d_type, sec, &all_market,
                    None, &sto_buy, &sto_ba, &sto_ret, &sto_vol,
                );

                // A06: 一阶差分
                let cur_mean_br = mean(&sto_buy.iter().filter(|v| v.is_finite()).copied().collect::<Vec<f32>>());
                group_feats[5] = if prev_mean_br[pi].is_finite() && cur_mean_br.is_finite() {
                    cur_mean_br - prev_mean_br[pi]
                } else { f32::NAN };
                // A07: 二阶差分
                group_feats[6] = if prev_mean_br[pi].is_finite() && prev2_mean_br[pi].is_finite() && cur_mean_br.is_finite() {
                    (cur_mean_br - prev_mean_br[pi]) - (prev_mean_br[pi] - prev2_mean_br[pi])
                } else { f32::NAN };

                let cur_mean_ba = mean(&sto_ba.iter().filter(|v| v.is_finite()).copied().collect::<Vec<f32>>());
                group_feats[5 + 13] = if prev_mean_ba[pi].is_finite() && cur_mean_ba.is_finite() {
                    cur_mean_ba - prev_mean_ba[pi]
                } else { f32::NAN };
                group_feats[6 + 13] = if prev_mean_ba[pi].is_finite() && prev2_mean_ba[pi].is_finite() && cur_mean_ba.is_finite() {
                    (cur_mean_ba - prev_mean_ba[pi]) - (prev_mean_ba[pi] - prev2_mean_ba[pi])
                } else { f32::NAN };

                // B11: 超额 bid_ask
                let all_ba_mean = mean(&all_market_ba.iter().filter(|v| v.is_finite()).copied().collect::<Vec<f32>>());
                group_feats[13 + 10] = if all_ba_mean.is_finite() && cur_mean_ba.is_finite() {
                    cur_mean_ba - all_ba_mean
                } else { f32::NAN };

                // E01: 连续停留 ≥3秒的股票占比
                let mut stay_ge3 = 0usize;
                for &si in pool_idx.iter() {
                    if sec > 0 && prev_sec[pi] != usize::MAX && sec == prev_sec[pi] + 1 {
                        if pool_idx.contains(&si) { stay_seconds[si][pi] += 1; } else { stay_seconds[si][pi] = 0; }
                    } else {
                        stay_seconds[si][pi] = 1;
                    }
                    if stay_seconds[si][pi] >= 3 { stay_ge3 += 1; }
                }
                group_feats[33] = stay_ge3 as f32 / n_pool.max(1) as f32;

                // 选择 accum/cooc
                let (accum, infos, cooc) = if *group_type == 0 {
                    (&mut accum_hot, &mut infos_hot, &mut cooc_hot)
                } else {
                    (&mut accum_cold, &mut infos_cold, &mut cooc_cold)
                };

                // 为组内每只股票计算个股特征
                for (rank_i, &stock_i) in pool_idx.iter().enumerate() {
                    let rank_pct = rank_i as f32 / n_pool.max(1) as f32;
                    let mut per_stock = group_feats.clone();
                    let sb = sto_buy[rank_i];
                    let sba = sto_ba[rank_i];
                    let sv = sto_vol[rank_i];

                    // A08-A10, B08-B10, C03, A13, B13（个股相对位置）
                    fill_per_stock_features(&mut per_stock, &sto_buy, &sto_ba, &sto_vol, rank_i, sb, sba, sv);

                    // C04: 组总成交/全市场总成交
                    let pool_total_vol: f32 = sto_vol.iter().sum();
                    let mkt_total: f32 = valid_stocks.iter().map(|s| s.secs[sec].volume).sum();
                    per_stock[26 + 3] = if mkt_total > 0.0 { pool_total_vol / mkt_total } else { f32::NAN };

                    // InclusionInfo
                    let info = InclusionInfo {
                        second_idx: sec, rank_pct, pool_size: n_pool,
                        pool_codes: pool_codes.clone(),
                    };
                    let prev_info = infos[pi][stock_i].last();
                    let cont_feats = compute_continuity_features(&info, prev_info, &infos[pi][stock_i]);

                    let mut all_feats = Vec::with_capacity(FEAT_PER_INCLUSION);
                    all_feats.extend_from_slice(&per_stock);
                    all_feats.extend_from_slice(&cont_feats);
                    all_feats.resize(FEAT_PER_INCLUSION, f32::NAN);

                    accum[pi][stock_i].push(all_feats);
                    infos[pi][stock_i].push(info);

                    for &other_i in pool_idx.iter() {
                        if other_i != stock_i {
                            *cooc[pi][stock_i].entry(other_i).or_insert(0) += 1;
                        }
                    }
                }

                // 更新历史状态
                prev2_mean_br[pi] = prev_mean_br[pi];
                prev_mean_br[pi] = cur_mean_br;
                prev2_mean_ba[pi] = prev_mean_ba[pi];
                prev_mean_ba[pi] = cur_mean_ba;
                prev_pool_codes[pi] = pool_codes;
                prev_sec[pi] = sec;
            }
        }
    }

    // ③ 降维 + 共现因子
    let col_names: Vec<String> = (0..FEAT_PER_INCLUSION).map(|i| format!("f{i:02}")).collect();
    let feat_per_mat = features_per_group_n(FEAT_PER_INCLUSION);
    let mut all_factors: Vec<Vec<f32>> = vec![vec![f32::NAN; N_FACTORS]; n_valid];

    // 辅助函数：将一个 z×40 矩阵降维后写入 all_factors
    let reduce_and_write = |all_factors: &mut [Vec<f32>], stock_i: usize, offset: &mut usize,
                             inclusion_feats: &[Vec<f32>], col_names: &[String]| {
        let z = inclusion_feats.len();
        if z < 2 {
            for _ in 0..feat_per_mat {
                if *offset < N_FACTORS { all_factors[stock_i][*offset] = f32::NAN; *offset += 1; }
            }
            return;
        }
        let flat: Vec<f32> = inclusion_feats.iter().flat_map(|v| v.iter().copied()).collect();
        let arr = Array2::from_shape_vec((z, FEAT_PER_INCLUSION), flat)
            .unwrap_or_else(|_| Array2::zeros((0, FEAT_PER_INCLUSION)));
        let (vals, _) = get_features_factors_rust_full(&arr.view(), col_names, false);
        for &v in &vals {
            if *offset < N_FACTORS { all_factors[stock_i][*offset] = v; *offset += 1; }
        }
    };

    for stock_i in 0..n_valid {
        let mut offset = 0;
        // 8 矩阵：4 combo × (hot, cold)
        for pi in 0..N_PARAM_COMBOS {
            reduce_and_write(&mut all_factors, stock_i, &mut offset, &accum_hot[pi][stock_i], &col_names);
            reduce_and_write(&mut all_factors, stock_i, &mut offset, &accum_cold[pi][stock_i], &col_names);
        }
        // 对齐
        offset = 8 * feat_per_mat;

        // ④ 共现因子（步骤6+7）
        compute_cooccurrence_full(
            &valid_stocks, stock_i, &cooc_hot, &cooc_cold,
            &mut all_factors[stock_i], &mut offset,
        );
    }

    // ⑤ fan-out: 展开为 (codes, vals)
    let mut out_codes = Vec::with_capacity(n_valid);
    let mut out_vals = Vec::with_capacity(n_valid * N_FACTORS);
    for (stock_i, facs) in all_factors.iter().enumerate() {
        out_codes.push(valid_stocks[stock_i].code.clone());
        out_vals.extend_from_slice(facs);
    }

    Ok((out_codes, out_vals))
}

/// features_per_group 对 n 列的输出长度
fn features_per_group_n(n: usize) -> usize {
    21 * n + n * (n - 1) / 2
}

/// 计算共现因子（步骤6+7）：热点/冰点 → mean/std + 差 + 绝对值差
/// 输出顺序与 hot_stock_pool_names() 对齐：
/// 每 combo: 11 basic × 2 stat × 4 type = 88，共 4×88 = 352
fn compute_cooccurrence_full(
    stocks: &[StockData],
    stock_i: usize,
    cooc_hot: &[Vec<FxHashMap<usize, u32>>],
    cooc_cold: &[Vec<FxHashMap<usize, u32>>],
    out: &mut [f32],
    offset: &mut usize,
) {
    for pi in 0..N_PARAM_COMBOS {
        let hot_ms = top10_basic_stats(stocks, stock_i, &cooc_hot[pi][stock_i]);
        let cold_ms = top10_basic_stats(stocks, stock_i, &cooc_cold[pi][stock_i]);

        // 11 basic features × 2 (mean/std) × 4 (hot/cold/diff/abs_diff)
        for j in 0..BASIC_FEAT_N {
            // stat=mean (index j), stat=std (index BASIC_FEAT_N + j)
            for stat_idx in [j, BASIC_FEAT_N + j] {
                let hv = hot_ms[stat_idx];
                let cv = cold_ms[stat_idx];
                let vals = [
                    hv,  // hot
                    cv,  // cold
                    if hv.is_finite() && cv.is_finite() { hv - cv } else { f32::NAN },  // diff
                    if hv.is_finite() && cv.is_finite() { (hv - cv).abs() } else { f32::NAN },  // abs_diff
                ];
                for v in vals {
                    if *offset < N_FACTORS { out[*offset] = v; *offset += 1; }
                }
            }
        }
    }
}

/// 取共现次数最多的 10 只股票，计算它们 11 个基础特征的 mean 和 std
/// 返回 [11 means, 11 stds] = 22 个 f32
fn top10_basic_stats(
    stocks: &[StockData],
    stock_i: usize,
    cooc_map: &FxHashMap<usize, u32>,
) -> [f32; BASIC_FEAT_N * 2] {
    let mut result = [f32::NAN; BASIC_FEAT_N * 2];
    if cooc_map.is_empty() { return result; }

    let mut sorted: Vec<(&usize, &u32)> = cooc_map.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    let top10: Vec<usize> = sorted.iter().take(10).map(|(&i, _)| i).collect();

    let all_basic: Vec<&[f32; BASIC_FEAT_N]> = top10.iter()
        .filter_map(|&si| if si < stocks.len() { Some(&stocks[si].basic_feats) } else { None })
        .collect();

    if all_basic.is_empty() { return result; }

    for j in 0..BASIC_FEAT_N {
        let col: Vec<f32> = all_basic.iter().map(|f| f[j]).filter(|v| v.is_finite()).collect();
        if col.len() >= 2 {
            let m = col.iter().sum::<f32>() / col.len() as f32;
            let var = col.iter().map(|v| (v - m).powi(2)).sum::<f32>() / col.len() as f32;
            result[j] = m;
            result[BASIC_FEAT_N + j] = var.sqrt();
        }
    }
    result
}

/// 计算某只股票与其他股票的相关性最高的 top_k 只的均值
fn correlation_top_k(values: &[f32], _rets: &[f32], _stock_rank: usize, _all_values: &[f32], _k: usize) -> f32 {
    // 简化实现：直接用组内排序后相邻的 k 只股票的均值
    if values.is_empty() || values.len() < 2 { return f32::NAN; }
    let mut sorted: Vec<(usize, f32)> = values.iter().enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // 找到 stock_rank 在排序中的位置
    let pos = sorted.iter().position(|(i, _)| *i == _stock_rank).unwrap_or(0);
    let start = if pos >= _k / 2 { pos - _k / 2 } else { 0 };
    let end = (start + _k).min(sorted.len());
    let neighbor_vals: Vec<f32> = sorted[start..end].iter()
        .filter(|(i, _)| *i != _stock_rank)
        .map(|(_, v)| *v)
        .collect();
    if neighbor_vals.is_empty() { return f32::NAN; }
    neighbor_vals.iter().sum::<f32>() / neighbor_vals.len() as f32
}

// ============================================================
// 因子名
// ============================================================

pub fn hot_stock_pool_names() -> Vec<String> {
    let col_names: Vec<String> = (0..FEAT_PER_INCLUSION).map(|i| format!("f{i:02}")).collect();
    let dummy = Array2::zeros((2, FEAT_PER_INCLUSION));
    let (_, reduced_names) = get_features_factors_rust_full(&dummy.view(), &col_names, false);

    let combo_labels = ["x60y3_buy", "x60y3_ba", "x15y10_buy", "x15y10_ba"];
    let group_labels = ["hot", "cold"];

    let mut names = Vec::with_capacity(N_FACTORS);

    // 降维特征名
    for combo in &combo_labels {
        for grp in &group_labels {
            for n in &reduced_names {
                names.push(format!("{combo}_{grp}_{n}"));
            }
        }
    }

    // 共现因子名
    let basic_names = [
        "total_buy_ratio", "total_return", "ret_15s_std", "ret_60s_std",
        "buy_ratio_15s_std", "buy_ratio_60s_std", "bid_ask_15s_std", "bid_ask_60s_std",
        "total_volume", "vol_15s_std", "vol_60s_std",
    ];
    let stat_names = ["mean", "std"];
    let cooc_types = ["hot", "cold", "diff", "abs_diff"];

    for combo in &combo_labels {
        for b in &basic_names {
            for s in &stat_names {
                for t in &cooc_types {
                    names.push(format!("cooc_{combo}_{b}_{s}_{t}"));
                }
            }
        }
    }

    assert_eq!(names.len(), N_FACTORS, "names.len()={} != N_FACTORS={}", names.len(), N_FACTORS);
    names
}

// ============================================================
// PyO3 入口
// ============================================================

#[pyfunction]
pub fn py_hot_stock_pool(_py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_hot_stock_pool_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_hot_stock_pool_names() -> Vec<String> {
    hot_stock_pool_names()
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_count() {
        let names = hot_stock_pool_names();
        assert_eq!(names.len(), N_FACTORS);
    }

    #[test]
    fn test_mean_std() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-6);
        assert!((std(&[1.0, 2.0, 3.0]) - 0.8164966).abs() < 1e-4);
    }

    #[test]
    fn test_herfindahl() {
        let h = herfindahl(&[1.0, 1.0, 1.0]);
        assert!((h - 1.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduced_per_combo() {
        assert_eq!(REDUCED_PER_COMBO, features_per_group_n(FEAT_PER_INCLUSION));
    }

    #[test]
    fn test_n_factors() {
        assert_eq!(N_FACTORS, 8 * REDUCED_PER_COMBO + COOCCUR_FACTORS);
    }
}
