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
use crate::features::{get_features_factors_rust_full, get_features_factors_rust_values_only};
use chrono::NaiveDate;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashSet;
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

/// 秒槽位起始偏移（epoch秒对应当天零点）
/// adjust_afternoon 后，交易时间映射到 [34200, 48420] 连续区间
const SEC_OFFSET: i64 = 9 * 3600 + 30 * 60; // 34200

/// 上午结束 epoch（调整后）
const MORNING_END: i64 = 11 * 3600 + 30 * 60; // 41400
/// 下午开始 epoch（调整后，原13:00前移90分）
const AFTERNOON_START: i64 = MORNING_END + 1; // 41401
/// 下午结束 epoch（调整后，原14:57前移90分）
const AFTERNOON_END: i64 = MORNING_END + (14 * 3600 + 57 * 60 - 13 * 3600); // 48420
/// 调整后交易秒数：上午 7201 (09:30-11:30 含) + 下午 7020 (11:30:01-13:27 含) = 14221
const ADJUSTED_SECONDS: usize = ((MORNING_END - SEC_OFFSET + 1) + (AFTERNOON_END - AFTERNOON_START + 1)) as usize;

/// 每秒处理的步长
const SECOND_STEP: usize = 2;

/// 每次入选特征矩阵的最大行数（z）。超过则均匀采样，降低降维 O(z²) 开销。
const MAX_Z: usize = 50;

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

/// 简易 bitset：用于 prev_pool_set 的快速构建和查找
struct PoolBitset {
    bits: Vec<u64>,
    n_valid: usize,
}

impl PoolBitset {
    fn new(n_valid: usize) -> Self {
        Self { bits: vec![0u64; (n_valid + 63) / 64], n_valid }
    }
    #[inline(always)]
    fn set(&mut self, idx: usize) {
        self.bits[idx / 64] |= 1u64 << (idx % 64);
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> bool {
        (self.bits[idx / 64] & (1u64 << (idx % 64))) != 0
    }
    fn clear(&mut self) {
        for w in self.bits.iter_mut() { *w = 0; }
    }
    fn build_from(&mut self, indices: &[usize]) {
        self.clear();
        for &i in indices { self.set(i); }
    }
    /// 与另一个 bitset 的交集大小
    fn intersection_count(&self, other: &Self) -> usize {
        self.bits.iter().zip(other.bits.iter())
            .map(|(&a, &b)| (a & b).count_ones() as usize)
            .sum()
    }
}

/// 单股预计算数据
struct StockData {
    code: String,
    secs: Vec<SecStat>,
    // 共现用的 11 个基础特征
    basic_feats: [f32; BASIC_FEAT_N],
}

/// 预计算的滚动窗口缓存：为每只股票预计算 8 组滚动均值（4 字段 × 2 窗口）
/// 字段顺序: [buy_15, buy_60, ba_15, ba_60, ret_15, ret_60, vol_15, vol_60]
/// 布局: rolling[field * ADJUSTED_SECONDS + sec]
struct RollingCache {
    data: Vec<f32>, // 8 * ADJUSTED_SECONDS 长度
}

impl RollingCache {
    #[inline]
    fn get(&self, field: usize, sec: usize) -> f32 {
        self.data[field * ADJUSTED_SECONDS + sec]
    }
    /// 根据窗口 x 和 field_type 自动选择正确的缓存列
    /// field_type: 0=buy_ratio, 1=bid_ask, 2=ret, 3=volume
    #[inline]
    fn get_by_x(&self, x: usize, field_type: u8, sec: usize) -> f32 {
        let base = match (field_type, x) {
            (0, 15) => 0, (0, 60) => 1,
            (1, 15) => 2, (1, 60) => 3,
            (2, 15) => 4, (2, 60) => 5,
            (3, 15) => 6, (3, 60) => 7,
            _ => return f32::NAN,
        };
        self.data[base * ADJUSTED_SECONDS + sec]
    }

    /// 增量式计算所有 8 组滚动均值（单 pass O(n)，替代 O(x) 重复扫描）
    fn compute(secs: &[SecStat]) -> Self {
        let n = ADJUSTED_SECONDS;
        let total = 8 * n;
        let mut data = vec![f32::NAN; total];

        // 为每个 field×window 组合计算滚动值
        // 布局: [buy_15, buy_60, ba_15, ba_60, ret_15, ret_60, vol_15, vol_60]
        let configs: [(usize, usize, fn(&SecStat) -> f32, bool); 8] = [
            (0, 15, |s: &SecStat| s.buy_ratio, false),     // buy_15
            (1, 60, |s: &SecStat| s.buy_ratio, false),     // buy_60
            (2, 15, |s: &SecStat| s.bid_ask_mean, false),  // ba_15
            (3, 60, |s: &SecStat| s.bid_ask_mean, false),  // ba_60
            (4, 15, |s: &SecStat| s.ret_val, false),       // ret_15
            (5, 60, |s: &SecStat| s.ret_val, false),       // ret_60
            (6, 15, |s: &SecStat| s.volume, true),         // vol_15 (sum)
            (7, 60, |s: &SecStat| s.volume, true),         // vol_60 (sum)
        ];
        for &(col, win, getter, is_sum) in &configs {
            let base = col * n;
            let mut sum: f64 = 0.0;
            let mut cnt: u32 = 0;
            for sec in 0..n {
                let v = getter(&secs[sec]);
                if v.is_finite() { sum += v as f64; cnt += 1; }
                if sec >= win {
                    let old = getter(&secs[sec - win]);
                    if old.is_finite() { sum -= old as f64; cnt -= 1; }
                }
                if cnt > 0 && sec >= win - 1 {
                    data[base + sec] = if is_sum { sum as f32 } else { (sum / cnt as f64) as f32 };
                }
            }
        }
        Self { data }
    }
}

/// 某次入选时该股的特征快照（最小化存储，连续性即时计算）
#[derive(Clone, Copy)]
struct InclusionInfo {
    second_idx: usize,
    rank_pct: f32,
    pool_size: usize,
    cont_f01: f32, cont_f02: f32, cont_f03: f32,
    cont_f04: f32, cont_f05: f32, cont_f06: f32,
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

/// epoch秒 → 数组索引。epoch 是 time_sec (UTC_epoch + 28800)。
/// day_midnight_cst 通过 cst_midnight_epoch(date) 获得。
/// 相减得到日内绝对秒（09:30 = 34200），再映射到连续数组索引。
#[inline]
fn sec_to_idx(epoch: f32, day_midnight_cst: i64) -> Option<usize> {
    let e = (epoch as i64) - day_midnight_cst;
    if e < SEC_OFFSET || e > AFTERNOON_END {
        return None;
    }
    if e <= MORNING_END {
        Some((e - SEC_OFFSET) as usize)
    } else {
        // e >= AFTERNOON_START (交易时段连续，中间只有午休但 adjust 后已消除)
        Some((MORNING_END - SEC_OFFSET + 1 + e - AFTERNOON_START) as usize)
    }
}

/// 计算 date (YYYYMMDD) 对应的 CST 零点在 time_sec 中的值。
/// read_trade_fast_inner 输出的 time_sec = UTC_epoch + 28800。
/// CST 00:00:00 = UTC 前一日 16:00:00, time_sec = (days_since_epoch - 1)*86400 + 16*3600 + 28800 = days_since_epoch * 86400
fn cst_midnight_epoch(date: i64) -> i64 {
    let year = (date / 10000) as i32;
    let month = ((date / 100) % 100) as u32;
    let day = (date % 100) as u32;
    let nd = NaiveDate::from_ymd_opt(year, month, day).expect("invalid date");
    // nd.and_hms_opt(0,0,0) gives NaiveDateTime at UTC midnight
    // .and_utc().timestamp() gives Unix epoch seconds for UTC midnight
    // CST midnight time_sec = UTC_midnight_epoch + 0 (since time_sec already has +28800,
    // and CST midnight = UTC previous-day 16:00, the math simplifies)
    nd.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp()
}

// ============================================================
// Per-stock 数据准备
// ============================================================

/// 与某只股票排名相邻的 k 只股票的值均值（O(n_pool)，近似替代 Pearson 相关 top-k）
fn neighbor_mean(values: &[f32], stock_rank: usize, k: usize) -> f32 {
    if values.len() < 2 { return f32::NAN; }
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

/// neighbor_mean 无堆分配版本（内联到热路径）
#[inline(always)]
fn neighbor_mean_inline(values: &[f32], stock_rank: usize, k: usize) -> f32 {
    let n = values.len();
    if n < 2 { return f32::NAN; }
    let start = if stock_rank >= k / 2 { stock_rank - k / 2 } else { 0 };
    let end = (start + k).min(n);
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    for i in start..end {
        if i != stock_rank && values[i].is_finite() {
            sum += values[i];
            cnt += 1;
        }
    }
    if cnt == 0 { f32::NAN } else { sum / cnt as f32 }
}

/// 在已排序数组中计算值的排名百分位
fn rank_pct_in(sorted: &[f32], val: f32) -> f32 {
    if sorted.is_empty() || !val.is_finite() { return f32::NAN; }
    let pos = sorted.partition_point(|&v| v < val);
    pos as f32 / sorted.len().max(1) as f32
}

/// 均值 + 标准差（接受已 filter finite 的数组）
fn mean_std(v: &[f32]) -> (f32, f32) {
    let n = v.len();
    if n < 2 { return (f32::NAN, f32::NAN); }
    let m = v.iter().sum::<f32>() / n as f32;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
    (m, var.sqrt())
}

/// 从逐笔成交构建 per-stock 每秒统计 + 基础特征（优化版，单 pass）
fn build_stock_data(code: &str, date: i64, trades: &[crate::fast_csv_reader::TradeRecord]) -> Option<StockData> {
    let n_secs = ADJUSTED_SECONDS;
    let day_mid = cst_midnight_epoch(date);
    let mut buy_vol = vec![0.0f64; n_secs];
    let mut total_vol = vec![0.0f64; n_secs];
    let mut bid_ask_sum = vec![0.0f64; n_secs];
    let mut bid_ask_cnt = vec![0u32; n_secs];
    let mut first_prices = vec![0.0f32; n_secs];
    let mut last_prices = vec![0.0f32; n_secs];
    let mut has_data = vec![false; n_secs];

    for t in trades {
        let idx = match sec_to_idx(t.time_sec, day_mid) {
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

/// 单次遍历全市场，同时选出 top-k/bottom-k 并返回全市场 buy_ratio 和 bid_ask 数组
fn select_top_bottom_full(
    caches: &[RollingCache],
    x: usize,
    d_type: usize,
    sec: usize,
    k: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f32>, Vec<f32>) {
    // 返回 (top, bottom, all_d_vals, all_ba_vals)
    let d_field: u8 = if d_type == 0 { 0 } else { 1 };
    let n = caches.len();
    let mut all_d = vec![f32::NAN; n];
    let mut all_ba = vec![f32::NAN; n];
    let mut valid: Vec<(usize, f32)> = Vec::with_capacity(n);
    for (si, cache) in caches.iter().enumerate() {
        let dv = cache.get_by_x(x, d_field, sec);
        let bv = cache.get_by_x(x, 1, sec);
        all_d[si] = dv;
        all_ba[si] = bv;
        if dv.is_finite() { valid.push((si, dv)); }
    }
    if valid.len() < k * 2 {
        return (Vec::new(), Vec::new(), all_d, all_ba);
    }
    let k_adj = k.min(valid.len().saturating_sub(1));
    let (top_part, _, bottom_part) = valid.select_nth_unstable_by(k_adj, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top_sorted: Vec<(usize, f32)> = top_part.to_vec();
    top_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut bot = bottom_part.to_vec();
    let bot_k = k.min(bot.len());
    let ( _, _, bot_bot) = if bot_k > 0 {
        bot.select_nth_unstable_by(bot_k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    } else {
        return (top_sorted.iter().map(|(i, _)| *i).collect(), Vec::new(), all_d, all_ba);
    };
    let mut bot_sorted: Vec<(usize, f32)> = bot_bot.to_vec();
    bot_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    (top_sorted.iter().map(|(i, _)| *i).collect(),
     bot_sorted.iter().map(|(i, _)| *i).collect(),
     all_d, all_ba)
}

/// 组特征 34 维 — 栈数组版本（无堆分配）
#[allow(clippy::too_many_arguments)]
fn build_group_features_arr(
    br_finite: &[f32], ba_finite: &[f32], ret_finite: &[f32], vol_finite: &[f32],
    sto_buy: &[f32], sto_ba: &[f32], sto_ret: &[f32],
    br_sorted: &[f32], ba_sorted: &[f32],
    mean_br: f32, mean_ba: f32, mkt_mean_d: f32, mkt_mean_ba: f32,
) -> [f32; 34] {
    let mut f = [f32::NAN; 34];
    // A01-A05
    f[0] = mean_br;
    f[1] = std(br_finite);
    f[2] = skew(br_finite);
    f[3] = kurtosis(br_finite);
    f[4] = if br_sorted.len() >= 2 { percentile_sorted(&br_sorted, 0.90) - percentile_sorted(&br_sorted, 0.10) } else { f32::NAN };
    // A06-A07 占位 (5,6)
    // A08-A10 占位 (7,8,9)
    // A11
    f[10] = if mkt_mean_d.is_finite() && mean_br.is_finite() { mean_br - mkt_mean_d } else { f32::NAN };
    // A12: 截面相关（Pearson 近似，无堆分配）
    f[11] = corr_fast(sto_buy, sto_ret);
    // A13 占位 (12)

    // B01-B05
    f[13] = mean_ba;
    f[14] = std(ba_finite);
    f[15] = skew(ba_finite);
    f[16] = kurtosis(ba_finite);
    f[17] = if ba_sorted.len() >= 2 { percentile_sorted(&ba_sorted, 0.90) - percentile_sorted(&ba_sorted, 0.10) } else { f32::NAN };
    // B06-B07 占位 (18,19)
    // B08-B10 占位 (20,21,22)
    // B11
    f[23] = if mkt_mean_ba.is_finite() && mean_ba.is_finite() { mean_ba - mkt_mean_ba } else { f32::NAN };
    // B12: 截面相关（Pearson 近似）
    f[24] = corr_fast(sto_ba, sto_ret);
    // B13 占位 (25)

    // C01-C04
    f[26] = herfindahl(vol_finite);
    f[27] = top_k_concentration(vol_finite, 3);
    // C03 占位 (28), C04 占位 (29)

    // D01-D03
    f[30] = mean(ret_finite);
    f[31] = std(ret_finite);
    f[32] = skew(ret_finite);

    // E01 占位 (33)
    f
}

/// Pearson 相关（无堆分配，单 pass O(n)）。用于 A12/B12 近似替代 Spearman。
#[inline]
fn corr_fast(a: &[f32], b: &[f32]) -> f32 {
    let mut n: i32 = 0;
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    let mut saa = 0.0f64;
    let mut sbb = 0.0f64;
    let mut sab = 0.0f64;
    for i in 0..a.len() {
        let av = a[i];
        let bv = b[i];
        if av.is_finite() && bv.is_finite() {
            n += 1;
            let avd = av as f64;
            let bvd = bv as f64;
            sa += avd; sb += bvd;
            saa += avd * avd; sbb += bvd * bvd;
            sab += avd * bvd;
        }
    }
    if n < 3 { return f32::NAN; }
    let nf = n as f64;
    let num = sab - sa * sb / nf;
    let den = ((saa - sa * sa / nf) * (sbb - sb * sb / nf)).sqrt();
    if den < 1e-15 { return f32::NAN; }
    (num / den) as f32
}
fn q90_q10_sorted(v: &[f32]) -> f32 {
    let mut s: Vec<f32> = v.iter().filter(|x| x.is_finite()).copied().collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if s.len() < 2 { return f32::NAN; }
    percentile_sorted(&s, 0.90) - percentile_sorted(&s, 0.10)
}

/// 填充 A08-A10, B08-B10, C03 — 栈数组版本
#[allow(clippy::too_many_arguments)]
fn fill_per_stock_arr(
    feats: &mut [f32; 34],
    br_sorted: &[f32], ba_sorted: &[f32], vol_sorted: &[f32],
    sb: f32, sba: f32, sv: f32,
    br_m: f32, br_s: f32, ba_m: f32, ba_s: f32,
) {
    feats[7] = rank_pct_in(br_sorted, sb);
    feats[8] = if br_s > 1e-12 && sb.is_finite() { (sb - br_m) / br_s } else { f32::NAN };
    feats[9] = if sb.is_finite() && !br_sorted.is_empty() { sb - br_sorted[br_sorted.len() / 2] } else { f32::NAN };
    feats[20] = rank_pct_in(ba_sorted, sba);
    feats[21] = if ba_s > 1e-12 && sba.is_finite() { (sba - ba_m) / ba_s } else { f32::NAN };
    feats[22] = if sba.is_finite() && !ba_sorted.is_empty() { sba - ba_sorted[ba_sorted.len() / 2] } else { f32::NAN };
    feats[28] = rank_pct_in(vol_sorted, sv);
}

// ============================================================
// 步骤2+3：计算入选特征（40维）
// ============================================================


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
    let t1 = std::time::Instant::now();
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
    let n_valid = valid_stocks.len();
    if n_valid == 0 {
        return Ok((vec![], vec![]));
    }

    // 预计算每只股票的滚动窗口缓存
    let rolling_caches: Vec<RollingCache> = valid_stocks
        .par_iter()
        .map(|sd| RollingCache::compute(&sd.secs))
        .collect();


    // ② 主循环
    // ② 主循环：按参数组合并行（4 组合互相独立）
    // 优化：stack arrays + 延迟共现 + 预分配 buffer
    type Feat40 = [f32; FEAT_PER_INCLUSION];
    type StockAccumPi = Vec<Vec<Feat40>>;          // [n_valid] → Vec<[f32;40]>
    // pool_log: 每个 (pi, group_type) 的所有 pool 成员列表
    type PoolLog = Vec<Vec<usize>>;                // list of pools, each pool is Vec<usize>

    struct PiResult {
        accum_hot: StockAccumPi,
        accum_cold: StockAccumPi,
        pool_log_hot: PoolLog,
        pool_log_cold: PoolLog,
    }

    let pi_results: Vec<PiResult> = PARAM_CONFIGS.par_iter().map(|&(x, y, d_type)| {
        let n_top = ((n_valid as f64) * y).ceil() as usize;
        let mut accum_hot: StockAccumPi = vec![Vec::new(); n_valid];
        let mut accum_cold: StockAccumPi = vec![Vec::new(); n_valid];
        let mut pool_log_hot: PoolLog = Vec::with_capacity(ADJUSTED_SECONDS / SECOND_STEP);
        let mut pool_log_cold: PoolLog = Vec::with_capacity(ADJUSTED_SECONDS / SECOND_STEP);
        // infos 只需 last（连续性即时计算），用 per-stock Option<InclusionInfo>
        let mut last_info_hot: Vec<Option<InclusionInfo>> = vec![None; n_valid];
        let mut last_info_cold: Vec<Option<InclusionInfo>> = vec![None; n_valid];

        let mut prev_mean_br = [f32::NAN; 2];
        let mut prev2_mean_br = [f32::NAN; 2];
        let mut prev_mean_ba = [f32::NAN; 2];
        let mut prev2_mean_ba = [f32::NAN; 2];
        let mut prev_pool_set: [PoolBitset; 2] = [
            PoolBitset::new(n_valid), PoolBitset::new(n_valid),
        ];
        let mut prev_sec = [usize::MAX; 2];
        let mut stay_seconds = vec![[0u32; 2]; n_valid];

        // 可复用的 buffer（避免每步 alloc）
        let mut buf_all_vals_d = vec![f32::NAN; n_valid];
        let mut buf_all_ba_vals = vec![f32::NAN; n_valid];

        for sec in (15..ADJUSTED_SECONDS).step_by(SECOND_STEP) {
            if sec < x - 1 { continue; }
            if n_top < 2 { continue; }

            // 单次遍历全市场，填 buf 并选出 top/bottom
            let d_field: u8 = if d_type == 0 { 0 } else { 1 };
            let mut valid_pairs: Vec<(usize, f32)> = Vec::with_capacity(n_valid);
            for (si, cache) in rolling_caches.iter().enumerate() {
                let dv = cache.get_by_x(x, d_field, sec);
                let bv = cache.get_by_x(x, 1, sec);
                buf_all_vals_d[si] = dv;
                buf_all_ba_vals[si] = bv;
                if dv.is_finite() { valid_pairs.push((si, dv)); }
            }
            if valid_pairs.len() < n_top * 2 { continue; }

            // select_nth_unstable for top-k
            let k_adj = n_top.min(valid_pairs.len().saturating_sub(1));
            let (top_part, _, bottom_part) = valid_pairs.select_nth_unstable_by(k_adj, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut top_indices: Vec<usize> = top_part.iter().map(|(i, _)| *i).collect();
            top_indices.sort_by(|a, b| {
                let va = rolling_caches[*a].get_by_x(x, d_field, sec);
                let vb = rolling_caches[*b].get_by_x(x, d_field, sec);
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut bot = bottom_part.to_vec();
            let bot_k = n_top.min(bot.len());
            let (_, _, bot_bot) = bot.select_nth_unstable_by(bot_k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut bottom_indices: Vec<usize> = bot_bot.iter().map(|(i, _)| *i).collect();
            bottom_indices.sort_by(|a, b| {
                let va = rolling_caches[*a].get_by_x(x, d_field, sec);
                let vb = rolling_caches[*b].get_by_x(x, d_field, sec);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            });

            // 全市场总量
            let mkt_total_vol: f32 = (0..n_valid).map(|i| valid_stocks[i].secs[sec].volume).sum();
            let mkt_mean_d = { let vf: Vec<f32> = buf_all_vals_d.iter().copied().filter(|v| v.is_finite()).collect(); mean(&vf) };
            let mkt_mean_ba = { let vf: Vec<f32> = buf_all_ba_vals.iter().copied().filter(|v| v.is_finite()).collect(); mean(&vf) };

            for (gt, pool_idx) in [0usize, 1].iter().zip([&top_indices, &bottom_indices].iter()) {
                let gt = *gt;
                let n_pool = pool_idx.len();
                if n_pool == 0 { continue; }

                let sto_buy: Vec<f32> = pool_idx.iter().map(|&i| rolling_caches[i].get_by_x(x, 0, sec)).collect();
                let sto_ba: Vec<f32> = pool_idx.iter().map(|&i| rolling_caches[i].get_by_x(x, 1, sec)).collect();
                let sto_ret: Vec<f32> = pool_idx.iter().map(|&i| rolling_caches[i].get_by_x(x, 2, sec)).collect();
                let sto_vol: Vec<f32> = pool_idx.iter().map(|&i| rolling_caches[i].get_by_x(x, 3, sec)).collect();

                let br_finite: Vec<f32> = sto_buy.iter().copied().filter(|v| v.is_finite()).collect();
                let ba_finite: Vec<f32> = sto_ba.iter().copied().filter(|v| v.is_finite()).collect();
                let ret_finite: Vec<f32> = sto_ret.iter().copied().filter(|v| v.is_finite()).collect();
                let vol_finite: Vec<f32> = sto_vol.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();

                let mean_br = mean(&br_finite);
                let mean_ba = mean(&ba_finite);
                let pool_total_vol: f32 = sto_vol.iter().sum();
                let c04_val = if mkt_total_vol > 0.0 { pool_total_vol / mkt_total_vol } else { f32::NAN };

                // Pre-sort for q90_q10 and fill_per_stock (reuse across all stocks in pool)
                let mut br_sorted: Vec<f32> = br_finite.clone();
                br_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mut ba_sorted: Vec<f32> = ba_finite.clone();
                ba_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let vol_sorted: Vec<f32> = { let mut v = vol_finite.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); v };
                let (br_m, br_s) = mean_std(&br_sorted);
                let (ba_m, ba_s) = mean_std(&ba_sorted);

                // group_feats: 栈数组 [f32; 34]
                let mut group_feats: [f32; 34] = build_group_features_arr(
                    &br_finite, &ba_finite, &ret_finite, &vol_finite,
                    &sto_buy, &sto_ba, &sto_ret, &br_sorted, &ba_sorted,
                    mean_br, mean_ba, mkt_mean_d, mkt_mean_ba,
                );

                // A06/A07/B06/B07 差分
                group_feats[5] = if prev_mean_br[gt].is_finite() && mean_br.is_finite() { mean_br - prev_mean_br[gt] } else { f32::NAN };
                group_feats[6] = if prev_mean_br[gt].is_finite() && prev2_mean_br[gt].is_finite() && mean_br.is_finite() {
                    (mean_br - prev_mean_br[gt]) - (prev_mean_br[gt] - prev2_mean_br[gt]) } else { f32::NAN };
                group_feats[18] = if prev_mean_ba[gt].is_finite() && mean_ba.is_finite() { mean_ba - prev_mean_ba[gt] } else { f32::NAN };
                group_feats[19] = if prev_mean_ba[gt].is_finite() && prev2_mean_ba[gt].is_finite() && mean_ba.is_finite() {
                    (mean_ba - prev_mean_ba[gt]) - (prev_mean_ba[gt] - prev2_mean_ba[gt]) } else { f32::NAN };

                // E01 + 连续性 overlap（bitset 交集，O(n_valid/64)）
                let grp_overlap = if prev_sec[gt] != usize::MAX {
                    let mut cur_bs = PoolBitset::new(n_valid);
                    for &si in pool_idx.iter() { cur_bs.set(si); }
                    let ov = cur_bs.intersection_count(&prev_pool_set[gt]);
                    // stay_seconds 更新
                    for &si in pool_idx.iter() {
                        if sec == prev_sec[gt] + SECOND_STEP {
                            if prev_pool_set[gt].get(si) { stay_seconds[si][gt] += 1; } else { stay_seconds[si][gt] = 0; }
                        } else { stay_seconds[si][gt] = 1; }
                    }
                    ov
                } else {
                    for &si in pool_idx.iter() { stay_seconds[si][gt] = 1; }
                    0usize
                };
                let stay_ge3 = pool_idx.iter().filter(|&&si| stay_seconds[si][gt] >= 3).count();
                group_feats[33] = stay_ge3 as f32 / n_pool.max(1) as f32;

                let (accum, last_info, pool_log) = if gt == 0 {
                    (&mut accum_hot, &mut last_info_hot, &mut pool_log_hot)
                } else {
                    (&mut accum_cold, &mut last_info_cold, &mut pool_log_cold)
                };

                // 记录 pool 到 pool_log（用于后续共现重建）
                let pool_id = pool_log.len();
                pool_log.push(pool_idx.to_vec());

                // 连续性：overlap 对全组相同，只算一次
                let (grp_overlap, grp_gap, grp_prev_rank_avail) = if prev_sec[gt] != usize::MAX {
                    (grp_overlap, sec - prev_sec[gt], true)
                } else {
                    (0usize, 0usize, false)
                };

                for (rank_i, &stock_i) in pool_idx.iter().enumerate() {
                    let rank_pct = rank_i as f32 / n_pool.max(1) as f32;
                    let mut per_stock = group_feats;
                    fill_per_stock_arr(&mut per_stock, &br_sorted, &ba_sorted, &vol_sorted,
                                       sto_buy[rank_i], sto_ba[rank_i], sto_vol[rank_i], br_m, br_s, ba_m, ba_s);
                    per_stock[12] = neighbor_mean_inline(&sto_buy, rank_i, 3);
                    per_stock[25] = neighbor_mean_inline(&sto_ba, rank_i, 3);
                    per_stock[29] = c04_val;

                    // 连续性（使用组级别预计算的 overlap）
                    let (cf01, cf02, cf03, cf04, cf05) = if grp_prev_rank_avail {
                        if let Some(prev) = &last_info[stock_i] {
                            let denom = std::cmp::min(n_pool, prev.pool_size).max(1);
                            (grp_overlap as f32, grp_overlap as f32 / denom as f32,
                             rank_pct - prev.rank_pct,
                             if grp_gap == SECOND_STEP { 2.0 } else { 1.0 }, grp_gap as f32)
                        } else {
                            (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN)
                        }
                    } else {
                        (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN)
                    };
                    let info = InclusionInfo {
                        second_idx: sec, rank_pct, pool_size: n_pool,
                        cont_f01: cf01, cont_f02: cf02, cont_f03: cf03,
                        cont_f04: cf04, cont_f05: cf05, cont_f06: f32::NAN,
                    };

                    // 组装 [f32; 40] 栈数组
                    let mut all_feats: Feat40 = [f32::NAN; FEAT_PER_INCLUSION];
                    all_feats[..34].copy_from_slice(&per_stock);
                    all_feats[34] = cf01; all_feats[35] = cf02; all_feats[36] = cf03;
                    all_feats[37] = cf04; all_feats[38] = cf05; all_feats[39] = f32::NAN;

                    accum[stock_i].push(all_feats);
                    last_info[stock_i] = Some(info);
                }

                prev2_mean_br[gt] = prev_mean_br[gt]; prev_mean_br[gt] = mean_br;
                prev2_mean_ba[gt] = prev_mean_ba[gt]; prev_mean_ba[gt] = mean_ba;
                prev_pool_set[gt].build_from(pool_idx);
                prev_sec[gt] = sec;
            }
        }

        PiResult { accum_hot, accum_cold, pool_log_hot, pool_log_cold }
    }).collect();

    // 提取结果
    let all_accum_hot: Vec<StockAccumPi> = pi_results.iter().map(|r| r.accum_hot.clone()).collect();
    let all_accum_cold: Vec<StockAccumPi> = pi_results.iter().map(|r| r.accum_cold.clone()).collect();
    let all_pool_log_hot: Vec<PoolLog> = pi_results.iter().map(|r| r.pool_log_hot.clone()).collect();
    let all_pool_log_cold: Vec<PoolLog> = pi_results.iter().map(|r| r.pool_log_cold.clone()).collect();

    // ③ 降维 + 共现因子（rayon 并行，per-stock 独立计算）
    let col_names: Vec<String> = (0..FEAT_PER_INCLUSION).map(|i| format!("f{i:02}")).collect();
    let feat_per_mat = features_per_group_n(FEAT_PER_INCLUSION);

    // 预构建倒排索引：stock → pool_ids（per pi × hot/cold）
    // stock_pool_hot[pi][stock_i] = Vec<pool_id>
    let stock_pool_hot: Vec<Vec<Vec<usize>>> = all_pool_log_hot.iter().map(|pool_log| {
        let mut idx: Vec<Vec<usize>> = vec![Vec::new(); n_valid];
        for (pool_id, members) in pool_log.iter().enumerate() {
            for &si in members { idx[si].push(pool_id); }
        }
        idx
    }).collect();
    let stock_pool_cold: Vec<Vec<Vec<usize>>> = all_pool_log_cold.iter().map(|pool_log| {
        let mut idx: Vec<Vec<usize>> = vec![Vec::new(); n_valid];
        for (pool_id, members) in pool_log.iter().enumerate() {
            for &si in members { idx[si].push(pool_id); }
        }
        idx
    }).collect();

    let all_factors: Vec<Vec<f32>> = (0..n_valid).into_par_iter().map(|stock_i| {
        let mut facs = vec![f32::NAN; N_FACTORS];
        let mut offset = 0usize;

        // 可复用的 flat cooc 数组（避免 HashMap 开销）
        let mut cooc_buf = vec![0u32; n_valid];

        // 降维
        for pi in 0..N_PARAM_COMBOS {
            for accum in [&all_accum_hot[pi][stock_i], &all_accum_cold[pi][stock_i]] {
                let z_full = accum.len();
                if z_full == 0 { offset += feat_per_mat; continue; }
                let flat: Vec<f32> = if z_full > MAX_Z {
                    let step = z_full as f64 / MAX_Z as f64;
                    (0..MAX_Z).flat_map(|i| {
                        let idx = (i as f64 * step) as usize;
                        accum[idx].iter().copied()
                    }).collect()
                } else {
                    accum.iter().flat_map(|v| v.iter().copied()).collect()
                };
                let nrows = if z_full > MAX_Z { MAX_Z } else { z_full };
                let arr = Array2::from_shape_vec((nrows, FEAT_PER_INCLUSION), flat)
                    .unwrap_or_else(|_| Array2::zeros((0, FEAT_PER_INCLUSION)));
                let vals = get_features_factors_rust_values_only(&arr.view(), false);
                for &v in &vals { if offset < N_FACTORS { facs[offset] = v; offset += 1; } }
            }
        }
        offset = 8 * feat_per_mat;

        // 共现因子：flat array 替代 HashMap（消除 hash 开销）
        let mut cooc_offset = offset;
        for pi in 0..N_PARAM_COMBOS {
            // hot cooc — flat array
            for &pool_id in &stock_pool_hot[pi][stock_i] {
                for &other in &all_pool_log_hot[pi][pool_id] {
                    cooc_buf[other] += 1;
                }
            }
            // 找 top-10
            let mut pairs_h: Vec<(usize, u32)> = cooc_buf.iter().enumerate()
                .filter(|&(si, &cnt)| si != stock_i && cnt > 0)
                .map(|(si, &cnt)| (si, cnt)).collect();
            pairs_h.sort_by(|a, b| b.1.cmp(&a.1));
            let top10_h: Vec<usize> = pairs_h.iter().take(10).map(|(si, _)| *si).collect();
            // clear touched entries
            for &(si, _) in &pairs_h { cooc_buf[si] = 0; }

            // cold cooc
            for &pool_id in &stock_pool_cold[pi][stock_i] {
                for &other in &all_pool_log_cold[pi][pool_id] {
                    cooc_buf[other] += 1;
                }
            }
            let mut pairs_c: Vec<(usize, u32)> = cooc_buf.iter().enumerate()
                .filter(|&(si, &cnt)| si != stock_i && cnt > 0)
                .map(|(si, &cnt)| (si, cnt)).collect();
            pairs_c.sort_by(|a, b| b.1.cmp(&a.1));
            let top10_c: Vec<usize> = pairs_c.iter().take(10).map(|(si, _)| *si).collect();
            for &(si, _) in &pairs_c { cooc_buf[si] = 0; }

            // 计算 hot/cold 的 basic stats
            let hot_ms = basic_stats_from_top10(&valid_stocks, &top10_h);
            let cold_ms = basic_stats_from_top10(&valid_stocks, &top10_c);
            for j in 0..BASIC_FEAT_N {
                for stat_idx in [j, BASIC_FEAT_N + j] {
                    let hv = hot_ms[stat_idx]; let cv = cold_ms[stat_idx];
                    let vals = [hv, cv,
                        if hv.is_finite() && cv.is_finite() { hv - cv } else { f32::NAN },
                        if hv.is_finite() && cv.is_finite() { (hv - cv).abs() } else { f32::NAN }];
                    for v in vals { if cooc_offset < N_FACTORS { facs[cooc_offset] = v; cooc_offset += 1; } }
                }
            }
        }
        facs
    }).collect();

    // ⑤ fan-out: 展开为 (codes, vals)
    let mut out_codes = Vec::with_capacity(n_valid);
    let mut out_vals = Vec::with_capacity(n_valid * N_FACTORS);
    for (stock_i, facs) in all_factors.iter().enumerate() {
        out_codes.push(valid_stocks[stock_i].code.clone());
        out_vals.extend_from_slice(facs);
    }

    Ok((out_codes, out_vals))
}

/// 从 top10 股票索引计算 11 个基础特征的 mean/std
fn basic_stats_from_top10(stocks: &[StockData], top10: &[usize]) -> [f32; BASIC_FEAT_N * 2] {
    let mut result = [f32::NAN; BASIC_FEAT_N * 2];
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
    _stock_i: usize,
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

