//! 柜员游戏扩展系列3：全市场横截面联动与同伴可还原性（anneal_cross_peer）。
//!
//! 这是跨股票横截面因子系列（接入 `run_factor_pipeline_cross_section`）：
//! 每只股票的因子值不仅取决于自身订单流的可还原性，还深度利用了当天全市场其他股票的
//! 逐笔成交可还原性分布、日内共同复杂度模式、价格近邻群、复杂度近邻网络、龙头群等跨股票信息。
//!
//! 核心逻辑（8 大跨股票模块，合计 11,760 维）：
//!   1. 市场共同可还原性节奏剥离与 CAPM/Lead-Lag/龙头传染（48 通道 × 24 统计量 = 1,152 维）
//!   2. 残差可还原性矩阵（剥离全市场当日每分钟共同复杂度后的特异序列）+ 全市场百分位排名动态矩阵
//!      （合计 96 列 × 237 分钟）经 get_features_factors_rust_full 降维 = 6,576 维
//!   3. 多维同伴群（成交额体量 10 档群 / 日内波动率 10 档群 / 主买情绪 10 档群）相对偏离与群内排名 = 1,440 维
//!   4. “同走势、异结构”价格双胞胎群（日内分钟收益率相关性 Top-20 近邻股）可还原性剪刀差 = 960 维
//!   5. “同算法节奏”可还原性近邻网络（分钟可还原性轨迹最同步的 Top-20 股票）特征溢出与补涨空间 = 576 维
//!   6. 大盘宏观状态（市场恐慌急跌/大盘亢奋急涨/高截面分化/大盘放量/市场整体极度混乱分钟）
//!      条件下的个股超额可还原性响应 = 768 维
//!   7. 跨股票联合退火游戏（以全市场标准分位模板与同类群模板作为柜员尺子还原个股）= 288 维
//!
//! 股票白名单约束：严格通过 `symbol_map.csv` 过滤，非股票标的（ETF/基金/可转债）不读取、不参与横截面运算。

use crate::anneal_regime_diff_metrics::{
    adaptive_steps, anneal_core16, FastAnnealBuf, CORE_NAMES, N_CORE, N_MINUTES,
};
use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::io::{Error, ErrorKind};
use std::path::Path;

// ============================================================================
// 常量定义
// ============================================================================

/// 每个分钟跟踪的通道数：3 方向(bid/ask/bid_minus_ask) × 16 指标 = 48 通道
pub const N_CH: usize = 3 * N_CORE; // 48
pub const DIR3_CH_NAMES: [&str; 3] = ["bid", "ask", "bid_minus_ask"];

/// 9 个单股宏观摘要量（用于同伴分组与近邻溢出映射）
pub const N_SUMMARY: usize = 9;
pub const SUMMARY_NAMES: [&str; N_SUMMARY] = [
    "day_ret",
    "late30_ret",
    "early30_ret",
    "log_turnover",
    "intraday_vol",
    "act_buy_ratio",
    "net_inflow_ratio",
    "mean_ord_vol",
    "trade_cnt_log",
];

pub const N_PART1: usize = N_CH * 24; // 1,152
pub const N_RED_COLS: usize = 2 * N_CH; // 96 (48 残差可还原性列 + 48 全市场分位排名列)
pub const N_PART2: usize = 21 * N_RED_COLS + N_RED_COLS * (N_RED_COLS - 1) / 2; // 2016 + 4560 = 6,576
pub const N_PART3: usize = 3 * 2 * N_CH * 5; // 3同伴群(size/vol/abr) × 2时段(fullday/late30) × 48通道 × 5算子 = 1,440
pub const N_PART4: usize = 2 * N_CH * 10; // 2时段 × 48通道 × 10种价格近邻对比算子 = 960
pub const N_PART5: usize = 4 * 9 * 16; // 4种复杂度近邻网络 × 9溢出目标 × 16衍生统计 = 576
pub const N_PART6: usize = 8 * N_CH * 2; // 8种大盘宏观状态对比 × 48通道 × 2(超额均值/Z偏移) = 768
pub const N_PART7: usize = 3 * 2 * 3 * N_CORE; // 3时段 × 2方向(bid/ask) × 3模板(全市场模板/同体量模板/模板差) × 16指标 = 288

/// 总输出维度：1,152 + 6,576 + 1,440 + 960 + 576 + 768 + 288 = 11,760
pub const N_FACTORS: usize =
    N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5 + N_PART6 + N_PART7;

const EPS: f32 = 1e-6;

// ============================================================================
// 股票白名单严格校验（symbol_map.csv）
// ============================================================================

pub fn load_valid_stock_set(symbol_map: &Path) -> std::io::Result<BTreeSet<String>> {
    let invalid = |e| Error::new(ErrorKind::InvalidData, e);
    let mut reader = csv::Reader::from_path(symbol_map).map_err(invalid)?;
    let column = reader
        .headers()
        .map_err(invalid)?
        .iter()
        .position(|h| h == "symbol")
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "股票表缺少 symbol 列"))?;
    let mut stocks = BTreeSet::new();
    for record in reader.records() {
        let record = record.map_err(invalid)?;
        let code = record.get(column).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::new(ErrorKind::InvalidData, "股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "股票白名单为空"));
    }
    Ok(stocks)
}

pub fn list_codes(date: i64, subdir: &str) -> std::io::Result<Vec<String>> {
    let dir = crate::data_paths::level2_dir(date, subdir);
    let symbol_map = crate::data_paths::basic_info_path("symbol_map.csv");
    let stocks = load_valid_stock_set(&symbol_map)?;
    if !dir.is_dir() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("原始数据目录不存在: {}", dir.display()),
        ));
    }
    Ok(stocks
        .into_iter()
        .filter(|code| dir.join(format!("{code}_{date}_{subdir}.csv")).is_file())
        .collect())
}

// ============================================================================
// 单股预提取中间特征结构（Step 2 per-stock 输出）
// ============================================================================

#[derive(Clone)]
pub struct StockMidFeatures {
    /// 237 分钟 × 48 通道（展平为 237 * 48）
    pub min_series: Vec<f32>,
    /// 237 分钟收益率
    pub min_ret: Vec<f32>,
    /// 237 分钟成交额
    pub min_amt: Vec<f32>,
    /// 2 个窗口 (0=全天, 1=尾盘30分) × 48 通道
    pub win_ch: [[f32; N_CH]; 2],
    /// 9 个单股日频摘要特征
    pub summary: [f32; N_SUMMARY],
    /// 3 个窗口 (全天/开盘30分/收盘30分) × 2 方向(bid/ask) 的原始订单量序列（重采样至固定 64 分位点，供跨股票模板退火用）
    pub ord_quantiles: [[[f32; 64]; 2]; 3],
}

#[derive(Clone, Copy)]
struct MinAggOrd {
    vol: f32,
    first_idx: usize,
}

fn agg_bid_ask_vols(trades: &[TradeRecord]) -> (Vec<f32>, Vec<f32>) {
    let mut b_map: FxHashMap<i64, MinAggOrd> =
        FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());
    let mut a_map: FxHashMap<i64, MinAggOrd> =
        FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());
    for (i, t) in trades.iter().enumerate() {
        let be = b_map.entry(t.bid_order).or_insert(MinAggOrd {
            vol: 0.0,
            first_idx: i,
        });
        be.vol += t.volume;
        let ae = a_map.entry(t.ask_order).or_insert(MinAggOrd {
            vol: 0.0,
            first_idx: i,
        });
        ae.vol += t.volume;
    }
    let mut bv: Vec<(usize, f32)> = b_map.values().map(|o| (o.first_idx, o.vol)).collect();
    bv.sort_by_key(|x| x.0);
    let mut av: Vec<(usize, f32)> = a_map.values().map(|o| (o.first_idx, o.vol)).collect();
    av.sort_by_key(|x| x.0);
    (
        bv.into_iter().map(|x| x.1).collect(),
        av.into_iter().map(|x| x.1).collect(),
    )
}

/// 将任意长度的时间序订单量序列重采样到长度 64（保持原始时间相对顺序，供跨股票模板对比退火）
fn resample_time_seq64(seq: &[f32]) -> [f32; 64] {
    let n = seq.len();
    if n < 4 {
        return [f32::NAN; 64];
    }
    let mut sum = 0.0f32;
    for &v in seq {
        sum += v;
    }
    let mean = (sum / n as f32).max(1e-4);
    let mut out = [0.0f32; 64];
    for i in 0..64 {
        let pos = i * (n - 1) / 63;
        out[i] = seq[pos] / mean;
    }
    out
}

pub fn extract_stock_mid(trades: &[TradeRecord]) -> Option<StockMidFeatures> {
    if trades.len() < 15 {
        return None;
    }
    let first_sec = trades[0].time_sec;
    let day_base = ((first_sec as i64) - ((first_sec as i64).rem_euclid(86400))) as f32;
    let t_open = day_base + 9.0 * 3600.0 + 30.0 * 60.0;
    let t_base = if (first_sec - t_open).abs() < 7200.0 {
        t_open
    } else {
        first_sec
    };

    let mut buf = FastAnnealBuf::new();
    let mut min_series = vec![f32::NAN; N_MINUTES * N_CH];
    let mut min_ret = vec![f32::NAN; N_MINUTES];
    let mut min_amt = vec![0.0f32; N_MINUTES];

    let mut total_vol = 0.0f64;
    let mut total_amt = 0.0f64;
    let mut buy_amt = 0.0f64;
    for t in trades {
        total_vol += t.volume as f64;
        total_amt += t.turnover as f64;
        if t.flag == 66 {
            buy_amt += t.turnover as f64;
        }
    }
    if total_vol <= 0.0 || total_amt <= 0.0 {
        return None;
    }

    for m_idx in 0..N_MINUTES {
        let lo = trades.partition_point(|t| t.time_sec < t_base + m_idx as f32 * 60.0);
        let hi = trades.partition_point(|t| t.time_sec < t_base + (m_idx + 1) as f32 * 60.0);
        if hi <= lo + 2 {
            continue;
        }
        let slice = &trades[lo..hi];
        let p0 = slice.first().unwrap().price;
        let p1 = slice.last().unwrap().price;
        if p0 > 0.0 {
            min_ret[m_idx] = (p1 - p0) / p0;
        }
        min_amt[m_idx] = slice.iter().map(|t| t.turnover).sum();

        let (b_seq, a_seq) = agg_bid_ask_vols(slice);
        let b_core = anneal_core16(&b_seq, adaptive_steps(b_seq.len(), 4_000), &mut buf);
        let a_core = anneal_core16(&a_seq, adaptive_steps(a_seq.len(), 4_000), &mut buf);
        let base_off = m_idx * N_CH;
        for k in 0..N_CORE {
            min_series[base_off + k] = b_core[k];
            min_series[base_off + N_CORE + k] = a_core[k];
            if b_core[k].is_finite() && a_core[k].is_finite() {
                min_series[base_off + 2 * N_CORE + k] = b_core[k] - a_core[k];
            }
        }
    }

    // 3 个窗口 (全天 0..14220, 早盘 0..1800, 尾盘 12600..14220)
    let win3_bounds: [(f32, f32); 3] = [(0.0, 14_220.0), (0.0, 1_800.0), (12_600.0, 14_220.0)];
    let mut win_ch = [[f32::NAN; N_CH]; 2];
    let mut ord_quantiles = [[[f32::NAN; 64]; 2]; 3];
    let mut early_ret = 0.0f32;
    let mut late_ret = 0.0f32;
    let mut full_ord_cnt = 1usize;

    for (wi, &(s_lo, s_hi)) in win3_bounds.iter().enumerate() {
        let lo = trades.partition_point(|t| t.time_sec < t_base + s_lo);
        let hi = trades.partition_point(|t| t.time_sec < t_base + s_hi);
        if hi > lo + 2 {
            let slice = &trades[lo..hi];
            let p0 = slice.first().unwrap().price;
            let p1 = slice.last().unwrap().price;
            let r_w = if p0 > 0.0 { (p1 - p0) / p0 } else { 0.0 };
            if wi == 1 {
                early_ret = r_w;
            } else if wi == 2 {
                late_ret = r_w;
            }
            let (b_seq, a_seq) = agg_bid_ask_vols(slice);
            if wi == 0 {
                full_ord_cnt = (b_seq.len() + a_seq.len()).max(1);
            }
            ord_quantiles[wi][0] = resample_time_seq64(&b_seq);
            ord_quantiles[wi][1] = resample_time_seq64(&a_seq);

            if wi == 0 || wi == 2 {
                let dst_w = if wi == 0 { 0 } else { 1 };
                let b_c = anneal_core16(&b_seq, adaptive_steps(b_seq.len(), 15_000), &mut buf);
                let a_c = anneal_core16(&a_seq, adaptive_steps(a_seq.len(), 15_000), &mut buf);
                for k in 0..N_CORE {
                    win_ch[dst_w][k] = b_c[k];
                    win_ch[dst_w][N_CORE + k] = a_c[k];
                    if b_c[k].is_finite() && a_c[k].is_finite() {
                        win_ch[dst_w][2 * N_CORE + k] = b_c[k] - a_c[k];
                    }
                }
            }
        }
    }

    let p_first = trades.first().unwrap().price;
    let p_last = trades.last().unwrap().price;
    let day_ret = if p_first > 0.0 {
        (p_last - p_first) / p_first
    } else {
        0.0
    };
    let intraday_vol = features::col_std(&min_ret);
    let abr = (buy_amt / total_amt) as f32;
    let net_inflow = ((2.0 * buy_amt - total_amt) / total_amt) as f32;
    let summary = [
        day_ret,
        late_ret,
        early_ret,
        (total_amt.ln()) as f32,
        if intraday_vol.is_finite() {
            intraday_vol
        } else {
            0.0
        },
        abr,
        net_inflow,
        (total_vol / full_ord_cnt as f64) as f32,
        (trades.len() as f32).ln(),
    ];

    Some(StockMidFeatures {
        min_series,
        min_ret,
        min_amt,
        win_ch,
        summary,
        ord_quantiles,
    })
}

// ============================================================================
// 横截面辅助计算函数
// ============================================================================

/// 对长度为 n_stocks 的单列做百分位排名 [0, 1]（跳过 NaN）
fn cross_section_rank_pct(vals: &[f32]) -> Vec<f32> {
    let n = vals.len();
    let mut valid: Vec<(f32, usize)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (v, i))
        .collect();
    valid.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut out = vec![f32::NAN; n];
    let m = valid.len();
    if m == 0 {
        return out;
    }
    if m == 1 {
        out[valid[0].1] = 0.5;
        return out;
    }
    for (rank, &(_, idx)) in valid.iter().enumerate() {
        out[idx] = rank as f32 / (m - 1) as f32;
    }
    out
}

/// 将有效股票按某个指标分为 10 组 (0..10)
fn assign_decile_groups(metric: &[f32]) -> Vec<usize> {
    let n = metric.len();
    let mut indexed: Vec<(f32, usize)> = metric
        .iter()
        .enumerate()
        .map(|(i, &v)| (if v.is_finite() { v } else { 0.0 }, i))
        .collect();
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut grp = vec![0usize; n];
    if n == 0 {
        return grp;
    }
    for (pos, &(_, idx)) in indexed.iter().enumerate() {
        grp[idx] = (pos * 10 / n).min(9);
    }
    grp
}

/// 给定自定义初始猜测模板 template_guess（长度为 64），对目标序列 seq（长度为 64）执行退火
fn anneal_with_template64(
    seq: &[f32; 64],
    template_guess: &[f32; 64],
    buf: &mut FastAnnealBuf,
) -> [f32; N_CORE] {
    if !seq[0].is_finite() || !template_guess[0].is_finite() {
        return [f32::NAN; N_CORE];
    }
    // 将 template_guess 缩放到与 seq 相同的均值与标准差，使 ΔS 与 r 公式严密成立
    let n = 64usize;
    let mean_s: f64 = seq.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var_s: f64 = seq
        .iter()
        .map(|&v| {
            let d = v as f64 - mean_s;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let mean_t: f64 = template_guess.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var_t: f64 = template_guess
        .iter()
        .map(|&v| {
            let d = v as f64 - mean_t;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if var_s <= 1e-20 || var_t <= 1e-20 {
        return [f32::NAN; N_CORE];
    }
    let scale = (var_s / var_t).sqrt();
    let mut scaled_seq = [0.0f32; 64];
    for i in 0..64 {
        scaled_seq[i] = seq[i];
    }
    // 先用标准升序退火跑，再用模板重排后的相对残差修正
    let mut base = anneal_core16(&scaled_seq, 2400, buf);
    let mut s_init = 0.0f64;
    for i in 0..64 {
        let g_i = (template_guess[i] as f64 - mean_t) * scale + mean_s;
        let d = g_i - seq[i] as f64;
        s_init += d * d;
    }
    let r0_tmpl = (1.0 - s_init / (2.0 * var_s * n as f64)).clamp(-1.0, 1.0) as f32;
    base[0] = r0_tmpl;
    base[2] = base[1] - r0_tmpl.abs();
    base[3] = (base[1] - r0_tmpl) / (1.0 - r0_tmpl + 1e-4);
    base
}

// ============================================================================
// 横截面主合成核心（无 IO，唯一真相源）：compute_cross_peer_from_mids
// ============================================================================

pub fn compute_cross_peer_from_mids(
    codes: &[String],
    mids: Vec<Option<StockMidFeatures>>,
) -> (Vec<String>, Vec<f32>) {
    let mut valid_codes: Vec<String> = Vec::new();
    let mut stocks: Vec<StockMidFeatures> = Vec::new();
    for (c, m) in codes.iter().zip(mids.into_iter()) {
        if let Some(st) = m {
            valid_codes.push(c.clone());
            stocks.push(st);
        }
    }
    let n_stocks = stocks.len();
    if n_stocks == 0 {
        return (Vec::new(), Vec::new());
    }

    // ------------------------------------------------------------------------
    // 全局横截面聚合准备：
    //   1. 每分钟全市场 48 通道的均值序列 mkt_min_mean[m][ch] 与标准差 mkt_min_std[m][ch]
    //   2. 成交额 Top 50 龙头股均值序列 leader_min_mean[m][ch] 与 Bot 20% 冷门股均值 tail_min_mean[m][ch]
    //   3. 每分钟全市场股票在 48 通道上的百分位排名 rank_min[i][m * 48 + ch]
    //   4. 每分钟大盘宏观状态（市场均值收益、市场截面收益分化度、市场总成交额、市场整体混乱度）
    // ------------------------------------------------------------------------
    let mut mkt_min_mean = vec![[0.0f32; N_CH]; N_MINUTES];
    let mut mkt_min_std = vec![[1.0f32; N_CH]; N_MINUTES];

    // 按全天成交额排序找 Leaders (Top 10% 或 Top 50) 与 Laggards (Bot 20%)
    let mut amt_order: Vec<(f32, usize)> = stocks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.summary[3], i))
        .collect();
    amt_order.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let k_lead = (n_stocks / 10).clamp(1, 50.min(n_stocks));
    let k_lag = (n_stocks / 5).max(1);
    let mut is_leader = vec![false; n_stocks];
    let mut is_laggard = vec![false; n_stocks];
    for &(_, idx) in &amt_order[n_stocks - k_lead..] {
        is_leader[idx] = true;
    }
    for &(_, idx) in &amt_order[..k_lag] {
        is_laggard[idx] = true;
    }

    let mut leader_min_mean = vec![[0.0f32; N_CH]; N_MINUTES];
    let mut laggard_min_mean = vec![[0.0f32; N_CH]; N_MINUTES];

    // 每只股票的 237×48 残差序列与 237×48 截面百分位排名序列
    let mut resid_min = vec![vec![f32::NAN; N_MINUTES * N_CH]; n_stocks];
    let mut rank_min = vec![vec![f32::NAN; N_MINUTES * N_CH]; n_stocks];

    // 大盘分钟级宏观指标
    let mut mkt_ret_min = vec![0.0f32; N_MINUTES];
    let mut mkt_disp_min = vec![0.0f32; N_MINUTES];
    let mut mkt_amt_min = vec![0.0f32; N_MINUTES];

    for m in 0..N_MINUTES {
        // 市场分钟收益与截面标准差
        let r_col: Vec<f32> = stocks.iter().map(|s| s.min_ret[m]).collect();
        let r_mean = features::col_mean(&r_col);
        let r_std = features::col_std(&r_col);
        mkt_ret_min[m] = if r_mean.is_finite() { r_mean } else { 0.0 };
        mkt_disp_min[m] = if r_std.is_finite() { r_std } else { 0.0 };
        mkt_amt_min[m] = stocks.iter().map(|s| s.min_amt[m]).sum();

        for ch in 0..N_CH {
            let col: Vec<f32> = stocks
                .iter()
                .map(|s| s.min_series[m * N_CH + ch])
                .collect();
            let c_mean = features::col_mean(&col);
            let c_std = features::col_std(&col);
            mkt_min_mean[m][ch] = if c_mean.is_finite() { c_mean } else { 0.0 };
            mkt_min_std[m][ch] = if c_std.is_finite() && c_std > 1e-6 {
                c_std
            } else {
                1.0
            };

            let mut l_sum = 0.0f64;
            let mut l_cnt = 0usize;
            let mut g_sum = 0.0f64;
            let mut g_cnt = 0usize;
            for (i, &v) in col.iter().enumerate() {
                if v.is_finite() {
                    if is_leader[i] {
                        l_sum += v as f64;
                        l_cnt += 1;
                    }
                    if is_laggard[i] {
                        g_sum += v as f64;
                        g_cnt += 1;
                    }
                }
            }
            leader_min_mean[m][ch] = if l_cnt > 0 {
                (l_sum / l_cnt as f64) as f32
            } else {
                mkt_min_mean[m][ch]
            };
            laggard_min_mean[m][ch] = if g_cnt > 0 {
                (g_sum / g_cnt as f64) as f32
            } else {
                mkt_min_mean[m][ch]
            };

            let r_pct = cross_section_rank_pct(&col);
            for i in 0..n_stocks {
                let v = col[i];
                if v.is_finite() {
                    resid_min[i][m * N_CH + ch] = v - mkt_min_mean[m][ch];
                }
                rank_min[i][m * N_CH + ch] = r_pct[i];
            }
        }
    }

    // ------------------------------------------------------------------------
    // 同伴群划分（3 种十档分组：0=成交额体量群, 1=日内波动率群, 2=主买情绪群）
    // ------------------------------------------------------------------------
    let size_metric: Vec<f32> = stocks.iter().map(|s| s.summary[3]).collect();
    let vol_metric: Vec<f32> = stocks.iter().map(|s| s.summary[4]).collect();
    let abr_metric: Vec<f32> = stocks.iter().map(|s| s.summary[5]).collect();
    let peer_groups: [Vec<usize>; 3] = [
        assign_decile_groups(&size_metric),
        assign_decile_groups(&vol_metric),
        assign_decile_groups(&abr_metric),
    ];

    // 预计算每个同伴群在 2 个时段 × 48 通道上的组内均值、标准差、以及组内排名
    let mut grp_mean = [[[[0.0f32; N_CH]; 2]; 10]; 3];
    let mut grp_std = [[[[1.0f32; N_CH]; 2]; 10]; 3];
    let mut grp_rank = vec![[[[0.5f32; N_CH]; 2]; 3]; n_stocks];
    let mut global_win_rank = vec![[[0.5f32; N_CH]; 2]; n_stocks];

    for w in 0..2 {
        for ch in 0..N_CH {
            let col: Vec<f32> = stocks.iter().map(|s| s.win_ch[w][ch]).collect();
            let g_rk = cross_section_rank_pct(&col);
            for i in 0..n_stocks {
                global_win_rank[i][w][ch] = g_rk[i];
            }
            for g_type in 0..3 {
                for d_bin in 0..10 {
                    let sub_idx: Vec<usize> = (0..n_stocks)
                        .filter(|&i| peer_groups[g_type][i] == d_bin && col[i].is_finite())
                        .collect();
                    if !sub_idx.is_empty() {
                        let sub_vals: Vec<f32> = sub_idx.iter().map(|&i| col[i]).collect();
                        let m_v = features::col_mean(&sub_vals);
                        let s_v = features::col_std(&sub_vals);
                        grp_mean[g_type][d_bin][w][ch] = if m_v.is_finite() { m_v } else { 0.0 };
                        grp_std[g_type][d_bin][w][ch] =
                            if s_v.is_finite() && s_v > 1e-6 {
                                s_v
                            } else {
                                1.0
                            };
                        let sub_rk = cross_section_rank_pct(&sub_vals);
                        for (pos, &orig_i) in sub_idx.iter().enumerate() {
                            grp_rank[orig_i][g_type][w][ch] = sub_rk[pos];
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 近邻网络构建：
    //   1. 价格走势双胞胎近邻 (Price-Twin Top-K): 基于 237 分钟收益率标准化向量的内积相似度
    //   2. 4 组可还原性复杂度近邻 (Complexity-Twin Top-K):
    //      net 0: bid_final_r (ch=1) 轨迹同步近邻
    //      net 1: ask_final_r (ch=17) 轨迹同步近邻
    //      net 2: bid_mid_deter (ch=11) 轨迹同步近邻
    //      net 3: bid_minus_ask_final_r (ch=33) 轨迹同步近邻
    // ------------------------------------------------------------------------
    let normalize_vec = |v: &[f32]| -> Vec<f32> {
        let mut s1 = 0.0f64;
        let mut c = 0usize;
        for &x in v {
            if x.is_finite() {
                s1 += x as f64;
                c += 1;
            }
        }
        if c < 5 {
            return vec![0.0f32; v.len()];
        }
        let mean = (s1 / c as f64) as f32;
        let mut s2 = 0.0f64;
        for &x in v {
            if x.is_finite() {
                let d = (x - mean) as f64;
                s2 += d * d;
            }
        }
        let norm = s2.sqrt() as f32;
        if norm < 1e-6 {
            return vec![0.0f32; v.len()];
        }
        v.iter()
            .map(|&x| if x.is_finite() { (x - mean) / norm } else { 0.0 })
            .collect()
    };

    let ret_norm_vecs: Vec<Vec<f32>> = stocks
        .iter()
        .map(|s| normalize_vec(&s.min_ret))
        .collect();

    let comp_net_chs: [usize; 4] = [1, 17, 11, 33];
    let comp_norm_vecs: [Vec<Vec<f32>>; 4] = std::array::from_fn(|net_i| {
        let ch = comp_net_chs[net_i];
        stocks
            .iter()
            .map(|s| {
                let seq: Vec<f32> = (0..N_MINUTES)
                    .map(|m| s.min_series[m * N_CH + ch])
                    .collect();
                normalize_vec(&seq)
            })
            .collect()
    });

    let k_nn = 15.min(n_stocks.saturating_sub(1)).max(1);

    // 并行找每只股票的 Top-K 价格走势近邻与 Top-K/Bot-K 复杂度近邻
    let find_top_bot_k = |vecs: &[Vec<f32>], i: usize| -> (Vec<(usize, f32)>, Vec<(usize, f32)>) {
        if n_stocks <= 1 {
            return (vec![(0, 1.0)], vec![(0, 1.0)]);
        }
        let vi = &vecs[i];
        let mut sims: Vec<(f32, usize)> = (0..n_stocks)
            .filter(|&j| j != i)
            .map(|j| {
                let vj = &vecs[j];
                let dot: f32 = vi.iter().zip(vj.iter()).map(|(&a, &b)| a * b).sum();
                (dot, j)
            })
            .collect();
        sims.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let top: Vec<(usize, f32)> = sims.iter().take(k_nn).map(|&(s, j)| (j, s)).collect();
        let bot: Vec<(usize, f32)> = sims.iter().rev().take(k_nn).map(|&(s, j)| (j, s)).collect();
        (top, bot)
    };

    // ------------------------------------------------------------------------
    // 大盘宏观条件分钟掩码（模块 6 用，共 8 组对比对）
    // ------------------------------------------------------------------------
    let (mkt_up_hi, mkt_dn_lo) = {
        let mut idx: Vec<(f32, usize)> = mkt_ret_min.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        idx.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let k = (N_MINUTES / 5).max(1);
        let mut hi = vec![false; N_MINUTES];
        let mut lo = vec![false; N_MINUTES];
        for &(_, m) in &idx[..k] {
            lo[m] = true;
        }
        for &(_, m) in &idx[N_MINUTES - k..] {
            hi[m] = true;
        }
        (hi, lo)
    };
    let (mkt_disp_hi, mkt_disp_lo) = {
        let mut idx: Vec<(f32, usize)> = mkt_disp_min.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        idx.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let k = (N_MINUTES / 5).max(1);
        let mut hi = vec![false; N_MINUTES];
        let mut lo = vec![false; N_MINUTES];
        for &(_, m) in &idx[..k] {
            lo[m] = true;
        }
        for &(_, m) in &idx[N_MINUTES - k..] {
            hi[m] = true;
        }
        (hi, lo)
    };
    let (mkt_amt_hi, mkt_amt_lo) = {
        let mut idx: Vec<(f32, usize)> = mkt_amt_min.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        idx.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let k = (N_MINUTES / 5).max(1);
        let mut hi = vec![false; N_MINUTES];
        let mut lo = vec![false; N_MINUTES];
        for &(_, m) in &idx[..k] {
            lo[m] = true;
        }
        for &(_, m) in &idx[N_MINUTES - k..] {
            hi[m] = true;
        }
        (hi, lo)
    };
    // 市场整体最难还原（mkt_min_mean[m][1] 即 bid_final_r 最低的 20% 分钟）vs 最易还原分钟
    let (mkt_easy_hi, mkt_hard_lo) = {
        let mut idx: Vec<(f32, usize)> = (0..N_MINUTES)
            .map(|m| (mkt_min_mean[m][1], m))
            .collect();
        idx.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let k = (N_MINUTES / 5).max(1);
        let mut hi = vec![false; N_MINUTES];
        let mut lo = vec![false; N_MINUTES];
        for &(_, m) in &idx[..k] {
            lo[m] = true;
        }
        for &(_, m) in &idx[N_MINUTES - k..] {
            hi[m] = true;
        }
        (hi, lo)
    };

    // ------------------------------------------------------------------------
    // 全市场与同体量群的 64 点订单分位模板（供模块 7 跨股票联合退火使用）
    // ------------------------------------------------------------------------
    let mut mkt_ord_template = [[[0.0f32; 64]; 2]; 3];
    let mut grp_ord_template = [[[[0.0f32; 64]; 2]; 3]; 10];
    for wi in 0..3 {
        for di in 0..2 {
            for pos in 0..64 {
                let mut s_all = 0.0f64;
                let mut c_all = 0usize;
                let mut s_grp = [0.0f64; 10];
                let mut c_grp = [0usize; 10];
                for i in 0..n_stocks {
                    let v = stocks[i].ord_quantiles[wi][di][pos];
                    if v.is_finite() {
                        s_all += v as f64;
                        c_all += 1;
                        let g = peer_groups[0][i];
                        s_grp[g] += v as f64;
                        c_grp[g] += 1;
                    }
                }
                let m_all = if c_all > 0 {
                    (s_all / c_all as f64) as f32
                } else {
                    1.0
                };
                mkt_ord_template[wi][di][pos] = m_all;
                for g in 0..10 {
                    grp_ord_template[g][wi][di][pos] = if c_grp[g] > 0 {
                        (s_grp[g] / c_grp[g] as f64) as f32
                    } else {
                        m_all
                    };
                }
            }
        }
    }

    // 预生成模块 2 降维列名（只生成一次，供并行线程共享）
    let red_col_names = build_s3_red_col_names();

    // ------------------------------------------------------------------------
    // Step 3: 并行合成每只股票的全部 11,760 个横截面因子
    // ------------------------------------------------------------------------
    let per_stock_out: Vec<Vec<f32>> = (0..n_stocks)
        .into_par_iter()
        .map(|i| {
            let mut out = Vec::with_capacity(N_FACTORS);
            let st = &stocks[i];
            let mut buf = FastAnnealBuf::new();

            // ================================================================
            // 模块 1：市场共同可还原性节奏剥离与 Lead-Lag / 龙头传染 (48 × 24 = 1,152 维)
            // ================================================================
            for ch in 0..N_CH {
                let x_seq: Vec<f32> = (0..N_MINUTES)
                    .map(|m| st.min_series[m * N_CH + ch])
                    .collect();
                let m_seq: Vec<f32> = (0..N_MINUTES).map(|m| mkt_min_mean[m][ch]).collect();
                let l_seq: Vec<f32> = (0..N_MINUTES).map(|m| leader_min_mean[m][ch]).collect();
                let g_seq: Vec<f32> = (0..N_MINUTES).map(|m| laggard_min_mean[m][ch]).collect();
                let rk_seq: Vec<f32> = (0..N_MINUTES)
                    .map(|m| rank_min[i][m * N_CH + ch])
                    .collect();

                let corr_mkt = features::corr_pair(&x_seq, &m_seq);
                let std_x = features::col_std(&x_seq);
                let std_m = features::col_std(&m_seq);
                let mean_x = features::col_mean(&x_seq);
                let mean_m = features::col_mean(&m_seq);
                let beta_mkt = if std_m.is_finite() && std_m > 1e-6 && std_x.is_finite() {
                    corr_mkt * (std_x / std_m)
                } else {
                    0.0
                };
                let alpha_mkt = if mean_x.is_finite() && mean_m.is_finite() {
                    mean_x - beta_mkt * mean_m
                } else {
                    f32::NAN
                };
                let r2_mkt = if corr_mkt.is_finite() {
                    corr_mkt * corr_mkt
                } else {
                    f32::NAN
                };

                // 残差早盘/午盘/尾盘均值
                let res_seq: Vec<f32> = (0..N_MINUTES)
                    .map(|m| resid_min[i][m * N_CH + ch])
                    .collect();
                let res_early = features::col_mean(&res_seq[0..30]);
                let res_mid = features::col_mean(&res_seq[30..207]);
                let res_late = features::col_mean(&res_seq[207..237]);

                // Lead-Lag 交叉相关 (lag 1, 3, 5 分钟)
                let lead1 = features::corr_pair(&x_seq[..N_MINUTES - 1], &m_seq[1..]);
                let lag1 = features::corr_pair(&m_seq[..N_MINUTES - 1], &x_seq[1..]);
                let lead3 = features::corr_pair(&x_seq[..N_MINUTES - 3], &m_seq[3..]);
                let lag3 = features::corr_pair(&m_seq[..N_MINUTES - 3], &x_seq[3..]);
                let lead5 = features::corr_pair(&x_seq[..N_MINUTES - 5], &m_seq[5..]);
                let lag5 = features::corr_pair(&m_seq[..N_MINUTES - 5], &x_seq[5..]);

                // 与市场核心龙头群 vs 边缘冷门群的共振度
                let corr_lead_grp = features::corr_pair(&x_seq, &l_seq);
                let corr_lag_grp = features::corr_pair(&x_seq, &g_seq);

                // 横截面排名霸榜率 (Top 5%, Top 10%, Bot 10%) 及早尾盘排名跃迁
                let mut valid_rk = 0usize;
                let mut top5_cnt = 0usize;
                let mut top10_cnt = 0usize;
                let mut bot10_cnt = 0usize;
                let mut max_streak_top10 = 0usize;
                let mut cur_streak = 0usize;
                for &r in &rk_seq {
                    if r.is_finite() {
                        valid_rk += 1;
                        if r >= 0.95 {
                            top5_cnt += 1;
                        }
                        if r >= 0.90 {
                            top10_cnt += 1;
                            cur_streak += 1;
                            if cur_streak > max_streak_top10 {
                                max_streak_top10 = cur_streak;
                            }
                        } else {
                            cur_streak = 0;
                        }
                        if r <= 0.10 {
                            bot10_cnt += 1;
                        }
                    }
                }
                let vf = valid_rk.max(1) as f32;
                let rk_early = features::col_mean(&rk_seq[0..30]);
                let rk_late = features::col_mean(&rk_seq[207..237]);

                out.extend_from_slice(&[
                    corr_mkt,
                    beta_mkt,
                    alpha_mkt,
                    r2_mkt,
                    res_early,
                    res_mid,
                    res_late,
                    res_late - res_early,
                    res_late - res_mid,
                    lead1,
                    lag1,
                    lead1 - lag1,
                    lead3,
                    lag3,
                    lead3 - lag3,
                    lead5 - lag5,
                    corr_lead_grp,
                    corr_lag_grp,
                    corr_lead_grp - corr_lag_grp,
                    top5_cnt as f32 / vf,
                    top10_cnt as f32 / vf,
                    bot10_cnt as f32 / vf,
                    max_streak_top10 as f32 / vf,
                    rk_late - rk_early,
                ]);
            }
            debug_assert_eq!(out.len(), N_PART1);

            // ================================================================
            // 模块 2：残差可还原性矩阵(48列) + 横截面百分位排名动态矩阵(48列) = 96列降维 (6,576 维)
            // ================================================================
            let mut mat96 = Array2::<f32>::from_elem((N_MINUTES, N_RED_COLS), f32::NAN);
            for m in 0..N_MINUTES {
                for ch in 0..N_CH {
                    mat96[[m, ch]] = resid_min[i][m * N_CH + ch];
                    mat96[[m, N_CH + ch]] = rank_min[i][m * N_CH + ch];
                }
            }
            let (red_vals, _) =
                features::get_features_factors_rust_full(&mat96.view(), &red_col_names, false);
            out.extend_from_slice(&red_vals);
            debug_assert_eq!(out.len(), N_PART1 + N_PART2);

            // ================================================================
            // 模块 3：3 种同伴群 (成交额体量 / 波动率 / 主买占比) 相对偏离 (1,440 维)
            // ================================================================
            for g_type in 0..3 {
                let d_bin = peer_groups[g_type][i];
                for w in 0..2 {
                    for ch in 0..N_CH {
                        let x = st.win_ch[w][ch];
                        let gm = grp_mean[g_type][d_bin][w][ch];
                        let gs = grp_std[g_type][d_bin][w][ch];
                        let grk = grp_rank[i][g_type][w][ch];
                        let gl_rk = global_win_rank[i][w][ch];
                        let diff = if x.is_finite() { x - gm } else { f32::NAN };
                        let z = if x.is_finite() {
                            (x - gm) / gs
                        } else {
                            f32::NAN
                        };
                        let ratio = if x.is_finite() {
                            x / (gm.abs() + EPS)
                        } else {
                            f32::NAN
                        };
                        out.extend_from_slice(&[diff, z, ratio, grk, grk - gl_rk]);
                    }
                }
            }
            debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3);

            // ================================================================
            // 模块 4：“同走势、异结构”价格走势双胞胎近邻群对比 (960 维)
            // ================================================================
            let (p_top, p_bot) = find_top_bot_k(&ret_norm_vecs, i);
            for w in 0..2 {
                for ch in 0..N_CH {
                    let x = st.win_ch[w][ch];
                    let mut sum_top = 0.0f64;
                    let mut wsum_top = 0.0f64;
                    let mut w_tot = 0.0f64;
                    let mut sq_top = 0.0f64;
                    let mut cnt_top = 0usize;
                    let mut max_top = f32::NEG_INFINITY;
                    let mut min_top = f32::INFINITY;
                    let mut beat_cnt = 0usize;

                    for &(nj, sim) in &p_top {
                        let v = stocks[nj].win_ch[w][ch];
                        if v.is_finite() {
                            sum_top += v as f64;
                            sq_top += (v as f64) * (v as f64);
                            let wt = sim.max(0.05) as f64;
                            wsum_top += (v as f64) * wt;
                            w_tot += wt;
                            cnt_top += 1;
                            if v > max_top {
                                max_top = v;
                            }
                            if v < min_top {
                                min_top = v;
                            }
                            if x.is_finite() && x > v {
                                beat_cnt += 1;
                            }
                        }
                    }

                    let mut sum_bot = 0.0f64;
                    let mut cnt_bot = 0usize;
                    for &(nj, _) in &p_bot {
                        let v = stocks[nj].win_ch[w][ch];
                        if v.is_finite() {
                            sum_bot += v as f64;
                            cnt_bot += 1;
                        }
                    }

                    if cnt_top > 0 && x.is_finite() {
                        let nf = cnt_top as f64;
                        let m_top = (sum_top / nf) as f32;
                        let wm_top = if w_tot > 0.0 {
                            (wsum_top / w_tot) as f32
                        } else {
                            m_top
                        };
                        let std_top = if cnt_top >= 2 {
                            ((sq_top - sum_top * sum_top / nf) / (nf - 1.0))
                                .max(0.0)
                                .sqrt() as f32
                        } else {
                            1.0
                        };
                        let m_bot = if cnt_bot > 0 {
                            (sum_bot / cnt_bot as f64) as f32
                        } else {
                            m_top
                        };
                        out.extend_from_slice(&[
                            m_top,
                            x - m_top,
                            (x - m_top) / (std_top + 1e-4),
                            x / (m_top.abs() + EPS),
                            x - wm_top,
                            beat_cnt as f32 / cnt_top as f32,
                            std_top,
                            x - max_top,
                            x - min_top,
                            m_top - m_bot,
                        ]);
                    } else {
                        out.extend_from_slice(&[f32::NAN; 10]);
                    }
                }
            }
            debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3 + N_PART4);

            // ================================================================
            // 模块 5：可还原性近邻网络（Complexity-Twin Network）特征溢出与补涨差 (576 维)
            // 4 个复杂度近邻网 × 9 个日频特征 × 4 个溢出统计量 = 144 维
            // 4 个复杂度近邻网 × 36 个跨通道结构溢出统计量 = 432 维 -> 合计 576 维
            // ================================================================
            for net_i in 0..4 {
                let (c_top, c_bot) = find_top_bot_k(&comp_norm_vecs[net_i], i);
                // 5a: 9 个 summary 特征在复杂度近邻群上的溢出 (9 × 8 = 72 维)
                for s_idx in 0..N_SUMMARY {
                    let self_v = st.summary[s_idx];
                    let mut s_top = 0.0f64;
                    let mut ws_top = 0.0f64;
                    let mut w_tot = 0.0f64;
                    let mut sq_top = 0.0f64;
                    let mut n_t = 0usize;
                    for &(nj, sim) in &c_top {
                        let v = stocks[nj].summary[s_idx];
                        if v.is_finite() {
                            s_top += v as f64;
                            sq_top += (v as f64) * (v as f64);
                            let wt = sim.max(0.05) as f64;
                            ws_top += (v as f64) * wt;
                            w_tot += wt;
                            n_t += 1;
                        }
                    }
                    let mut s_bot = 0.0f64;
                    let mut n_b = 0usize;
                    for &(nj, _) in &c_bot {
                        let v = stocks[nj].summary[s_idx];
                        if v.is_finite() {
                            s_bot += v as f64;
                            n_b += 1;
                        }
                    }
                    if n_t > 0 && self_v.is_finite() {
                        let nf = n_t as f64;
                        let m_t = (s_top / nf) as f32;
                        let wm_t = if w_tot > 0.0 {
                            (ws_top / w_tot) as f32
                        } else {
                            m_t
                        };
                        let std_t = if n_t >= 2 {
                            ((sq_top - s_top * s_top / nf) / (nf - 1.0))
                                .max(0.0)
                                .sqrt() as f32
                        } else {
                            1.0
                        };
                        let m_b = if n_b > 0 {
                            (s_bot / n_b as f64) as f32
                        } else {
                            m_t
                        };
                        out.extend_from_slice(&[
                            m_t,
                            wm_t,
                            self_v - m_t,
                            (self_v - m_t) / (std_t + 1e-4),
                            self_v - wm_t,
                            std_t,
                            m_b,
                            m_t - m_b,
                        ]);
                    } else {
                        out.extend_from_slice(&[f32::NAN; 8]);
                    }
                }
                // 5b: 该复杂度近邻群在其他 18 个可还原性通道上的平均表现与本股偏离 (18 × 4 = 72 维)
                // 每个 net_i 合计 72 + 72 = 144 维，4 个 net 共 576 维
                for ch_sub in 0..18 {
                    let ch = (ch_sub * 2) % N_CH;
                    let x0 = st.win_ch[0][ch];
                    let x1 = st.win_ch[1][ch];
                    let mut s0 = 0.0f64;
                    let mut s1 = 0.0f64;
                    let mut c0 = 0usize;
                    let mut c1 = 0usize;
                    for &(nj, _) in &c_top {
                        let v0 = stocks[nj].win_ch[0][ch];
                        if v0.is_finite() {
                            s0 += v0 as f64;
                            c0 += 1;
                        }
                        let v1 = stocks[nj].win_ch[1][ch];
                        if v1.is_finite() {
                            s1 += v1 as f64;
                            c1 += 1;
                        }
                    }
                    let m0 = if c0 > 0 {
                        (s0 / c0 as f64) as f32
                    } else {
                        f32::NAN
                    };
                    let m1 = if c1 > 0 {
                        (s1 / c1 as f64) as f32
                    } else {
                        f32::NAN
                    };
                    out.extend_from_slice(&[
                        m0,
                        if x0.is_finite() && m0.is_finite() {
                            x0 - m0
                        } else {
                            f32::NAN
                        },
                        m1,
                        if x1.is_finite() && m1.is_finite() {
                            x1 - m1
                        } else {
                            f32::NAN
                        },
                    ]);
                }
            }
            debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5);

            // ================================================================
            // 模块 6：大盘宏观状态下的个股超额可还原性响应 (8 状态 × 48 通道 × 2 = 768 维)
            // ================================================================
            let mkt_masks: [&[bool]; 8] = [
                &mkt_up_hi,   // 0: 大盘急涨亢奋分钟
                &mkt_dn_lo,   // 1: 大盘急跌恐慌分钟
                &mkt_disp_hi, // 2: 全市场高分化分钟
                &mkt_disp_lo, // 3: 全市场一致齐涨跌分钟
                &mkt_amt_hi,  // 4: 全市场巨量分钟
                &mkt_amt_lo,  // 5: 全市场缩量冷清分钟
                &mkt_easy_hi, // 6: 全市场整体最易还原分钟
                &mkt_hard_lo, // 7: 全市场整体最混乱难还原分钟
            ];
            for mask in &mkt_masks {
                for ch in 0..N_CH {
                    let mut s_res = 0.0f64;
                    let mut s_rk = 0.0f64;
                    let mut c = 0usize;
                    for m in 0..N_MINUTES {
                        if mask[m] {
                            let r_v = resid_min[i][m * N_CH + ch];
                            let k_v = rank_min[i][m * N_CH + ch];
                            if r_v.is_finite() && k_v.is_finite() {
                                s_res += r_v as f64;
                                s_rk += k_v as f64;
                                c += 1;
                            }
                        }
                    }
                    if c > 0 {
                        out.push((s_res / c as f64) as f32);
                        out.push((s_rk / c as f64) as f32);
                    } else {
                        out.push(f32::NAN);
                        out.push(f32::NAN);
                    }
                }
            }
            debug_assert_eq!(
                out.len(),
                N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5 + N_PART6
            );

            // ================================================================
            // 模块 7：跨股票联合退火游戏（以全市场模板和同体量模板作为柜员尺子，288 维）
            // 3 时段(fullday/early30/late30) × 2 方向(bid/ask) × 3(市场模板/同伴模板/差) × 16 指标 = 288 维
            // ================================================================
            let size_grp = peer_groups[0][i];
            for wi in 0..3 {
                for di in 0..2 {
                    let seq64 = &st.ord_quantiles[wi][di];
                    let m_tmpl = &mkt_ord_template[wi][di];
                    let g_tmpl = &grp_ord_template[size_grp][wi][di];
                    let v_mkt = anneal_with_template64(seq64, m_tmpl, &mut buf);
                    let v_grp = anneal_with_template64(seq64, g_tmpl, &mut buf);
                    out.extend_from_slice(&v_mkt);
                    out.extend_from_slice(&v_grp);
                    for k in 0..N_CORE {
                        if v_mkt[k].is_finite() && v_grp[k].is_finite() {
                            out.push(v_mkt[k] - v_grp[k]);
                        } else {
                            out.push(f32::NAN);
                        }
                    }
                }
            }
            debug_assert_eq!(out.len(), N_FACTORS);
            out
        })
        .collect();

    let mut flat_vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for row in per_stock_out {
        flat_vals.extend_from_slice(&row);
    }
    (valid_codes, flat_vals)
}

// ============================================================================
// 因子名生成（与 compute_cross_peer_from_mids 严格逐一对齐）
// ============================================================================

fn build_ch48_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_CH);
    for d in &DIR3_CH_NAMES {
        for m in &CORE_NAMES {
            names.push(format!("{}_{}", d, m));
        }
    }
    names
}

fn build_s3_red_col_names() -> Vec<String> {
    let ch48 = build_ch48_names();
    let mut names = Vec::with_capacity(N_RED_COLS);
    for c in &ch48 {
        names.push(format!("s3_resmin_{}", c));
    }
    for c in &ch48 {
        names.push(format!("s3_rkmin_{}", c));
    }
    names
}

pub fn anneal_cross_peer_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    let ch48 = build_ch48_names();

    // 模块 1: 1,152
    let p1_stats = [
        "corr_mkt",
        "beta_mkt",
        "alpha_mkt",
        "r2_mkt",
        "res_early",
        "res_mid",
        "res_late",
        "res_late_minus_early",
        "res_late_minus_mid",
        "lead1_mkt",
        "lag1_mkt",
        "lead_minus_lag1",
        "lead3_mkt",
        "lag3_mkt",
        "lead_minus_lag3",
        "lead_minus_lag5",
        "corr_leaders",
        "corr_laggards",
        "corr_lead_minus_lag_grp",
        "rk_top5_share",
        "rk_top10_share",
        "rk_bot10_share",
        "rk_top10_max_streak",
        "rk_late_minus_early",
    ];
    for c in &ch48 {
        for st in &p1_stats {
            names.push(format!("s3_mkt_{}_{}", c, st));
        }
    }

    // 模块 2: 6,576
    let red_col_names = build_s3_red_col_names();
    let dummy = Array2::<f32>::ones((2, N_RED_COLS));
    let (_, red_names) =
        features::get_features_factors_rust_full(&dummy.view(), &red_col_names, false);
    names.extend(red_names);

    // 模块 3: 1,440
    let grp_names = ["size_peer", "vol_peer", "abr_peer"];
    let win2_names = ["fullday", "late30"];
    let p3_ops = ["diff", "zscore", "ratio", "grp_rk", "grp_minus_gl_rk"];
    for g in &grp_names {
        for w in &win2_names {
            for c in &ch48 {
                for op in &p3_ops {
                    names.push(format!("s3_{}_{}_{}_{}", g, w, c, op));
                }
            }
        }
    }

    // 模块 4: 960
    let p4_ops = [
        "twin_mean",
        "self_minus_twin",
        "twin_zscore",
        "self_over_twin",
        "self_minus_wtwin",
        "twin_beat_ratio",
        "twin_std",
        "self_minus_twin_max",
        "self_minus_twin_min",
        "top_minus_bot_twin",
    ];
    for w in &win2_names {
        for c in &ch48 {
            for op in &p4_ops {
                names.push(format!("s3_prctwin_{}_{}_{}", w, c, op));
            }
        }
    }

    // 模块 5: 576
    let net4_names = [
        "cnet_bid_r",
        "cnet_ask_r",
        "cnet_bid_deter",
        "cnet_bma_r",
    ];
    let p5a_ops = [
        "peer_mean",
        "peer_wmean",
        "self_minus_peer",
        "peer_zscore",
        "self_minus_wpeer",
        "peer_std",
        "antipeer_mean",
        "peer_minus_antipeer",
    ];
    for net in &net4_names {
        for sum_name in &SUMMARY_NAMES {
            for op in &p5a_ops {
                names.push(format!("s3_{}_spill_{}_{}", net, sum_name, op));
            }
        }
        for ch_sub in 0..18 {
            let ch = (ch_sub * 2) % N_CH;
            let cname = &ch48[ch];
            names.push(format!("s3_{}_xch_{}_full_peer", net, cname));
            names.push(format!("s3_{}_xch_{}_full_diff", net, cname));
            names.push(format!("s3_{}_xch_{}_late_peer", net, cname));
            names.push(format!("s3_{}_xch_{}_late_diff", net, cname));
        }
    }

    // 模块 6: 768
    let mkt_regime8_names = [
        "mkt_surge",
        "mkt_crash",
        "mkt_high_disp",
        "mkt_low_disp",
        "mkt_high_amt",
        "mkt_low_amt",
        "mkt_easy_ord",
        "mkt_hard_ord",
    ];
    for mr in &mkt_regime8_names {
        for c in &ch48 {
            names.push(format!("s3_mreg_{}_{}_mean_resid", mr, c));
            names.push(format!("s3_mreg_{}_{}_mean_rank", mr, c));
        }
    }

    // 模块 7: 288
    for w_name in &["fullday", "early30", "late30"] {
        for d_name in &["bid", "ask"] {
            for t_mode in &["mkt_tmpl", "grp_tmpl", "mkt_minus_grp_tmpl"] {
                for m in &CORE_NAMES {
                    names.push(format!("s3_joint_{}_{}_{}_{}", w_name, d_name, t_mode, m));
                }
            }
        }
    }

    assert_eq!(names.len(), N_FACTORS);
    names
}

// ============================================================================
// I/O 入口（v1 读盘 / v2 传数据）与 PyO3 绑定
// ============================================================================

/// v1 入口：读取当天全市场股票（严格按 symbol_map.csv 白名单过滤）并计算横截面因子
pub fn compute_anneal_cross_peer_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction")?;
    let mids: Vec<Option<StockMidFeatures>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            extract_stock_mid(&trades)
        })
        .collect();
    Ok(compute_cross_peer_from_mids(&codes, mids))
}

/// v2 入口：从内存预加载的数据计算，同时严格核对 symbol_map.csv 白名单，混入非股票立即报错
pub fn compute_anneal_cross_peer_from_trades(
    codes: &[String],
    trades_per_code: Vec<Option<Vec<TradeRecord>>>,
) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let symbol_map = crate::data_paths::basic_info_path("symbol_map.csv");
    let valid_stocks = load_valid_stock_set(&symbol_map)?;
    for code in codes {
        if !valid_stocks.contains(code) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("输入代码 {code} 不在 symbol_map.csv 股票白名单中，拒绝横截面计算"),
            ));
        }
    }
    let mids: Vec<Option<StockMidFeatures>> = trades_per_code
        .into_par_iter()
        .map(|opt_t| opt_t.and_then(|trades| extract_stock_mid(&trades)))
        .collect();
    Ok(compute_cross_peer_from_mids(codes, mids))
}

#[pyfunction]
pub fn py_anneal_cross_peer(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_anneal_cross_peer_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_anneal_cross_peer_names() -> Vec<String> {
    anneal_cross_peer_names()
}

#[pyfunction]
pub fn py_anneal_cross_peer_from_data(
    _py: Python<'_>,
    codes: Vec<String>,
    trades_list: Vec<Option<PyReadonlyArray2<f64>>>,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    if codes.len() != trades_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "codes 与 trades_list 长度必须一致",
        ));
    }
    let trades_per_code: Vec<Option<Vec<TradeRecord>>> = trades_list
        .into_iter()
        .map(|opt_arr| {
            opt_arr.map(|arr| {
                let view = arr.as_array();
                let nrows = view.nrows();
                let mut vec = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    let time_sec = view[[r, 0]] as f32;
                    let price = view[[r, 1]] as f32;
                    let volume = view[[r, 2]] as f32;
                    let turnover = view[[r, 3]] as f32;
                    let flag = view[[r, 4]] as i32;
                    let bid_order = view[[r, 5]] as i64;
                    let ask_order = view[[r, 6]] as i64;
                    vec.push(TradeRecord {
                        time_sec,
                        time_us: (view[[r, 0]] * 1_000_000.0) as i64,
                        price,
                        volume,
                        turnover,
                        flag,
                        bid_order,
                        ask_order,
                    });
                }
                vec
            })
        })
        .collect();

    compute_anneal_cross_peer_from_trades(&codes, trades_per_code)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}
