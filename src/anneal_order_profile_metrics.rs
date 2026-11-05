//! 柜员游戏扩展系列2：多维订单画像与博弈序可还原性差异（anneal_order_profile）。
//!
//! 核心逻辑：
//!   1. 多被还原目标画像：订单总成交量(vol)、拆单成交笔数(subcnt)、存活时长(dur)、
//!      价格穿透跨度(pspan)、均笔成交量(mvol) × 4 时段 × 3 方向 × 16 指标 = 960 维
//!   2. 跨目标可还原性剪刀差（量 vs 笔数、量 vs 时长、量 vs 价格跨度等 6 对）= 1,152 维
//!   3. 多排列坐标轴对比：末次成交离场序(last_ord)、原始报单号序(id_ord)、
//!      订单均价空间序(vwap_ord) vs 首次成交入场序(first_ord) 及其两两差 = 1,440 维
//!   4. 多猜测先验模板（降序/U型/倒U型 vs 升序）+ Spearman 秩还原差 + 大小单位置对齐度 = 1,344 维
//!   5. 对手方博弈错配（扫盘进攻 sweep / 冰山防守 iceberg / 机构对倒 block）
//!      + 高低价位区（high_zone vs low_zone）+ 真实成交 vs 撤单序列(retreat)可还原性对比 = 1,920 维
//!   6. 237 分钟 × 7 个跨画像/跨序通道 × 16 指标（112 列）经 get_features_factors_rust_full 降维 = 8,568 维
//!
//! 合计固定输出：15,384 维。
//! 全程确定性：固定种子 xorshift64 + 严格全序排序，同股同日逐比特一致。

use crate::anneal_regime_diff_metrics::{
    adaptive_steps, anneal_core16, anneal_core16_with_mode, FastAnnealBuf, CORE_NAMES, N_CORE,
    N_MINUTES,
};
use crate::fast_csv_reader::{build_trade_record, read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// ============================================================================
// 常量定义
// ============================================================================

pub const WIN4_NAMES: [&str; 4] = ["fullday", "early30", "mid3h", "late30"];
pub const WIN4_BOUNDS: [(f32, f32); 4] = [
    (0.0, 14_220.0),      // 全天
    (0.0, 1_800.0),       // 开盘30分
    (1_800.0, 12_600.0),  // 盘中3小时
    (12_600.0, 14_220.0), // 收盘30分
];

pub const DIR3_NAMES: [&str; 3] = ["bid", "ask", "mixed"];
pub const TARGET5_NAMES: [&str; 5] = ["vol", "subcnt", "dur", "pspan", "mvol"];
pub const TARGET_PAIRS: [(usize, usize, &str); 6] = [
    (0, 1, "vol_vs_subcnt"),
    (0, 2, "vol_vs_dur"),
    (0, 3, "vol_vs_pspan"),
    (0, 4, "vol_vs_mvol"),
    (1, 2, "subcnt_vs_dur"),
    (2, 3, "dur_vs_pspan"),
];

pub const AXIS3_NAMES: [&str; 3] = ["last_ord", "id_ord", "vwap_ord"];
pub const AXIS_CONTRAST_NAMES: [&str; 6] = [
    "first_vs_last",
    "first_vs_id",
    "first_vs_vwap",
    "last_vs_id",
    "last_vs_vwap",
    "id_vs_vwap",
];

pub const PRIOR3_NAMES: [&str; 3] = ["desc", "ushape", "invu"];
pub const PRIOR_DIFF3_NAMES: [&str; 3] = ["asc_vs_desc", "ushape_vs_asc", "ushape_vs_invu"];

pub const DEEP12_NAMES: [&str; 12] = [
    "spearman_r",
    "pearson_minus_spearman",
    "top10_pos_recall",
    "bot40_pos_recall",
    "top_minus_bot_recall",
    "top10_pos_dist",
    "bot40_pos_dist",
    "top_minus_bot_dist",
    "early10_gain_share",
    "late20_stagnation",
    "accel_curve_area",
    "rank_mae_norm",
];

pub const GAME_SLICE12_NAMES: [&str; 12] = [
    "sweep_actbid",
    "sweep_actask",
    "iceberg_pasbid",
    "iceberg_pasask",
    "block_bid",
    "block_ask",
    "high_zone_bid",
    "high_zone_ask",
    "low_zone_bid",
    "low_zone_ask",
    "retreat_bid",
    "retreat_ask",
];

pub const GAME_DIFF8_NAMES: [&str; 8] = [
    "sweep_bid_vs_ask",
    "iceberg_bid_vs_ask",
    "sweep_vs_iceberg_bid",
    "sweep_vs_iceberg_ask",
    "high_vs_low_zone_bid",
    "high_vs_low_zone_ask",
    "trade_vs_retreat_bid",
    "trade_vs_retreat_ask",
];

pub const S2_MIN_CH7_NAMES: [&str; 7] = [
    "bid_subcnt",
    "ask_subcnt",
    "bid_dur",
    "ask_dur",
    "bid_lastord",
    "ask_lastord",
    "bid_vwapord",
];

pub const N_MIN_COLS: usize = 7 * N_CORE; // 112
pub const N_PART1: usize = 5 * 4 * 3 * N_CORE; // 960
pub const N_PART2: usize = 6 * 4 * 3 * N_CORE; // 1,152
pub const N_PART3: usize = 3 * 3 * 3 * N_CORE + 6 * 3 * 3 * N_CORE + 3 * 3 * N_CORE; // 432 + 864 + 144 = 1,440
pub const N_PART4: usize = 6 * 4 * 3 * N_CORE + 4 * 3 * 12 + 4 * 12; // 1,152 + 144 + 48 = 1,344
pub const N_PART5: usize = 12 * 3 * N_CORE + 8 * 3 * 2 * N_CORE + 12 * 3 * N_CORE; // 576 + 768 + 576 = 1,920
pub const N_PART6: usize = 21 * N_MIN_COLS + N_MIN_COLS * (N_MIN_COLS - 1) / 2; // 8,568

/// 总输出维度：960 + 1,152 + 1,440 + 1,344 + 1,920 + 8,568 = 15,384
pub const EXPECTED_LEN: usize = N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5 + N_PART6;

const EPS: f32 = 1e-6;

// ============================================================================
// 丰富订单画像结构与聚合
// ============================================================================

#[derive(Clone, Copy)]
struct RichOrder {
    order_id: i64,
    vol: f32,
    turnover: f32,
    sub_cnt: u32,
    n_act: u32,
    n_pas: u32,
    first_idx: usize,
    last_idx: usize,
    first_us: i64,
    last_us: i64,
    min_prc: f32,
    max_prc: f32,
}

impl RichOrder {
    #[inline]
    fn dur_sec(&self) -> f32 {
        ((self.last_us - self.first_us).max(0) as f64 * 1e-6) as f32
    }

    #[inline]
    fn price_span_rel(&self) -> f32 {
        let vwap = self.vwap();
        if vwap > 0.0 {
            (self.max_prc - self.min_prc).max(0.0) / vwap
        } else {
            0.0
        }
    }

    #[inline]
    fn mean_vol(&self) -> f32 {
        if self.sub_cnt > 0 {
            self.vol / self.sub_cnt as f32
        } else {
            self.vol
        }
    }

    #[inline]
    fn vwap(&self) -> f32 {
        if self.vol > 0.0 {
            self.turnover / self.vol
        } else {
            self.min_prc
        }
    }

    #[inline]
    fn target_val(&self, target_idx: usize) -> f32 {
        match target_idx {
            0 => self.vol,
            1 => self.sub_cnt as f32,
            2 => self.dur_sec(),
            3 => self.price_span_rel(),
            4 => self.mean_vol(),
            _ => self.vol,
        }
    }
}

fn aggregate_rich_orders(
    trades: &[TradeRecord],
) -> (FxHashMap<i64, RichOrder>, FxHashMap<i64, RichOrder>) {
    let mut bid_map: FxHashMap<i64, RichOrder> =
        FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());
    let mut ask_map: FxHashMap<i64, RichOrder> =
        FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());

    for (i, t) in trades.iter().enumerate() {
        let be = bid_map.entry(t.bid_order).or_insert(RichOrder {
            order_id: t.bid_order,
            vol: 0.0,
            turnover: 0.0,
            sub_cnt: 0,
            n_act: 0,
            n_pas: 0,
            first_idx: i,
            last_idx: i,
            first_us: t.time_us,
            last_us: t.time_us,
            min_prc: t.price,
            max_prc: t.price,
        });
        be.vol += t.volume;
        be.turnover += t.turnover;
        be.sub_cnt += 1;
        be.last_idx = i;
        if t.time_us < be.first_us {
            be.first_us = t.time_us;
        }
        if t.time_us > be.last_us {
            be.last_us = t.time_us;
        }
        if t.price < be.min_prc {
            be.min_prc = t.price;
        }
        if t.price > be.max_prc {
            be.max_prc = t.price;
        }
        if t.flag == 66 {
            be.n_act += 1;
        } else if t.flag == 83 {
            be.n_pas += 1;
        }

        let ae = ask_map.entry(t.ask_order).or_insert(RichOrder {
            order_id: t.ask_order,
            vol: 0.0,
            turnover: 0.0,
            sub_cnt: 0,
            n_act: 0,
            n_pas: 0,
            first_idx: i,
            last_idx: i,
            first_us: t.time_us,
            last_us: t.time_us,
            min_prc: t.price,
            max_prc: t.price,
        });
        ae.vol += t.volume;
        ae.turnover += t.turnover;
        ae.sub_cnt += 1;
        ae.last_idx = i;
        if t.time_us < ae.first_us {
            ae.first_us = t.time_us;
        }
        if t.time_us > ae.last_us {
            ae.last_us = t.time_us;
        }
        if t.price < ae.min_prc {
            ae.min_prc = t.price;
        }
        if t.price > ae.max_prc {
            ae.max_prc = t.price;
        }
        if t.flag == 83 {
            ae.n_act += 1;
        } else if t.flag == 66 {
            ae.n_pas += 1;
        }
    }
    (bid_map, ask_map)
}

/// 按首次出现顺序排序返回 (bid_orders, ask_orders, mixed_orders)
fn sorted_orders_by_first(
    bid_map: &FxHashMap<i64, RichOrder>,
    ask_map: &FxHashMap<i64, RichOrder>,
) -> [Vec<RichOrder>; 3] {
    let mut b_vec: Vec<RichOrder> = bid_map.values().copied().collect();
    b_vec.sort_by_key(|o| (o.first_idx, o.order_id));

    let mut a_vec: Vec<RichOrder> = ask_map.values().copied().collect();
    a_vec.sort_by_key(|o| (o.first_idx, o.order_id));

    let mut m_vec: Vec<(usize, u8, RichOrder)> = Vec::with_capacity(b_vec.len() + a_vec.len());
    m_vec.extend(b_vec.iter().map(|&o| (o.first_idx, 0u8, o)));
    m_vec.extend(a_vec.iter().map(|&o| (o.first_idx, 1u8, o)));
    m_vec.sort_by_key(|x| (x.0, x.1, x.2.order_id));
    let m_orders: Vec<RichOrder> = m_vec.into_iter().map(|x| x.2).collect();

    [b_vec, a_vec, m_orders]
}

/// 按指定坐标轴排序提取 volume 序列：
/// axis_mode = 0: last_idx (末次成交离场序)
/// axis_mode = 1: order_id (原始报单流水号序)
/// axis_mode = 2: vwap (成交均价空间序)
fn extract_vol_by_axis(orders: &[RichOrder], axis_mode: usize) -> Vec<f32> {
    let mut tmp: Vec<RichOrder> = orders.to_vec();
    match axis_mode {
        0 => tmp.sort_by_key(|o| (o.last_idx, o.first_idx, o.order_id)),
        1 => tmp.sort_by_key(|o| (o.order_id, o.first_idx)),
        2 => tmp.sort_by(|a, b| {
            a.vwap()
                .total_cmp(&b.vwap())
                .then_with(|| a.first_idx.cmp(&b.first_idx))
                .then_with(|| a.order_id.cmp(&b.order_id))
        }),
        _ => {}
    }
    tmp.into_iter().map(|o| o.vol).collect()
}

// ============================================================================
// 深度位置对齐与 Spearman 还原度分析（12 维）
// ============================================================================

fn ordinal_rank_f32(vals: &[f32]) -> Vec<f32> {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]).then_with(|| a.cmp(&b)));
    let mut out = vec![0.0f32; n];
    for (rank, &i) in idx.iter().enumerate() {
        out[i] = rank as f32;
    }
    out
}

/// 计算退火后的深度对齐指标（12 维）：
/// 0: spearman_r (退火后 guess 与 true_seq 的秩相关)
/// 1: pearson_minus_spearman (线性还原度 - 秩还原度)
/// 2: top10_pos_recall (大单 Top10% 位置命中率)
/// 3: bot40_pos_recall (小单 Bot40% 位置命中率)
/// 4: top_minus_bot_recall (大单位置命中率 - 小单位置命中率)
/// 5: top10_pos_dist (真实大单与其退火匹配值的位置归一化距离)
/// 6: bot40_pos_dist (真实小单的位置归一化距离)
/// 7: top_minus_bot_dist (大小单位置距离差)
/// 8: early10_gain_share (前 10% 步数完成的还原度占比)
/// 9: late20_stagnation (后 20% 步数无改善步数占比)
/// 10: accel_curve_area (归一化面积曲率)
/// 11: rank_mae_norm (退火后归一化秩绝对误差)
fn compute_deep_alignment12(seq: &[f32], m_max: usize, buf: &mut FastAnnealBuf) -> [f32; 12] {
    let n = seq.len();
    if n < 10 || m_max < 20 {
        return [f32::NAN; 12];
    }
    let core = anneal_core16(seq, m_max, buf);
    let pearson_r = core[1];
    if !pearson_r.is_finite() || buf.guess.len() != n {
        return [f32::NAN; 12];
    }

    let r_true = ordinal_rank_f32(seq);
    let r_guess = ordinal_rank_f32(&buf.guess);
    let nf = n as f32;

    let spearman_r = features::corr_pair(&r_true, &r_guess);
    let top_thresh = (0.90 * nf).floor();
    let bot_thresh = (0.40 * nf).ceil();

    let mut top_cnt = 0usize;
    let mut top_hit = 0usize;
    let mut top_dist_sum = 0.0f32;

    let mut bot_cnt = 0usize;
    let mut bot_hit = 0usize;
    let mut bot_dist_sum = 0.0f32;

    let mut mae_sum = 0.0f32;

    for i in 0..n {
        let rt = r_true[i];
        let rg = r_guess[i];
        let r_diff = (rt - rg).abs() / nf;
        mae_sum += r_diff;

        if rt >= top_thresh {
            top_cnt += 1;
            if rg >= top_thresh {
                top_hit += 1;
            }
            top_dist_sum += r_diff;
        }
        if rt < bot_thresh {
            bot_cnt += 1;
            if rg < bot_thresh {
                bot_hit += 1;
            }
            bot_dist_sum += r_diff;
        }
    }

    let top_rec = if top_cnt > 0 {
        top_hit as f32 / top_cnt as f32
    } else {
        f32::NAN
    };
    let bot_rec = if bot_cnt > 0 {
        bot_hit as f32 / bot_cnt as f32
    } else {
        f32::NAN
    };
    let top_dist = if top_cnt > 0 {
        top_dist_sum / top_cnt as f32
    } else {
        f32::NAN
    };
    let bot_dist = if bot_cnt > 0 {
        bot_dist_sum / bot_cnt as f32
    } else {
        f32::NAN
    };

    let early_share = (1.0 - core[4]).clamp(0.0, 1.0);
    let late_stag = (1.0 - core[12].abs()).clamp(0.0, 1.0);
    let accel_area = pearson_r - (1.0 - core[7]);
    let rank_mae = mae_sum / nf;

    [
        spearman_r,
        pearson_r - spearman_r,
        top_rec,
        bot_rec,
        top_rec - bot_rec,
        top_dist,
        bot_dist,
        top_dist - bot_dist,
        early_share,
        late_stag,
        accel_area,
        rank_mae,
    ]
}

#[inline]
fn safe_diff(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        a - b
    } else {
        f32::NAN
    }
}

#[inline]
fn safe_ratio(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        a / (b.abs() + EPS)
    } else {
        f32::NAN
    }
}

// ============================================================================
// 纯计算核心：compute_anneal_order_profile_from_records
// ============================================================================

pub fn compute_anneal_order_profile_from_records(
    trades_no_retreat: &[TradeRecord],
    trades_with_retreat: &[TradeRecord],
) -> Vec<f32> {
    if trades_no_retreat.len() < 10 {
        return vec![f32::NAN; EXPECTED_LEN];
    }

    let mut out = Vec::with_capacity(EXPECTED_LEN);
    let mut buf = FastAnnealBuf::new();

    let first_sec = trades_no_retreat[0].time_sec;
    let day_base = ((first_sec as i64) - ((first_sec as i64).rem_euclid(86400))) as f32;
    let t_open = day_base + 9.0 * 3600.0 + 30.0 * 60.0;
    let t_base = if (first_sec - t_open).abs() < 7200.0 {
        t_open
    } else {
        first_sec
    };

    // 预计算 4 个宏观窗口的 RichOrder 列表（按首次成交序）
    let mut win4_orders: [Option<[Vec<RichOrder>; 3]>; 4] = [None, None, None, None];
    for (w_idx, &(s_lo, s_hi)) in WIN4_BOUNDS.iter().enumerate() {
        let lo = trades_no_retreat.partition_point(|t| t.time_sec < t_base + s_lo);
        let hi = trades_no_retreat.partition_point(|t| t.time_sec < t_base + s_hi);
        if hi > lo + 2 {
            let (b_map, a_map) = aggregate_rich_orders(&trades_no_retreat[lo..hi]);
            win4_orders[w_idx] = Some(sorted_orders_by_first(&b_map, &a_map));
        }
    }

    // ------------------------------------------------------------------------
    // 模块 1：5 种目标画像 (vol/subcnt/dur/pspan/mvol) × 4 时段 × 3 方向 × 16 指标 = 960 维
    // ------------------------------------------------------------------------
    let mut target_metrics = [[[[f32::NAN; N_CORE]; 3]; 4]; 5];
    for t_idx in 0..5 {
        for w_idx in 0..4 {
            if let Some(ref dirs) = win4_orders[w_idx] {
                for d_idx in 0..3 {
                    let seq: Vec<f32> = dirs[d_idx].iter().map(|o| o.target_val(t_idx)).collect();
                    let m_steps = adaptive_steps(seq.len(), 20_000);
                    target_metrics[t_idx][w_idx][d_idx] = anneal_core16(&seq, m_steps, &mut buf);
                }
            }
            for d_idx in 0..3 {
                out.extend_from_slice(&target_metrics[t_idx][w_idx][d_idx]);
            }
        }
    }
    debug_assert_eq!(out.len(), N_PART1);

    // ------------------------------------------------------------------------
    // 模块 2：6 对跨目标画像可还原性剪刀差 × 4 时段 × 3 方向 × 16 指标 = 1,152 维
    // ------------------------------------------------------------------------
    for &(ta, tb, _) in &TARGET_PAIRS {
        for w_idx in 0..4 {
            for d_idx in 0..3 {
                for k in 0..N_CORE {
                    out.push(safe_diff(
                        target_metrics[ta][w_idx][d_idx][k],
                        target_metrics[tb][w_idx][d_idx][k],
                    ));
                }
            }
        }
    }
    debug_assert_eq!(out.len(), N_PART1 + N_PART2);

    // ------------------------------------------------------------------------
    // 模块 3：多排列坐标轴对比 (last_ord / id_ord / vwap_ord vs first_ord) = 1,440 维
    // 3 种新坐标轴 × 3 时段(fullday=0, early30=1, late30=3) × 3 方向 × 16 指标 = 432 维
    // 6 对坐标轴两两差 × 3 时段 × 3 方向 × 16 指标 = 864 维
    // 额外：mid3h(w=2) 下 3 种新坐标轴 × 3 方向 × 16 指标 = 144 维 -> 合计 1,440 维
    // ------------------------------------------------------------------------
    let axis_wins = [0usize, 1usize, 3usize]; // fullday, early30, late30
    let mut axis_metrics = [[[[f32::NAN; N_CORE]; 3]; 3]; 4]; // [axis: 0=first, 1=last, 2=id, 3=vwap][w_i: 0..3][d_idx: 0..3]
    for (wi, &w_idx) in axis_wins.iter().enumerate() {
        for d_idx in 0..3 {
            axis_metrics[0][wi][d_idx] = target_metrics[0][w_idx][d_idx];
        }
        if let Some(ref dirs) = win4_orders[w_idx] {
            for ax in 0..3 {
                for d_idx in 0..3 {
                    let seq = extract_vol_by_axis(&dirs[d_idx], ax);
                    let m_steps = adaptive_steps(seq.len(), 20_000);
                    axis_metrics[ax + 1][wi][d_idx] = anneal_core16(&seq, m_steps, &mut buf);
                }
            }
        }
    }
    // 输出 3 新坐标轴 × 3 时段 × 3 方向 × 16 = 432
    for ax in 1..4 {
        for wi in 0..3 {
            for d_idx in 0..3 {
                out.extend_from_slice(&axis_metrics[ax][wi][d_idx]);
            }
        }
    }
    // 输出 6 对坐标轴差 × 3 时段 × 3 方向 × 16 = 864
    let ax_pairs: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for &(aa, ab) in &ax_pairs {
        for wi in 0..3 {
            for d_idx in 0..3 {
                for k in 0..N_CORE {
                    out.push(safe_diff(
                        axis_metrics[aa][wi][d_idx][k],
                        axis_metrics[ab][wi][d_idx][k],
                    ));
                }
            }
        }
    }
    // 输出 mid3h (w_idx=2) 的 3 新坐标轴 × 3 方向 × 16 = 144
    for ax in 0..3 {
        for d_idx in 0..3 {
            let vals = if let Some(ref dirs) = win4_orders[2] {
                let seq = extract_vol_by_axis(&dirs[d_idx], ax);
                anneal_core16(&seq, adaptive_steps(seq.len(), 20_000), &mut buf)
            } else {
                [f32::NAN; N_CORE]
            };
            out.extend_from_slice(&vals);
        }
    }
    debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3);

    // ------------------------------------------------------------------------
    // 模块 4：多先验猜测模板 (降序/U型/倒U型 vs 升序) + 深度位置对齐与 Spearman 差 = 1,344 维
    // 3 种非升序先验 × 4 时段 × 3 方向 × 16 指标 = 576 维
    // 3 组先验对比差 × 4 时段 × 3 方向 × 16 指标 = 576 维
    // 4 时段 × 3 方向 × 12 深度对齐指标 = 144 维
    // 4 时段 × (bid-ask, bid_first-bid_last, ask_first-ask_last, vol-subcnt) × 12 深度指标 = 48 维 -> 合计 1,344 维
    // ------------------------------------------------------------------------
    let mut prior_metrics = [[[[f32::NAN; N_CORE]; 3]; 4]; 4]; // [mode 0..4][w 0..4][d 0..3]
    for w_idx in 0..4 {
        for d_idx in 0..3 {
            prior_metrics[0][w_idx][d_idx] = target_metrics[0][w_idx][d_idx];
        }
        if let Some(ref dirs) = win4_orders[w_idx] {
            for mode in 1..4usize {
                for d_idx in 0..3 {
                    let seq: Vec<f32> = dirs[d_idx].iter().map(|o| o.vol).collect();
                    let m_steps = adaptive_steps(seq.len(), 18_000);
                    prior_metrics[mode][w_idx][d_idx] =
                        anneal_core16_with_mode(&seq, m_steps, mode as u8, &mut buf);
                }
            }
        }
    }
    // 3 新先验原始值: 576
    for mode in 1..4 {
        for w_idx in 0..4 {
            for d_idx in 0..3 {
                out.extend_from_slice(&prior_metrics[mode][w_idx][d_idx]);
            }
        }
    }
    // 3 先验差值 (asc-desc, ushape-asc, ushape-invu): 576
    for &(ma, mb) in &[(0usize, 1usize), (2usize, 0usize), (2usize, 3usize)] {
        for w_idx in 0..4 {
            for d_idx in 0..3 {
                for k in 0..N_CORE {
                    out.push(safe_diff(
                        prior_metrics[ma][w_idx][d_idx][k],
                        prior_metrics[mb][w_idx][d_idx][k],
                    ));
                }
            }
        }
    }
    // 深度对齐 4 时段 × 3 方向 × 12 指标 = 144 维
    let mut deep_mat = [[[f32::NAN; 12]; 3]; 4];
    for w_idx in 0..4 {
        if let Some(ref dirs) = win4_orders[w_idx] {
            for d_idx in 0..3 {
                let seq: Vec<f32> = dirs[d_idx].iter().map(|o| o.vol).collect();
                deep_mat[w_idx][d_idx] =
                    compute_deep_alignment12(&seq, adaptive_steps(seq.len(), 18_000), &mut buf);
            }
        }
        for d_idx in 0..3 {
            out.extend_from_slice(&deep_mat[w_idx][d_idx]);
        }
    }
    // 4 时段 × 买卖深度对齐差 (bid - ask) × 12 指标 = 48 维
    for w_idx in 0..4 {
        for k in 0..12 {
            out.push(safe_diff(deep_mat[w_idx][0][k], deep_mat[w_idx][1][k]));
        }
    }
    debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3 + N_PART4);

    // ------------------------------------------------------------------------
    // 模块 5：对手方博弈错配（大吃小扫盘 / 小吃大冰山 / 大对大对倒）
    //         + 高低价格带 + 真实成交 vs 撤单序列 = 1,920 维
    // ------------------------------------------------------------------------
    // 在 3 个窗口 (fullday=0, early30=1, late30=3) 下构造 12 个博弈/价格/撤单子集
    let game_wins = [0usize, 1usize, 3usize];
    let mut game_metrics = [[[f32::NAN; N_CORE]; 3]; 12]; // [12 slices][3 wins]

    for (wi, &w_idx) in game_wins.iter().enumerate() {
        let (s_lo, s_hi) = WIN4_BOUNDS[w_idx];
        // 先处理撤单序列 (slice 10=retreat_bid, 11=retreat_ask)
        let r_lo = trades_with_retreat.partition_point(|t| t.time_sec < t_base + s_lo);
        let r_hi = trades_with_retreat.partition_point(|t| t.time_sec < t_base + s_hi);
        if r_hi > r_lo {
            let mut ret_bid = Vec::new();
            let mut ret_ask = Vec::new();
            for t in &trades_with_retreat[r_lo..r_hi] {
                if t.flag == 32 {
                    if t.bid_order > 0 && (t.ask_order == 0 || t.bid_order >= t.ask_order) {
                        ret_bid.push(t.volume);
                    }
                    if t.ask_order > 0 && (t.bid_order == 0 || t.ask_order > t.bid_order) {
                        ret_ask.push(t.volume);
                    }
                }
            }
            game_metrics[10][wi] =
                anneal_core16(&ret_bid, adaptive_steps(ret_bid.len(), 15_000), &mut buf);
            game_metrics[11][wi] =
                anneal_core16(&ret_ask, adaptive_steps(ret_ask.len(), 15_000), &mut buf);
        }

        if let Some(ref dirs) = win4_orders[w_idx] {
            let b_ords = &dirs[0];
            let a_ords = &dirs[1];

            // 计算订单量 70% 分位作为大单阈值，计算窗口 VWAP 中位数切高低价区
            let b_vol_p70 = {
                let mut v: Vec<f32> = b_ords.iter().map(|o| o.vol).collect();
                v.sort_by(|a, b| a.total_cmp(b));
                if v.is_empty() {
                    0.0
                } else {
                    v[(v.len() * 7 / 10).min(v.len() - 1)]
                }
            };
            let a_vol_p70 = {
                let mut v: Vec<f32> = a_ords.iter().map(|o| o.vol).collect();
                v.sort_by(|a, b| a.total_cmp(b));
                if v.is_empty() {
                    0.0
                } else {
                    v[(v.len() * 7 / 10).min(v.len() - 1)]
                }
            };
            let prc_med = {
                let mut v: Vec<f32> = dirs[2].iter().map(|o| o.vwap()).collect();
                v.sort_by(|a, b| a.total_cmp(b));
                if v.is_empty() {
                    0.0
                } else {
                    v[v.len() / 2]
                }
            };

            // 0: sweep_actbid (主买且拆单笔数>=2，扫盘进攻买单)
            let s0: Vec<f32> = b_ords
                .iter()
                .filter(|o| o.n_act > 0 && o.sub_cnt >= 2)
                .map(|o| o.vol)
                .collect();
            // 1: sweep_actask (主卖且拆单笔数>=2，扫盘进攻卖单)
            let s1: Vec<f32> = a_ords
                .iter()
                .filter(|o| o.n_act > 0 && o.sub_cnt >= 2)
                .map(|o| o.vol)
                .collect();
            // 2: iceberg_pasbid (被动买且笔数>=2，冰山防守买单)
            let s2: Vec<f32> = b_ords
                .iter()
                .filter(|o| o.n_pas > 0 && o.sub_cnt >= 2)
                .map(|o| o.vol)
                .collect();
            // 3: iceberg_pasask (被动卖且笔数>=2，冰山防守卖单)
            let s3: Vec<f32> = a_ords
                .iter()
                .filter(|o| o.n_pas > 0 && o.sub_cnt >= 2)
                .map(|o| o.vol)
                .collect();
            // 4: block_bid (大单但少笔数 sub_cnt<=2，一锤定音大买单)
            let s4: Vec<f32> = b_ords
                .iter()
                .filter(|o| o.vol >= b_vol_p70 && o.sub_cnt <= 2)
                .map(|o| o.vol)
                .collect();
            // 5: block_ask (大单但少笔数 sub_cnt<=2，一锤定音大卖单)
            let s5: Vec<f32> = a_ords
                .iter()
                .filter(|o| o.vol >= a_vol_p70 && o.sub_cnt <= 2)
                .map(|o| o.vol)
                .collect();
            // 6: high_zone_bid (高价区买单)
            let s6: Vec<f32> = b_ords
                .iter()
                .filter(|o| o.vwap() >= prc_med)
                .map(|o| o.vol)
                .collect();
            // 7: high_zone_ask (高价区卖单)
            let s7: Vec<f32> = a_ords
                .iter()
                .filter(|o| o.vwap() >= prc_med)
                .map(|o| o.vol)
                .collect();
            // 8: low_zone_bid (低价区买单)
            let s8: Vec<f32> = b_ords
                .iter()
                .filter(|o| o.vwap() < prc_med)
                .map(|o| o.vol)
                .collect();
            // 9: low_zone_ask (低价区卖单)
            let s9: Vec<f32> = a_ords
                .iter()
                .filter(|o| o.vwap() < prc_med)
                .map(|o| o.vol)
                .collect();

            for (sl_i, seq) in [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9].iter().enumerate() {
                game_metrics[sl_i][wi] =
                    anneal_core16(seq, adaptive_steps(seq.len(), 15_000), &mut buf);
            }
        }
    }

    // 5a: 12 个博弈/价格/撤单子集 × 3 窗口 × 16 指标 = 576 维
    for sl_i in 0..12 {
        for wi in 0..3 {
            out.extend_from_slice(&game_metrics[sl_i][wi]);
        }
    }
    // 5b: 8 组 핵심博弈对比 × 3 窗口 × 16 指标 (diff + ratio) = 768 维
    // (0 vs 1, 2 vs 3, 0 vs 2, 1 vs 3, 6 vs 8, 7 vs 9, trade_bid vs retreat_bid, trade_ask vs retreat_ask)
    for wi in 0..3 {
        let w_idx = game_wins[wi];
        let pairs_vals: [([f32; N_CORE], [f32; N_CORE]); 8] = [
            (game_metrics[0][wi], game_metrics[1][wi]),
            (game_metrics[2][wi], game_metrics[3][wi]),
            (game_metrics[0][wi], game_metrics[2][wi]),
            (game_metrics[1][wi], game_metrics[3][wi]),
            (game_metrics[6][wi], game_metrics[8][wi]),
            (game_metrics[7][wi], game_metrics[9][wi]),
            (target_metrics[0][w_idx][0], game_metrics[10][wi]),
            (target_metrics[0][w_idx][1], game_metrics[11][wi]),
        ];
        for (va, vb) in &pairs_vals {
            for k in 0..N_CORE {
                out.push(safe_diff(va[k], vb[k]));
            }
            for k in 0..N_CORE {
                out.push(safe_ratio(va[k], vb[k]));
            }
        }
    }
    // 5c: 跨目标在博弈子集上的扩展：全天 12 个博弈子集在 (subcnt, dur, mvol) 3 个画像上的可还原性 (12 × 3 × 16 = 576 维)
    // 576 + 768 + 576 = 1,920 维
    if let Some(ref dirs) = win4_orders[0] {
        let b_ords = &dirs[0];
        let a_ords = &dirs[1];
        let b_vol_p70 = {
            let mut v: Vec<f32> = b_ords.iter().map(|o| o.vol).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            if v.is_empty() {
                0.0
            } else {
                v[(v.len() * 7 / 10).min(v.len() - 1)]
            }
        };
        let a_vol_p70 = {
            let mut v: Vec<f32> = a_ords.iter().map(|o| o.vol).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            if v.is_empty() {
                0.0
            } else {
                v[(v.len() * 7 / 10).min(v.len() - 1)]
            }
        };
        let prc_med = {
            let mut v: Vec<f32> = dirs[2].iter().map(|o| o.vwap()).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            if v.is_empty() {
                0.0
            } else {
                v[v.len() / 2]
            }
        };

        let sub_orders: [Vec<RichOrder>; 12] = [
            b_ords.iter().copied().filter(|o| o.n_act > 0 && o.sub_cnt >= 2).collect(),
            a_ords.iter().copied().filter(|o| o.n_act > 0 && o.sub_cnt >= 2).collect(),
            b_ords.iter().copied().filter(|o| o.n_pas > 0 && o.sub_cnt >= 2).collect(),
            a_ords.iter().copied().filter(|o| o.n_pas > 0 && o.sub_cnt >= 2).collect(),
            b_ords.iter().copied().filter(|o| o.vol >= b_vol_p70 && o.sub_cnt <= 2).collect(),
            a_ords.iter().copied().filter(|o| o.vol >= a_vol_p70 && o.sub_cnt <= 2).collect(),
            b_ords.iter().copied().filter(|o| o.vwap() >= prc_med).collect(),
            a_ords.iter().copied().filter(|o| o.vwap() >= prc_med).collect(),
            b_ords.iter().copied().filter(|o| o.vwap() < prc_med).collect(),
            a_ords.iter().copied().filter(|o| o.vwap() < prc_med).collect(),
            b_ords.iter().copied().filter(|o| o.n_act > 0 && o.n_pas > 0).collect(),
            a_ords.iter().copied().filter(|o| o.n_act > 0 && o.n_pas > 0).collect(),
        ];
        for sl_ords in &sub_orders {
            for &t_idx in &[1usize, 2usize, 4usize] {
                let seq: Vec<f32> = sl_ords.iter().map(|o| o.target_val(t_idx)).collect();
                let vals = anneal_core16(&seq, adaptive_steps(seq.len(), 12_000), &mut buf);
                out.extend_from_slice(&vals);
            }
        }
    } else {
        out.extend(std::iter::repeat(f32::NAN).take(12 * 3 * N_CORE));
    }
    debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5);

    // ------------------------------------------------------------------------
    // 模块 6：237 分钟 × 7 个跨画像/跨序通道 × 16 指标（112 列）降维 = 8,568 维
    // 7 个通道：
    //   0: bid_subcnt (买单拆单笔数可还原性)
    //   1: ask_subcnt (卖单拆单笔数可还原性)
    //   2: bid_dur    (买单存活时长可还原性)
    //   3: ask_dur    (卖单存活时长可还原性)
    //   4: bid_lastord(买单离场序成交量可还原性)
    //   5: ask_lastord(卖单离场序成交量可还原性)
    //   6: bid_vwapord(买单均价空间序成交量可还原性)
    // ------------------------------------------------------------------------
    let mut min_mat = Array2::<f32>::from_elem((N_MINUTES, N_MIN_COLS), f32::NAN);
    for m_idx in 0..N_MINUTES {
        let lo = trades_no_retreat.partition_point(|t| t.time_sec < t_base + m_idx as f32 * 60.0);
        let hi =
            trades_no_retreat.partition_point(|t| t.time_sec < t_base + (m_idx + 1) as f32 * 60.0);
        if hi <= lo + 2 {
            continue;
        }
        let (b_map, a_map) = aggregate_rich_orders(&trades_no_retreat[lo..hi]);
        let dirs = sorted_orders_by_first(&b_map, &a_map);

        let seqs: [Vec<f32>; 7] = [
            dirs[0].iter().map(|o| o.sub_cnt as f32).collect(),
            dirs[1].iter().map(|o| o.sub_cnt as f32).collect(),
            dirs[0].iter().map(|o| o.dur_sec()).collect(),
            dirs[1].iter().map(|o| o.dur_sec()).collect(),
            extract_vol_by_axis(&dirs[0], 0),
            extract_vol_by_axis(&dirs[1], 0),
            extract_vol_by_axis(&dirs[0], 2),
        ];
        for (c_i, seq) in seqs.iter().enumerate() {
            let vals = anneal_core16(seq, adaptive_steps(seq.len(), 5_000), &mut buf);
            for k in 0..N_CORE {
                min_mat[[m_idx, c_i * N_CORE + k]] = vals[k];
            }
        }
    }

    let min_col_names = build_s2_min_col_names();
    let (reduced, _) =
        features::get_features_factors_rust_full(&min_mat.view(), &min_col_names, false);
    out.extend_from_slice(&reduced);

    debug_assert_eq!(out.len(), EXPECTED_LEN);
    out
}

// ============================================================================
// 因子名生成（与 compute_anneal_order_profile_from_records 严格逐一对齐）
// ============================================================================

fn build_s2_min_col_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_MIN_COLS);
    for ch in &S2_MIN_CH7_NAMES {
        for m in &CORE_NAMES {
            names.push(format!("s2_min_{}_{}", ch, m));
        }
    }
    names
}

pub fn anneal_order_profile_names() -> Vec<String> {
    let mut names = Vec::with_capacity(EXPECTED_LEN);

    // 模块 1: 960
    for t_name in &TARGET5_NAMES {
        for w_name in &WIN4_NAMES {
            for d_name in &DIR3_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_tgt_{}_{}_{}_{}", t_name, w_name, d_name, m));
                }
            }
        }
    }

    // 模块 2: 1,152
    for &(_, _, p_name) in &TARGET_PAIRS {
        for w_name in &WIN4_NAMES {
            for d_name in &DIR3_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_tgtdiff_{}_{}_{}_{}", p_name, w_name, d_name, m));
                }
            }
        }
    }

    // 模块 3: 1,440
    let axis_win_names = ["fullday", "early30", "late30"];
    for ax_name in &AXIS3_NAMES {
        for w_name in &axis_win_names {
            for d_name in &DIR3_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_axis_{}_{}_{}_{}", ax_name, w_name, d_name, m));
                }
            }
        }
    }
    for ac_name in &AXIS_CONTRAST_NAMES {
        for w_name in &axis_win_names {
            for d_name in &DIR3_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_axisdiff_{}_{}_{}_{}", ac_name, w_name, d_name, m));
                }
            }
        }
    }
    for ax_name in &AXIS3_NAMES {
        for d_name in &DIR3_NAMES {
            for m in &CORE_NAMES {
                names.push(format!("s2_axis_{}_mid3h_{}_{}", ax_name, d_name, m));
            }
        }
    }

    // 模块 4: 1,344
    for pr_name in &PRIOR3_NAMES {
        for w_name in &WIN4_NAMES {
            for d_name in &DIR3_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_prior_{}_{}_{}_{}", pr_name, w_name, d_name, m));
                }
            }
        }
    }
    for pd_name in &PRIOR_DIFF3_NAMES {
        for w_name in &WIN4_NAMES {
            for d_name in &DIR3_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_priordiff_{}_{}_{}_{}", pd_name, w_name, d_name, m));
                }
            }
        }
    }
    for w_name in &WIN4_NAMES {
        for d_name in &DIR3_NAMES {
            for dm in &DEEP12_NAMES {
                names.push(format!("s2_deep_{}_{}_{}", w_name, d_name, dm));
            }
        }
    }
    for w_name in &WIN4_NAMES {
        for dm in &DEEP12_NAMES {
            names.push(format!("s2_deepdiff_{}_bid_minus_ask_{}", w_name, dm));
        }
    }

    // 模块 5: 1,920
    for sl_name in &GAME_SLICE12_NAMES {
        for w_name in &axis_win_names {
            for m in &CORE_NAMES {
                names.push(format!("s2_game_{}_{}_{}", sl_name, w_name, m));
            }
        }
    }
    for w_name in &axis_win_names {
        for gd_name in &GAME_DIFF8_NAMES {
            for m in &CORE_NAMES {
                names.push(format!("s2_gamediff_{}_{}_{}", gd_name, w_name, m));
            }
            for m in &CORE_NAMES {
                names.push(format!("s2_gameratio_{}_{}_{}", gd_name, w_name, m));
            }
        }
    }
    let extra_sl_names = [
        "sweep_actbid",
        "sweep_actask",
        "iceberg_pasbid",
        "iceberg_pasask",
        "block_bid",
        "block_ask",
        "high_zone_bid",
        "high_zone_ask",
        "low_zone_bid",
        "low_zone_ask",
        "dual_role_bid",
        "dual_role_ask",
    ];
    for sl_name in &extra_sl_names {
        for t_name in &["subcnt", "dur", "mvol"] {
            for m in &CORE_NAMES {
                names.push(format!("s2_gametgt_{}_{}_{}", sl_name, t_name, m));
            }
        }
    }

    // 模块 6: 8,568
    let min_col_names = build_s2_min_col_names();
    let dummy = Array2::<f32>::ones((2, N_MIN_COLS));
    let (_, red_names) =
        features::get_features_factors_rust_full(&dummy.view(), &min_col_names, false);
    names.extend(red_names);

    assert_eq!(names.len(), EXPECTED_LEN);
    names
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

pub fn compute_anneal_order_profile_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    // 一次读取含撤单的完整逐笔记录（with_retreat=true），再拆分出不含撤单的子序列，避免读两次磁盘
    let trades_all = read_trade_fast_inner(code, date, true, true, usize::MAX)?;
    let trades_no_retreat: Vec<TradeRecord> =
        trades_all.iter().copied().filter(|t| t.flag != 32).collect();
    Ok(compute_anneal_order_profile_from_records(
        &trades_no_retreat,
        &trades_all,
    ))
}

#[pyfunction]
pub fn py_anneal_order_profile(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_order_profile_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_anneal_order_profile_names() -> Vec<String> {
    anneal_order_profile_names()
}

#[pyfunction]
pub fn py_anneal_order_profile_from_data(
    _py: Python<'_>,
    exchtime_us: PyReadonlyArray1<i64>,
    price: PyReadonlyArray1<f32>,
    volume: PyReadonlyArray1<f32>,
    turnover: PyReadonlyArray1<f32>,
    flag: PyReadonlyArray1<i32>,
    bid_order: PyReadonlyArray1<i64>,
    ask_order: PyReadonlyArray1<i64>,
) -> PyResult<Vec<f32>> {
    let t = exchtime_us.as_array();
    let p = price.as_array();
    let v = volume.as_array();
    let tn = turnover.as_array();
    let f = flag.as_array();
    let bo = bid_order.as_array();
    let ao = ask_order.as_array();
    let n = t.len();
    if p.len() != n
        || v.len() != n
        || tn.len() != n
        || f.len() != n
        || bo.len() != n
        || ao.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有输入数组长度必须一致",
        ));
    }
    let trades_all: Vec<TradeRecord> = (0..n)
        .filter_map(|i| build_trade_record(t[i], p[i], v[i], tn[i], f[i], bo[i], ao[i], true))
        .collect();
    let trades_no_retreat: Vec<TradeRecord> =
        trades_all.iter().copied().filter(|r| r.flag != 32).collect();
    Ok(compute_anneal_order_profile_from_records(
        &trades_no_retreat,
        &trades_all,
    ))
}
