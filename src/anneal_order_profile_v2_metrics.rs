//! 柜员游戏扩展系列2 修正版：多维订单画像与博弈序可还原性差异（anneal_order_profile_v2）。
//!
//! 旧版 anneal_order_profile 保持不动，本文件是独立的新函数。相对旧版的修正：
//!   1. 时间：所有时段、分钟切分改用精确到微秒的 time_us。旧版用 f32 的 time_sec，
//!      在 2022 年附近精度只有 128 秒，237 个分钟窗口里约 126 个是空的，开盘基准偏到 09:29:36。
//!   2. 订单数混淆：旧版退火步数封顶 2 万步，订单越多越还原不动，全天还原度与订单个数秩相关约 -0.9。
//!      新版每条序列都配一个「同一批数值随机打乱」的对照（打乱用固定种子，从排序后的数值出发，
//!      因此对照只取决于数值集合本身），输出「真实顺序 − 打乱对照」的超额，只保留顺序信息。
//!   3. 与顺序无关的两项（posD_ratio、sign_flip_prob）不再当作还原度输出，
//!      其背后真正的信息「整手单 / 同量订单占比」改为显式的成交量结构指标（lot 模块）。
//!   4. 升序与降序 / U 型 / 倒 U 型先验统一步数上限；坐标轴、先验共用同一份打乱对照，差值里不再混入步数差和对照噪声。
//!   5. 深度对齐不再额外跑一次退火（复用真实序列那次的结果），删掉 3 个只是现有列换算的指标；
//!      秩用平均秩处理并列（成交量大量并列，旧版按下标硬排会让命中率随机）。
//!   6. 删掉「甲 ÷ (|乙| + 极小数)」的比值列，只保留差值。
//!   7. 高低价区用严格大于 / 小于中位数，等于中位数的订单不进任何一侧（价格离散，旧版高价区明显偏多）；
//!      「冰山 / 扫盘 / 一锤定音」改成名副其实的名字：被动多次成交 / 主动多次成交 / 大单少笔（严格大于 70 分位）。
//!   8. 撤单方向只认「一侧订单号为 0」的撤单记录，去掉旧版 bid_order >= ask_order 这种无含义的判定。
//!   9. 分钟模块去掉「按均价排序」通道（一分钟内价格几乎不变，退化成首次成交顺序），加入成交量通道和量结构通道。
//!
//! 速度：退火核心复制了一份接受「已排序数组」的版本，同一批数值只排一次序（见 anneal_presorted）。
//!
//! 固定输出 7873 列（见 EXPECTED_LEN），确定性：所有随机数都来自固定种子，HashMap 遍历结果先排序再用。

use crate::anneal_regime_diff_metrics::{adaptive_steps, FastAnnealBuf, CORE_NAMES, N_CORE};
use crate::fast_csv_reader::{build_trade_record, read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// ============================================================================
// 常量
// ============================================================================

const US: i64 = 1_000_000;
const N_MINUTES: usize = 237;

pub const WIN4_NAMES: [&str; 4] = ["fullday", "early30", "mid3h", "late30"];
/// 相对 09:30 的秒数区间（下午已前移 90 分钟，14220 秒 = 14:57）
const WIN4_BOUNDS_SEC: [(i64, i64); 4] = [(0, 14_220), (0, 1_800), (1_800, 12_600), (12_600, 14_220)];

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

/// 坐标轴：0 = 首次成交入场序（即 tgt_vol），1..3 为新坐标轴
pub const AXIS3_NAMES: [&str; 3] = ["last_ord", "id_ord", "vwap_ord"];
pub const AXIS_PAIRS: [(usize, usize, &str); 6] = [
    (0, 1, "first_vs_last"),
    (0, 2, "first_vs_id"),
    (0, 3, "first_vs_vwap"),
    (1, 2, "last_vs_id"),
    (1, 3, "last_vs_vwap"),
    (2, 3, "id_vs_vwap"),
];

/// 先验：0 = 升序（即 tgt_vol），1 降序，2 U 型，3 倒 U 型
pub const PRIOR3_NAMES: [&str; 3] = ["desc", "ushape", "invu"];
pub const PRIOR_PAIRS: [(usize, usize, &str); 3] =
    [(0, 1, "asc_vs_desc"), (2, 0, "ushape_vs_asc"), (2, 3, "ushape_vs_invu")];

/// 与顺序有关的 14 个核心指标 = CORE_NAMES[0..14]（去掉 posD_ratio、sign_flip_prob）
pub const N_ORD: usize = 14;

pub const DEEP8_NAMES: [&str; 8] = [
    "spearman_r",
    "top10_pos_recall",
    "bot40_pos_recall",
    "top_minus_bot_recall",
    "top10_pos_dist",
    "bot40_pos_dist",
    "top_minus_bot_dist",
    "rank_mae_norm",
];

/// 成交量结构（整手 / 同量订单）指标
pub const LOT7_NAMES: [&str; 7] = [
    "eq100_share",      // 恰好 100 股的订单占比
    "odd_lot_share",    // 非 100 整数倍（零股）的订单占比
    "round1000_share",  // 1000 股整数倍的订单占比
    "round10000_share", // 10000 股整数倍的订单占比
    "mode_share",       // 最常见成交量的订单占比
    "distinct_ratio",   // 不同成交量取值个数 / 订单数
    "tie_prob",         // 随机抽两单成交量恰好相同的概率
];

pub const GAME_SLICE12_NAMES: [&str; 12] = [
    "actmulti_bid",
    "actmulti_ask",
    "pasmulti_bid",
    "pasmulti_ask",
    "bigfew_bid",
    "bigfew_ask",
    "highprc_bid",
    "highprc_ask",
    "lowprc_bid",
    "lowprc_ask",
    "retreat_bid",
    "retreat_ask",
];
/// 博弈对比：下标 < 12 指 game 子集，100 / 101 指本时段买 / 卖全部订单的 tgt_vol
const GAME_PAIRS: [(usize, usize, &str); 8] = [
    (0, 1, "actmulti_bid_vs_ask"),
    (2, 3, "pasmulti_bid_vs_ask"),
    (0, 2, "act_vs_pas_bid"),
    (1, 3, "act_vs_pas_ask"),
    (6, 8, "high_vs_low_bid"),
    (7, 9, "high_vs_low_ask"),
    (100, 10, "trade_vs_retreat_bid"),
    (101, 11, "trade_vs_retreat_ask"),
];
pub const GAMETGT_SLICE12_NAMES: [&str; 12] = [
    "actmulti_bid",
    "actmulti_ask",
    "pasmulti_bid",
    "pasmulti_ask",
    "bigfew_bid",
    "bigfew_ask",
    "highprc_bid",
    "highprc_ask",
    "lowprc_bid",
    "lowprc_ask",
    "dualrole_bid",
    "dualrole_ask",
];
const GAMETGT_TARGETS: [usize; 3] = [1, 2, 4]; // subcnt, dur, mvol

pub const MIN_CH6_NAMES: [&str; 6] = ["bid_vol", "ask_vol", "bid_subcnt", "ask_subcnt", "bid_dur", "ask_dur"];
/// 分钟矩阵每个通道保留的核心指标下标：r0_init, final_r, half_life_norm, inertia_norm, decline_ratio, dr_std
const MIN_CORE_IDX: [usize; 6] = [0, 1, 4, 7, 8, 10];
pub const MIN_LOT4_NAMES: [&str; 4] = ["bid_tie_prob", "ask_tie_prob", "bid_mode_share", "ask_mode_share"];
const N_MIN_COLS: usize = 6 * 6 + 4; // 40

/// 退火步数上限：时段级统一 2 万步，分钟级 3000 步（分钟级每条序列还要跑一次打乱对照，控制在 1 秒预算内）
const CAP_WIN: usize = 20_000;
const CAP_MIN: usize = 3_000;
/// 打乱对照次数：时段级 2 次，分钟级 1 次
const N_SHUF_WIN: usize = 2;
const N_SHUF_MIN: usize = 1;
const SHUF_SEED: u64 = 0xA5A5_1234_5678_9ABC;

pub const N_PART_TGT: usize = 5 * 4 * 3 * N_ORD + 6 * 4 * 3 * N_ORD; // 840 + 1008
pub const N_PART_AXIS: usize = 3 * 4 * 3 * N_ORD + 6 * 4 * 3 * N_ORD; // 504 + 1008
pub const N_PART_PRIOR: usize = 3 * 4 * 3 * N_ORD + 3 * 4 * 3 * N_ORD; // 504 + 504
pub const N_PART_DEEP: usize = 4 * 3 * 8 + 4 * 8; // 96 + 32
pub const N_PART_GAME: usize = 12 * 4 * N_ORD + 8 * 4 * N_ORD + 12 * 3 * N_ORD; // 672 + 448 + 504
pub const N_PART_LOT: usize = 4 * 3 * 7 + 4 * 7 + 3 * 7; // 84 + 28 + 21
pub const N_PART_MIN: usize = 21 * N_MIN_COLS + N_MIN_COLS * (N_MIN_COLS - 1) / 2; // 1620

/// 总列数 7873
pub const EXPECTED_LEN: usize =
    N_PART_TGT + N_PART_AXIS + N_PART_PRIOR + N_PART_DEEP + N_PART_GAME + N_PART_LOT + N_PART_MIN;

const NAN_ORD: [f32; N_ORD] = [f32::NAN; N_ORD];
const NAN8: [f32; 8] = [f32::NAN; 8];
const NAN7: [f32; 7] = [f32::NAN; 7];

// ============================================================================
// 订单画像
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
    fn new(id: i64, i: usize, t: &TradeRecord) -> Self {
        RichOrder {
            order_id: id,
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
        }
    }
    #[inline]
    fn add(&mut self, i: usize, t: &TradeRecord, active: bool) {
        self.vol += t.volume;
        self.turnover += t.turnover;
        self.sub_cnt += 1;
        self.last_idx = i;
        self.first_us = self.first_us.min(t.time_us);
        self.last_us = self.last_us.max(t.time_us);
        self.min_prc = self.min_prc.min(t.price);
        self.max_prc = self.max_prc.max(t.price);
        if active {
            self.n_act += 1;
        } else {
            self.n_pas += 1;
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
    fn target_val(&self, k: usize) -> f32 {
        match k {
            0 => self.vol,
            1 => self.sub_cnt as f32,
            2 => ((self.last_us - self.first_us).max(0) as f64 * 1e-6) as f32,
            3 => {
                let v = self.vwap();
                if v > 0.0 {
                    (self.max_prc - self.min_prc).max(0.0) / v
                } else {
                    0.0
                }
            }
            _ => {
                if self.sub_cnt > 0 {
                    self.vol / self.sub_cnt as f32
                } else {
                    self.vol
                }
            }
        }
    }
}

/// 复用的聚合容器（避免每分钟重新分配哈希表）
struct OrderAgg {
    bid: FxHashMap<i64, RichOrder>,
    ask: FxHashMap<i64, RichOrder>,
}

impl OrderAgg {
    fn new(cap: usize) -> Self {
        Self {
            bid: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
            ask: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
        }
    }

    /// 聚合一段成交为订单，返回按首次成交序排好的 [买, 卖, 买卖混合]
    fn build(&mut self, trades: &[TradeRecord]) -> [Vec<RichOrder>; 3] {
        self.bid.clear();
        self.ask.clear();
        for (i, t) in trades.iter().enumerate() {
            self.bid
                .entry(t.bid_order)
                .or_insert_with(|| RichOrder::new(t.bid_order, i, t))
                .add(i, t, t.flag == 66);
            self.ask
                .entry(t.ask_order)
                .or_insert_with(|| RichOrder::new(t.ask_order, i, t))
                .add(i, t, t.flag == 83);
        }
        // HashMap 遍历顺序随机，必须按 (首次成交位置, 订单号) 排序后再用
        let mut b: Vec<RichOrder> = self.bid.values().copied().collect();
        b.sort_by_key(|o| (o.first_idx, o.order_id));
        let mut a: Vec<RichOrder> = self.ask.values().copied().collect();
        a.sort_by_key(|o| (o.first_idx, o.order_id));
        let mut m: Vec<(usize, u8, i64, RichOrder)> = Vec::with_capacity(b.len() + a.len());
        m.extend(b.iter().map(|o| (o.first_idx, 0u8, o.order_id, *o)));
        m.extend(a.iter().map(|o| (o.first_idx, 1u8, o.order_id, *o)));
        m.sort_by_key(|x| (x.0, x.1, x.2));
        let mixed = m.into_iter().map(|x| x.3).collect();
        [b, a, mixed]
    }
}

fn vol_by_axis(orders: &[RichOrder], axis: usize) -> Vec<f32> {
    let mut tmp: Vec<RichOrder> = orders.to_vec();
    match axis {
        1 => tmp.sort_by_key(|o| (o.last_idx, o.first_idx, o.order_id)),
        2 => tmp.sort_by_key(|o| (o.order_id, o.first_idx)),
        3 => tmp.sort_by(|a, b| {
            a.vwap()
                .total_cmp(&b.vwap())
                .then_with(|| a.first_idx.cmp(&b.first_idx))
                .then_with(|| a.order_id.cmp(&b.order_id))
        }),
        _ => {}
    }
    tmp.iter().map(|o| o.vol).collect()
}

// ============================================================================
// 退火核心：逐行复制自 anneal_regime_diff_metrics::anneal_core16_with_mode，唯一区别是
// 初始升序猜测直接用调用方传入的已排序数组，不在每次调用里重新排序。同一批数值在一个
// 时段方向里要退火 9 次（真实、2 次打乱、3 个坐标轴、3 个先验），只需排一次序。
// 随机数生成器、种子、接受规则与原函数完全相同，结果逐位一致。
// ============================================================================

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_DEAD_BEEF
            } else {
                seed
            },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.next_u64() >> 32).wrapping_mul(n as u64) >> 32) as usize
    }
}

fn anneal_presorted(
    seq: &[f32],
    pre_sorted: &[f32],
    m_max: usize,
    mode: u8,
    buf: &mut FastAnnealBuf,
) -> [f32; N_CORE] {
    let n = seq.len();
    if n < 3 || m_max < 10 {
        return [f32::NAN; N_CORE];
    }
    let mut sum = 0.0f64;
    for &v in seq {
        if !v.is_finite() {
            return [f32::NAN; N_CORE];
        }
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let mut sq_sum = 0.0f64;
    for &v in seq {
        let d = v as f64 - mean;
        sq_sum += d * d;
    }
    let sigma2 = sq_sum / n as f64;
    if sigma2 <= 1e-20 {
        return [f32::NAN; N_CORE];
    }

    // 构造初始猜测序列
    buf.guess.clear();
    buf.guess.extend_from_slice(pre_sorted);
    if mode == 1 {
        // 降序先验
        buf.guess.reverse();
    } else if mode == 2 {
        // U型先验：大值放两端，小值放中间
        let sorted = buf.guess.clone();
        let mut left = 0usize;
        let mut right = n - 1;
        for (idx, &val) in sorted.iter().rev().enumerate() {
            if idx % 2 == 0 {
                buf.guess[left] = val;
                left += 1;
            } else {
                buf.guess[right] = val;
                right = right.saturating_sub(1);
            }
        }
    } else if mode == 3 {
        // 倒U型先验：小值放两端，大值放中间
        let sorted = buf.guess.clone();
        let mut left = 0usize;
        let mut right = n - 1;
        for (idx, &val) in sorted.iter().enumerate() {
            if idx % 2 == 0 {
                buf.guess[left] = val;
                left += 1;
            } else {
                buf.guess[right] = val;
                right = right.saturating_sub(1);
            }
        }
    }

    let mut s = 0.0f64;
    for k in 0..n {
        let d = buf.guess[k] as f64 - seq[k] as f64;
        s += d * d;
    }
    let denom = 2.0 * sigma2 * n as f64;
    let inv_denom = 1.0 / denom;
    let r0 = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
    let hl_target = (1.0 + r0) * 0.5;
    let s_tol = (sigma2 * n as f64 * 1e-10).max(1e-12);
    let inv_m = 1.0 / (m_max as f64 - 1.0).max(1.0);

    let mut rng = XorShift64::new(SEED ^ ((mode as u64) << 32));
    let mut prev_r = r0;
    let mut final_r = r0;
    let mut peak_r = r0;
    let mut max_dd = 0.0f64;

    let mut a1_step = m_max;
    let mut a2_step = m_max;
    let mut a3_step = m_max;

    let mut inertia_sum = 0.0f64;
    let mut declines = 0u32;
    let mut dr_cnt = 0u32;
    let mut dr_s1 = 0.0f64;
    let mut dr_s2 = 0.0f64;

    let s1_end = m_max / 3;
    let s2_end = 2 * (m_max / 3);
    let mut s1_cnt = 0u32;
    let mut s1_s1 = 0.0f64;
    let mut s1_s2 = 0.0f64;
    let mut s2_cnt = 0u32;
    let mut s2_decl = 0u32;
    let mut s3_cnt = 0u32;
    let mut s3_s1 = 0.0f64;
    let mut s3_s2 = 0.0f64;
    let mut r_at_s2 = r0;

    let mut d_cnt = 0u32;
    let mut d_pos = 0u32;
    let mut d_flips = 0u32;
    let mut prev_d_sign: i8 = 0;

    let mut actual_steps = 0usize;

    for t in 0..m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = (i + 1) % n;
        }

        if d_cnt < 2000 && buf.guess[i] != buf.guess[j] {
            let dv = seq[j] - seq[i];
            d_cnt += 1;
            if dv > 0.0 {
                d_pos += 1;
            }
            let cur_sign: i8 = if dv > 0.0 {
                1
            } else if dv < 0.0 {
                -1
            } else {
                0
            };
            if prev_d_sign != 0 && cur_sign != 0 && cur_sign != prev_d_sign {
                d_flips += 1;
            }
            if cur_sign != 0 {
                prev_d_sign = cur_sign;
            }
        }

        let gi = buf.guess[i];
        let gj = buf.guess[j];
        let ti = seq[i];
        let tj = seq[j];
        let ds = 2.0f32 * (gi - gj) * (ti - tj);
        let ct = sigma2 * (1.0 - t as f64 * inv_m).max(0.0);
        if (ds as f64) < 0.0 || (ds as f64) < ct {
            s += ds as f64;
            buf.guess.swap(i, j);
        }

        let cr = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
        final_r = cr;
        actual_steps = t + 1;

        if t > 0 {
            let dv = cr - prev_r;
            dr_cnt += 1;
            dr_s1 += dv;
            dr_s2 += dv * dv;
            if cr < prev_r {
                declines += 1;
                if t >= s1_end && t < s2_end {
                    s2_decl += 1;
                }
            }
            if t < s1_end {
                s1_cnt += 1;
                s1_s1 += dv;
                s1_s2 += dv * dv;
            } else if t < s2_end {
                s2_cnt += 1;
            } else {
                s3_cnt += 1;
                s3_s1 += dv;
                s3_s2 += dv * dv;
            }
        }

        if a1_step == m_max && cr >= hl_target {
            a1_step = t;
        }
        if a2_step == m_max && cr >= 0.80 {
            a2_step = t;
        }
        if a3_step == m_max && cr >= 0.90 {
            a3_step = t;
        }
        inertia_sum += 1.0 - cr;
        if cr > peak_r {
            peak_r = cr;
        } else {
            let dd = peak_r - cr;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        if t + 1 == s2_end {
            r_at_s2 = cr;
        }
        prev_r = cr;
        if s <= s_tol {
            if t + 1 < s2_end {
                r_at_s2 = cr;
            }
            break;
        }
    }

    let m_f = m_max as f32;
    let act_f = actual_steps.max(1) as f32;
    let r0_f = r0 as f32;
    let fin_f = final_r as f32;

    let dr_std = if dr_cnt >= 2 {
        let nd = dr_cnt as f64;
        ((dr_s2 - dr_s1 * dr_s1 / nd) / (nd - 1.0)).max(0.0).sqrt() as f32
    } else {
        0.0
    };
    let hot_std = if s1_cnt >= 2 {
        let nd = s1_cnt as f64;
        ((s1_s2 - s1_s1 * s1_s1 / nd) / (nd - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };
    let cold_std = if s3_cnt >= 2 {
        let nd = s3_cnt as f64;
        ((s3_s2 - s3_s1 * s3_s1 / nd) / (nd - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };

    let mut out = [f32::NAN; N_CORE];
    out[0] = r0_f;
    out[1] = fin_f;
    out[2] = fin_f - r0_f.abs();
    out[3] = (fin_f - r0_f) / (1.0 - r0_f + 1e-4);
    out[4] = a1_step as f32 / m_f;
    out[5] = a2_step as f32 / m_f;
    out[6] = a3_step as f32 / m_f;
    out[7] = (inertia_sum / act_f as f64) as f32;
    out[8] = declines as f32 / act_f;
    out[9] = max_dd as f32;
    out[10] = dr_std;
    out[11] = if s2_cnt > 0 {
        s2_decl as f32 / s2_cnt as f32
    } else {
        0.0
    };
    out[12] = (final_r - r_at_s2) as f32;
    out[13] = (hot_std / (cold_std + 1e-6)) as f32;
    out[14] = if d_cnt > 0 {
        d_pos as f32 / d_cnt as f32
    } else {
        0.5
    };
    out[15] = if d_cnt > 1 {
        d_flips as f32 / (d_cnt - 1) as f32
    } else {
        0.5
    };
    out
}

// ============================================================================
// 退火 + 打乱对照
// ============================================================================

#[inline]
fn xorshift(st: &mut u64) -> u64 {
    let mut x = *st;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *st = x;
    x
}

struct Ev {
    buf: FastAnnealBuf,
    shuf: Vec<f32>,
    idx: Vec<usize>,
    ra: Vec<f32>,
    rb: Vec<f32>,
}

impl Ev {
    fn new() -> Self {
        Self {
            buf: FastAnnealBuf::new(),
            shuf: Vec::with_capacity(4096),
            idx: Vec::with_capacity(4096),
            ra: Vec::with_capacity(4096),
            rb: Vec::with_capacity(4096),
        }
    }
}

/// 平均秩（并列取平均名次），结果为 0..n-1 之间的名次
fn avg_rank(vals: &[f32], idx: &mut Vec<usize>, out: &mut Vec<f32>) {
    let n = vals.len();
    idx.clear();
    idx.extend(0..n);
    idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]).then_with(|| a.cmp(&b)));
    out.clear();
    out.resize(n, 0.0);
    let mut s = 0usize;
    while s < n {
        let mut e = s + 1;
        while e < n && vals[idx[e]] == vals[idx[s]] {
            e += 1;
        }
        let r = (s + e - 1) as f32 * 0.5;
        for &k in &idx[s..e] {
            out[k] = r;
        }
        s = e;
    }
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 {
        return f32::NAN;
    }
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for i in 0..n {
        sa += a[i] as f64;
        sb += b[i] as f64;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 1e-20 || sbb <= 1e-20 {
        return f32::NAN;
    }
    (sab / (saa * sbb).sqrt()) as f32
}

/// 退火结束后猜测序列与真实序列的位置对齐度（8 维）
fn deep8(seq: &[f32], guess: &[f32], idx: &mut Vec<usize>, ra: &mut Vec<f32>, rb: &mut Vec<f32>) -> [f32; 8] {
    let n = seq.len();
    if n < 10 || guess.len() != n {
        return NAN8;
    }
    avg_rank(seq, idx, ra);
    avg_rank(guess, idx, rb);
    let spearman = pearson(ra, rb);
    let inv = 1.0 / (n as f32 - 1.0);
    let (mut tc, mut th, mut td) = (0usize, 0usize, 0.0f32);
    let (mut bc, mut bh, mut bd) = (0usize, 0usize, 0.0f32);
    let mut mae = 0.0f32;
    for i in 0..n {
        let qt = ra[i] * inv;
        let qg = rb[i] * inv;
        let d = (qt - qg).abs();
        mae += d;
        if qt >= 0.9 {
            tc += 1;
            th += (qg >= 0.9) as usize;
            td += d;
        }
        if qt <= 0.4 {
            bc += 1;
            bh += (qg <= 0.4) as usize;
            bd += d;
        }
    }
    let f = |h: f32, c: usize| if c > 0 { h / c as f32 } else { f32::NAN };
    let top_rec = f(th as f32, tc);
    let bot_rec = f(bh as f32, bc);
    let top_d = f(td, tc);
    let bot_d = f(bd, bc);
    [spearman, top_rec, bot_rec, top_rec - bot_rec, top_d, bot_d, top_d - bot_d, mae / n as f32]
}

/// 打乱对照（sorted 为该批数值升序排好的数组）：从排好序的数值出发、用固定种子打乱 n_shuf 次，各跑一次升序先验退火，取平均。
/// 对照只取决于数值集合，与输入顺序无关，所以同一批订单的不同坐标轴 / 先验共用一份对照。
fn shuffle_baseline(sorted: &[f32], cap: usize, n_shuf: usize, want_deep: bool, ev: &mut Ev) -> ([f32; N_ORD], [f32; 8]) {
    let n = sorted.len();
    if n < 3 || n_shuf == 0 {
        return (NAN_ORD, NAN8);
    }
    let m = adaptive_steps(n, cap);
    let mut acc = [0.0f64; N_ORD];
    let mut accd = [0.0f64; 8];
    let mut cntd = [0u32; 8];
    for s in 0..n_shuf {
        ev.shuf.clear();
        ev.shuf.extend_from_slice(sorted);
        let mut st = SHUF_SEED ^ ((s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)) ^ (n as u64);
        if st == 0 {
            st = 1;
        }
        for i in (1..n).rev() {
            let j = (xorshift(&mut st) % (i as u64 + 1)) as usize;
            ev.shuf.swap(i, j);
        }
        let c = anneal_presorted(&ev.shuf, sorted, m, 0, &mut ev.buf);
        if !c[1].is_finite() {
            return (NAN_ORD, NAN8);
        }
        for k in 0..N_ORD {
            acc[k] += c[k] as f64;
        }
        if want_deep {
            let d = deep8(&ev.shuf, &ev.buf.guess, &mut ev.idx, &mut ev.ra, &mut ev.rb);
            for k in 0..8 {
                if d[k].is_finite() {
                    accd[k] += d[k] as f64;
                    cntd[k] += 1;
                }
            }
        }
    }
    let mut base = NAN_ORD;
    for k in 0..N_ORD {
        base[k] = (acc[k] / n_shuf as f64) as f32;
    }
    let mut based = NAN8;
    if want_deep {
        for k in 0..8 {
            if cntd[k] > 0 {
                based[k] = (accd[k] / cntd[k] as f64) as f32;
            }
        }
    }
    (base, based)
}

/// 真实顺序跑一次退火（指定先验），可选同时算深度对齐
fn run_true(seq: &[f32], sorted: &[f32], cap: usize, mode: u8, want_deep: bool, ev: &mut Ev) -> ([f32; N_CORE], [f32; 8]) {
    if seq.len() < 3 {
        return ([f32::NAN; N_CORE], NAN8);
    }
    let m = adaptive_steps(seq.len(), cap);
    let c = anneal_presorted(seq, sorted, m, mode, &mut ev.buf);
    let d = if want_deep && c[1].is_finite() {
        deep8(seq, &ev.buf.guess, &mut ev.idx, &mut ev.ra, &mut ev.rb)
    } else {
        NAN8
    };
    (c, d)
}

#[inline]
fn sd(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        a - b
    } else {
        f32::NAN
    }
}

fn excess(core: &[f32; N_CORE], base: &[f32; N_ORD]) -> [f32; N_ORD] {
    let mut o = NAN_ORD;
    for k in 0..N_ORD {
        o[k] = sd(core[k], base[k]);
    }
    o
}

/// 真实顺序相对打乱对照的超额（一步到位：对照 + 真实）
fn excess_seq(seq: &[f32], cap: usize, n_shuf: usize, ev: &mut Ev) -> [f32; N_ORD] {
    if seq.len() < 3 {
        return NAN_ORD;
    }
    let sorted = sorted_copy(seq);
    let (base, _) = shuffle_baseline(&sorted, cap, n_shuf, false, ev);
    let (c, _) = run_true(seq, &sorted, cap, 0, false, ev);
    excess(&c, &base)
}

fn sorted_copy(seq: &[f32]) -> Vec<f32> {
    let mut v = seq.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

fn diff_ord(a: &[f32; N_ORD], b: &[f32; N_ORD]) -> [f32; N_ORD] {
    let mut o = NAN_ORD;
    for k in 0..N_ORD {
        o[k] = sd(a[k], b[k]);
    }
    o
}

// ============================================================================
// 成交量结构（整手 / 同量订单）
// ============================================================================

fn lot7(vols: &[f32], keys: &mut Vec<u64>) -> [f32; 7] {
    let n = vols.len();
    if n < 3 {
        return NAN7;
    }
    keys.clear();
    keys.extend(vols.iter().map(|&v| v.max(0.0).round() as u64));
    let nf = n as f32;
    let (mut e100, mut odd, mut r1k, mut r10k) = (0usize, 0usize, 0usize, 0usize);
    for &k in keys.iter() {
        e100 += (k == 100) as usize;
        odd += (k % 100 != 0) as usize;
        r1k += (k > 0 && k % 1000 == 0) as usize;
        r10k += (k > 0 && k % 10000 == 0) as usize;
    }
    keys.sort_unstable();
    let (mut runs, mut max_run, mut sq) = (0usize, 0usize, 0.0f64);
    let mut s = 0usize;
    while s < n {
        let mut e = s + 1;
        while e < n && keys[e] == keys[s] {
            e += 1;
        }
        let len = e - s;
        runs += 1;
        max_run = max_run.max(len);
        sq += (len * len) as f64;
        s = e;
    }
    [
        e100 as f32 / nf,
        odd as f32 / nf,
        r1k as f32 / nf,
        r10k as f32 / nf,
        max_run as f32 / nf,
        runs as f32 / nf,
        (sq / (n as f64 * n as f64)) as f32,
    ]
}

// ============================================================================
// 主计算
// ============================================================================

fn quantile_sorted(mut v: Vec<f32>, num: usize, den: usize) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[(v.len() * num / den).min(v.len() - 1)])
}

/// 12 个博弈子集的订单列表（slice 10/11 为「同时有主动和被动成交」的订单，仅 gametgt 使用）
fn game_order_slices(dirs: &[Vec<RichOrder>; 3]) -> [Vec<RichOrder>; 12] {
    let (b, a) = (&dirs[0], &dirs[1]);
    let bp70 = quantile_sorted(b.iter().map(|o| o.vol).collect(), 7, 10).unwrap_or(f32::INFINITY);
    let ap70 = quantile_sorted(a.iter().map(|o| o.vol).collect(), 7, 10).unwrap_or(f32::INFINITY);
    let med = quantile_sorted(dirs[2].iter().map(|o| o.vwap()).collect(), 1, 2).unwrap_or(f32::NAN);
    let f = |v: &Vec<RichOrder>, p: &dyn Fn(&RichOrder) -> bool| -> Vec<RichOrder> {
        v.iter().copied().filter(|o| p(o)).collect()
    };
    [
        f(b, &|o| o.n_act > 0 && o.sub_cnt >= 2),
        f(a, &|o| o.n_act > 0 && o.sub_cnt >= 2),
        f(b, &|o| o.n_pas > 0 && o.sub_cnt >= 2),
        f(a, &|o| o.n_pas > 0 && o.sub_cnt >= 2),
        f(b, &|o| o.vol > bp70 && o.sub_cnt <= 2),
        f(a, &|o| o.vol > ap70 && o.sub_cnt <= 2),
        f(b, &|o| o.vwap() > med),
        f(a, &|o| o.vwap() > med),
        f(b, &|o| o.vwap() < med),
        f(a, &|o| o.vwap() < med),
        f(b, &|o| o.n_act > 0 && o.n_pas > 0),
        f(a, &|o| o.n_act > 0 && o.n_pas > 0),
    ]
}

/// 稳定按 time_us 排序（输入已有序时不动）
fn ensure_time_sorted(v: &mut Vec<TradeRecord>) {
    if !v.windows(2).all(|w| w[0].time_us <= w[1].time_us) {
        v.sort_by_key(|t| t.time_us);
    }
}

pub fn compute_anneal_order_profile_v2_from_records(
    trades: &[TradeRecord],
    trades_all: &[TradeRecord],
) -> Vec<f32> {
    if trades.len() < 10 {
        return vec![f32::NAN; EXPECTED_LEN];
    }
    let mut out: Vec<f32> = Vec::with_capacity(EXPECTED_LEN);
    let mut ev = Ev::new();
    let mut agg = OrderAgg::new(trades.len() / 2 + 1);
    let mut keys: Vec<u64> = Vec::with_capacity(4096);

    // time_us 已含 +8 小时与下午前移 90 分钟，按本地日历日取 09:30 为基准
    let day_us = 86_400 * US;
    let t_open = trades[0].time_us.div_euclid(day_us) * day_us + (9 * 3600 + 30 * 60) * US;
    let win_range = |v: &[TradeRecord], lo_s: i64, hi_s: i64| -> (usize, usize) {
        let lo = v.partition_point(|t| t.time_us < t_open + lo_s * US);
        let hi = v.partition_point(|t| t.time_us < t_open + hi_s * US);
        (lo, hi)
    };

    // ---------------- 各时段 × 各方向的基础计算 ----------------
    let mut tgt = [[[NAN_ORD; 3]; 4]; 5]; // [target][win][dir]
    let mut axis = [[[NAN_ORD; 3]; 4]; 4]; // [axis][win][dir]，axis 0 = tgt_vol
    let mut prior = [[[NAN_ORD; 3]; 4]; 4]; // [mode][win][dir]，mode 0 = tgt_vol
    let mut deep = [[NAN8; 3]; 4];
    let mut lot = [[NAN7; 3]; 4];
    let mut game = [[NAN_ORD; 4]; 12]; // [slice][win]
    let mut gametgt = [[NAN_ORD; 3]; 12]; // 全天 [slice][target]

    for w in 0..4 {
        let (s_lo, s_hi) = WIN4_BOUNDS_SEC[w];
        let (lo, hi) = win_range(trades, s_lo, s_hi);
        let (rlo, rhi) = win_range(trades_all, s_lo, s_hi);

        // 撤单序列（一侧订单号为 0 的撤单记录，按时间顺序）
        let mut ret_b = Vec::new();
        let mut ret_a = Vec::new();
        for t in &trades_all[rlo..rhi] {
            if t.flag == 32 {
                if t.bid_order > 0 && t.ask_order == 0 {
                    ret_b.push(t.volume);
                } else if t.ask_order > 0 && t.bid_order == 0 {
                    ret_a.push(t.volume);
                }
            }
        }
        game[10][w] = excess_seq(&ret_b, CAP_WIN, N_SHUF_WIN, &mut ev);
        game[11][w] = excess_seq(&ret_a, CAP_WIN, N_SHUF_WIN, &mut ev);

        if hi < lo + 3 {
            continue;
        }
        let dirs = agg.build(&trades[lo..hi]);
        for d in 0..3 {
            let ords = &dirs[d];
            let vol_seq: Vec<f32> = ords.iter().map(|o| o.vol).collect();
            lot[w][d] = lot7(&vol_seq, &mut keys);
            // 5 种画像：各自对照 + 真实
            for t in 0..5 {
                let seq: Vec<f32> = ords.iter().map(|o| o.target_val(t)).collect();
                let want_deep = t == 0;
                let sorted = sorted_copy(&seq);
                let (base, based) = shuffle_baseline(&sorted, CAP_WIN, N_SHUF_WIN, want_deep, &mut ev);
                let (c, dd) = run_true(&seq, &sorted, CAP_WIN, 0, want_deep, &mut ev);
                tgt[t][w][d] = excess(&c, &base);
                if t == 0 {
                    for k in 0..8 {
                        deep[w][d][k] = sd(dd[k], based[k]);
                    }
                    axis[0][w][d] = tgt[0][w][d];
                    prior[0][w][d] = tgt[0][w][d];
                    // 新坐标轴：同一批成交量换顺序，共用 vol 的对照
                    for ax in 1..4 {
                        let s2 = vol_by_axis(ords, ax);
                        let (c2, _) = run_true(&s2, &sorted, CAP_WIN, 0, false, &mut ev);
                        axis[ax][w][d] = excess(&c2, &base);
                    }
                    // 新先验：同一步数上限，共用 vol 的对照
                    for mode in 1..4u8 {
                        let (c3, _) = run_true(&seq, &sorted, CAP_WIN, mode, false, &mut ev);
                        prior[mode as usize][w][d] = excess(&c3, &base);
                    }
                }
            }
        }
        // 博弈子集（成交量序列）
        let slices = game_order_slices(&dirs);
        for sl in 0..10 {
            let seq: Vec<f32> = slices[sl].iter().map(|o| o.vol).collect();
            game[sl][w] = excess_seq(&seq, CAP_WIN, N_SHUF_WIN, &mut ev);
        }
        if w == 0 {
            for sl in 0..12 {
                for (ti, &t) in GAMETGT_TARGETS.iter().enumerate() {
                    let seq: Vec<f32> = slices[sl].iter().map(|o| o.target_val(t)).collect();
                    gametgt[sl][ti] = excess_seq(&seq, CAP_WIN, N_SHUF_WIN, &mut ev);
                }
            }
        }
    }

    // ---------------- 输出：画像 ----------------
    for t in 0..5 {
        for w in 0..4 {
            for d in 0..3 {
                out.extend_from_slice(&tgt[t][w][d]);
            }
        }
    }
    for &(a, b, _) in &TARGET_PAIRS {
        for w in 0..4 {
            for d in 0..3 {
                out.extend_from_slice(&diff_ord(&tgt[a][w][d], &tgt[b][w][d]));
            }
        }
    }
    // ---------------- 输出：坐标轴 ----------------
    for ax in 1..4 {
        for w in 0..4 {
            for d in 0..3 {
                out.extend_from_slice(&axis[ax][w][d]);
            }
        }
    }
    for &(a, b, _) in &AXIS_PAIRS {
        for w in 0..4 {
            for d in 0..3 {
                out.extend_from_slice(&diff_ord(&axis[a][w][d], &axis[b][w][d]));
            }
        }
    }
    // ---------------- 输出：先验 ----------------
    for mode in 1..4 {
        for w in 0..4 {
            for d in 0..3 {
                out.extend_from_slice(&prior[mode][w][d]);
            }
        }
    }
    for &(a, b, _) in &PRIOR_PAIRS {
        for w in 0..4 {
            for d in 0..3 {
                out.extend_from_slice(&diff_ord(&prior[a][w][d], &prior[b][w][d]));
            }
        }
    }
    // ---------------- 输出：深度对齐 ----------------
    for w in 0..4 {
        for d in 0..3 {
            out.extend_from_slice(&deep[w][d]);
        }
    }
    for w in 0..4 {
        for k in 0..8 {
            out.push(sd(deep[w][0][k], deep[w][1][k]));
        }
    }
    // ---------------- 输出：博弈 ----------------
    for sl in 0..12 {
        for w in 0..4 {
            out.extend_from_slice(&game[sl][w]);
        }
    }
    for &(a, b, _) in &GAME_PAIRS {
        for w in 0..4 {
            let va = match a {
                100 => tgt[0][w][0],
                101 => tgt[0][w][1],
                _ => game[a][w],
            };
            out.extend_from_slice(&diff_ord(&va, &game[b][w]));
        }
    }
    for sl in 0..12 {
        for ti in 0..3 {
            out.extend_from_slice(&gametgt[sl][ti]);
        }
    }
    // ---------------- 输出：成交量结构 ----------------
    for w in 0..4 {
        for d in 0..3 {
            out.extend_from_slice(&lot[w][d]);
        }
    }
    for w in 0..4 {
        for k in 0..7 {
            out.push(sd(lot[w][0][k], lot[w][1][k]));
        }
    }
    for d in 0..3 {
        for k in 0..7 {
            out.push(sd(lot[3][d][k], lot[1][d][k]));
        }
    }
    debug_assert_eq!(out.len(), EXPECTED_LEN - N_PART_MIN);

    // ---------------- 分钟矩阵：237 分钟 × 40 列，再降维 ----------------
    let mut min_mat = Array2::<f32>::from_elem((N_MINUTES, N_MIN_COLS), f32::NAN);
    for mi in 0..N_MINUTES {
        let (lo, hi) = win_range(trades, mi as i64 * 60, (mi as i64 + 1) * 60);
        if hi < lo + 3 {
            continue;
        }
        let dirs = agg.build(&trades[lo..hi]);
        for ch in 0..6 {
            let ords = &dirs[ch % 2];
            let t = [0usize, 0, 1, 1, 2, 2][ch];
            let seq: Vec<f32> = ords.iter().map(|o| o.target_val(t)).collect();
            let ex = excess_seq(&seq, CAP_MIN, N_SHUF_MIN, &mut ev);
            for (j, &k) in MIN_CORE_IDX.iter().enumerate() {
                min_mat[[mi, ch * 6 + j]] = ex[k];
            }
        }
        for d in 0..2 {
            let vols: Vec<f32> = dirs[d].iter().map(|o| o.vol).collect();
            let l = lot7(&vols, &mut keys);
            min_mat[[mi, 36 + d]] = l[6];
            min_mat[[mi, 38 + d]] = l[4];
        }
    }
    let (reduced, _) = features::get_features_factors_rust_full(&min_mat.view(), &min_col_names(), false);
    out.extend_from_slice(&reduced);

    if out.len() != EXPECTED_LEN {
        out.resize(EXPECTED_LEN, f32::NAN);
    }
    out
}

// ============================================================================
// 因子名（与输出顺序逐一对齐）
// ============================================================================

fn min_col_names() -> Vec<String> {
    let mut v = Vec::with_capacity(N_MIN_COLS);
    for ch in &MIN_CH6_NAMES {
        for &k in &MIN_CORE_IDX {
            v.push(format!("s2v2_min_{}_{}", ch, CORE_NAMES[k]));
        }
    }
    for n in &MIN_LOT4_NAMES {
        v.push(format!("s2v2_min_{}", n));
    }
    v
}

pub fn anneal_order_profile_v2_names() -> Vec<String> {
    let mut v = Vec::with_capacity(EXPECTED_LEN);
    let core = &CORE_NAMES[..N_ORD];
    for t in &TARGET5_NAMES {
        for w in &WIN4_NAMES {
            for d in &DIR3_NAMES {
                for m in core {
                    v.push(format!("s2v2_tgt_{}_{}_{}_{}", t, w, d, m));
                }
            }
        }
    }
    for (_, _, p) in &TARGET_PAIRS {
        for w in &WIN4_NAMES {
            for d in &DIR3_NAMES {
                for m in core {
                    v.push(format!("s2v2_tgtdiff_{}_{}_{}_{}", p, w, d, m));
                }
            }
        }
    }
    for a in &AXIS3_NAMES {
        for w in &WIN4_NAMES {
            for d in &DIR3_NAMES {
                for m in core {
                    v.push(format!("s2v2_axis_{}_{}_{}_{}", a, w, d, m));
                }
            }
        }
    }
    for (_, _, p) in &AXIS_PAIRS {
        for w in &WIN4_NAMES {
            for d in &DIR3_NAMES {
                for m in core {
                    v.push(format!("s2v2_axisdiff_{}_{}_{}_{}", p, w, d, m));
                }
            }
        }
    }
    for p in &PRIOR3_NAMES {
        for w in &WIN4_NAMES {
            for d in &DIR3_NAMES {
                for m in core {
                    v.push(format!("s2v2_prior_{}_{}_{}_{}", p, w, d, m));
                }
            }
        }
    }
    for (_, _, p) in &PRIOR_PAIRS {
        for w in &WIN4_NAMES {
            for d in &DIR3_NAMES {
                for m in core {
                    v.push(format!("s2v2_priordiff_{}_{}_{}_{}", p, w, d, m));
                }
            }
        }
    }
    for w in &WIN4_NAMES {
        for d in &DIR3_NAMES {
            for m in &DEEP8_NAMES {
                v.push(format!("s2v2_deep_{}_{}_{}", w, d, m));
            }
        }
    }
    for w in &WIN4_NAMES {
        for m in &DEEP8_NAMES {
            v.push(format!("s2v2_deepdiff_{}_bid_minus_ask_{}", w, m));
        }
    }
    for s in &GAME_SLICE12_NAMES {
        for w in &WIN4_NAMES {
            for m in core {
                v.push(format!("s2v2_game_{}_{}_{}", s, w, m));
            }
        }
    }
    for (_, _, p) in &GAME_PAIRS {
        for w in &WIN4_NAMES {
            for m in core {
                v.push(format!("s2v2_gamediff_{}_{}_{}", p, w, m));
            }
        }
    }
    for s in &GAMETGT_SLICE12_NAMES {
        for &t in &GAMETGT_TARGETS {
            for m in core {
                v.push(format!("s2v2_gametgt_{}_{}_{}", s, TARGET5_NAMES[t], m));
            }
        }
    }
    for w in &WIN4_NAMES {
        for d in &DIR3_NAMES {
            for m in &LOT7_NAMES {
                v.push(format!("s2v2_lot_{}_{}_{}", w, d, m));
            }
        }
    }
    for w in &WIN4_NAMES {
        for m in &LOT7_NAMES {
            v.push(format!("s2v2_lotdiff_{}_bid_minus_ask_{}", w, m));
        }
    }
    for d in &DIR3_NAMES {
        for m in &LOT7_NAMES {
            v.push(format!("s2v2_lotdiff_late30_minus_early30_{}_{}", d, m));
        }
    }
    let dummy = Array2::<f32>::ones((2, N_MIN_COLS));
    let (_, red) = features::get_features_factors_rust_full(&dummy.view(), &min_col_names(), false);
    v.extend(red);
    assert_eq!(v.len(), EXPECTED_LEN, "anneal_order_profile_v2 因子名数量不符");
    v
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

fn split_and_compute(mut all: Vec<TradeRecord>) -> Vec<f32> {
    ensure_time_sorted(&mut all);
    let trades: Vec<TradeRecord> = all.iter().copied().filter(|t| t.flag != 32).collect();
    compute_anneal_order_profile_v2_from_records(&trades, &all)
}

pub fn compute_anneal_order_profile_v2_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let all = read_trade_fast_inner(code, date, true, true, usize::MAX)?;
    Ok(split_and_compute(all))
}

/// 柜员游戏扩展系列2 修正版（单股单日 7873 列）
#[pyfunction]
pub fn py_anneal_order_profile_v2(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_order_profile_v2_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_anneal_order_profile_v2_names() -> Vec<String> {
    anneal_order_profile_v2_names()
}

#[pyfunction]
pub fn py_anneal_order_profile_v2_from_data(
    exchtime_us: PyReadonlyArray1<i64>,
    price: PyReadonlyArray1<f32>,
    volume: PyReadonlyArray1<f32>,
    turnover: PyReadonlyArray1<f32>,
    flag: PyReadonlyArray1<i32>,
    bid_order: PyReadonlyArray1<i64>,
    ask_order: PyReadonlyArray1<i64>,
) -> PyResult<Vec<f32>> {
    let (t, p, v, tn, f, bo, ao) = (
        exchtime_us.as_array(),
        price.as_array(),
        volume.as_array(),
        turnover.as_array(),
        flag.as_array(),
        bid_order.as_array(),
        ask_order.as_array(),
    );
    let n = t.len();
    if [p.len(), v.len(), tn.len(), f.len(), bo.len(), ao.len()].iter().any(|&l| l != n) {
        return Err(pyo3::exceptions::PyValueError::new_err("所有输入数组长度必须一致"));
    }
    let all: Vec<TradeRecord> = (0..n)
        .filter_map(|i| build_trade_record(t[i], p[i], v[i], tn[i], f[i], bo[i], ao[i], true))
        .collect();
    Ok(split_and_compute(all))
}
