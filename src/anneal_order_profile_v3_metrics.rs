//! 柜员游戏扩展系列2 第三版：v1 与 v2 合并 + 时间分段扩展（anneal_order_profile_v3）。
//!
//! 背景（两版回测对比的结论）：
//!   - v1（anneal_order_profile）的强度大部分来自「当天交易多活跃、订单多少、单笔多大」，
//!     被退火步数上限间接带进还原度指标；v2 减去打乱对照后把这部分信息一起减掉了，入选数和 IC 都下降。
//!   - v2 新加的成交量结构（整手 / 同量订单占比）有效，是两版里去掉市值与成交额后残留最强的。
//! 本版一次性合并：
//!   1. 每条序列同时输出 raw（真实顺序退火的 16 个指标，即 v1 口径，含 posD_ratio / sign_flip_prob）
//!      与 ex（相对固定种子打乱对照的 14 个超额，即 v2 口径）。raw 是同一次真实退火的结果，不增加计算。
//!   2. 时间全部用微秒精度 time_us（修 v1 的 128 秒精度问题）。
//!   3. 时段从 4 个扩到 5 个：全天 / 开盘 30 分 / 盘中 3 小时 / 下午 2 小时 / 尾盘 30 分。
//!      v2 只算全天的「博弈子集 × 其他画像」扩到全天 / 下午 / 尾盘三段；
//!      分钟矩阵分别在全天 237 分钟、下午 117 分钟、尾盘 30 分钟上各降维一次。
//!   4. 显式成交活跃度（flow）：订单数、平均 / 中位单笔金额、前 10% 大单金额占比、平均成交笔数，
//!      把 v1 退火指标间接携带的信息直接写成列；分钟矩阵也加入每分钟买卖单数与平均单笔金额。
//!   5. 分钟矩阵恢复 v1 的离场序通道和 raw 指标，同时保留 v2 的超额与量结构通道。
//!   6. 保留 v2 的全部修正（统一步数、共用对照、平均秩、无比值列、严格价区、撤单方向判定）。
//! 时段级 ex、lot 在 v2 已有的 4 个时段上与 v2 逐位相同（同一算法、同一种子），可用于核对。
//!
//! 固定输出 30045 列（见 EXPECTED_LEN），确定性：所有随机数都来自固定种子，HashMap 遍历结果先排序再用。

use crate::anneal_regime_diff_metrics::{adaptive_steps, FastAnnealBuf, CORE_NAMES, N_CORE};
use crate::fast_csv_reader::{build_trade_record, read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::{s, Array2};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// ============================================================================
// 常量
// ============================================================================

const US: i64 = 1_000_000;
const N_MINUTES: usize = 237;

/// 5 个时段（相对 09:30 的秒数，下午已前移 90 分钟：7200 = 13:00，14220 = 14:57）
pub const WIN_NAMES: [&str; 5] = ["fullday", "early30", "mid3h", "pm2h", "late30"];
const WIN_BOUNDS_SEC: [(i64, i64); 5] =
    [(0, 14_220), (0, 1_800), (1_800, 12_600), (7_200, 14_220), (12_600, 14_220)];
const NW: usize = 5;

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

/// 与顺序有关的 14 个核心指标 = CORE_NAMES[0..14]（ex 口径只输出这 14 个）
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
    "eq100_share",
    "odd_lot_share",
    "round1000_share",
    "round10000_share",
    "mode_share",
    "distinct_ratio",
    "tie_prob",
];
/// 成交活跃度指标
pub const FLOW5_NAMES: [&str; 5] =
    ["log_n_orders", "log_mean_amt", "log_median_amt", "top10_amt_share", "mean_fills"];
/// lot + flow 的时段对比
const LF_CONTRASTS: [(usize, usize, &str); 2] =
    [(4, 1, "late30_minus_early30"), (3, 0, "pm2h_minus_fullday")];

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
/// gametgt 计算的时段（下标指 WIN_NAMES）：全天、下午 2 小时、尾盘 30 分
const GT_WINS: [usize; 3] = [0, 3, 4];

// ---------------- 分钟矩阵 ----------------
/// raw 通道：买 / 卖 × 成交量、拆单笔数、存活时长、离场序成交量
pub const MIN_RAW_CH: [&str; 8] = [
    "bid_vol", "ask_vol", "bid_subcnt", "ask_subcnt", "bid_dur", "ask_dur", "bid_lastord", "ask_lastord",
];
/// raw 通道保留的指标：final_r, half_life_norm, inertia_norm, decline_ratio, max_drawdown, dr_std, mid_deter_ratio
const MIN_RAW_IDX: [usize; 7] = [1, 4, 7, 8, 9, 10, 11];
/// ex 通道（下标指 MIN_RAW_CH）：买卖成交量、买卖存活时长
const MIN_EX_CH: [usize; 4] = [0, 1, 4, 5];
/// ex 通道保留的指标：r0_init, final_r, inertia_norm
const MIN_EX_IDX: [usize; 3] = [0, 1, 7];
pub const MIN_LOT_NAMES: [&str; 6] = [
    "bid_tie_prob", "ask_tie_prob", "bid_mode_share", "ask_mode_share", "bid_eq100_share", "ask_eq100_share",
];
pub const MIN_FLOW_NAMES: [&str; 4] =
    ["bid_log_n_orders", "ask_log_n_orders", "bid_log_mean_amt", "ask_log_mean_amt"];
const N_MIN_COLS: usize = 8 * 7 + 4 * 3 + 6 + 4; // 78
/// 分钟矩阵降维的行区间：全天 237 分钟、下午 117 分钟、尾盘 30 分钟
const MIN_RANGES: [(usize, usize, &str); 3] = [(0, 237, "mfull"), (120, 237, "mpm"), (207, 237, "mlate")];

/// 退火步数上限：时段级 2 万步，分钟级 3000 步（与 v2 相同）
const CAP_WIN: usize = 20_000;
const CAP_MIN: usize = 3_000;
/// 打乱对照次数：时段级 2 次，分钟级 1 次（与 v2 相同）
const N_SHUF_WIN: usize = 2;
const N_SHUF_MIN: usize = 1;
const SHUF_SEED: u64 = 0xA5A5_1234_5678_9ABC;

const RE: usize = N_CORE + N_ORD; // 每条序列输出 raw16 + ex14 = 30
pub const N_PART_TGT: usize = 5 * NW * 3 * RE + 6 * NW * 3 * RE; // 2250 + 2700
pub const N_PART_AXIS: usize = 3 * NW * 3 * RE + 6 * NW * 3 * N_CORE; // 1350 + 1440
pub const N_PART_PRIOR: usize = 3 * NW * 3 * RE + 3 * NW * 3 * N_CORE; // 1350 + 720
pub const N_PART_DEEP: usize = NW * 3 * 16 + NW * 16; // 240 + 80
pub const N_PART_GAME: usize = 12 * NW * RE + 8 * NW * N_CORE; // 1800 + 640
pub const N_PART_GAMETGT: usize = 12 * 3 * 3 * RE; // 3240
pub const N_PART_LF: usize = NW * 3 * 12 + NW * 12 + 2 * 3 * 12; // 180 + 60 + 72
pub const N_PART_MIN: usize = 3 * (21 * N_MIN_COLS + N_MIN_COLS * (N_MIN_COLS - 1) / 2); // 13923

/// 总列数 30045
pub const EXPECTED_LEN: usize = N_PART_TGT
    + N_PART_AXIS
    + N_PART_PRIOR
    + N_PART_DEEP
    + N_PART_GAME
    + N_PART_GAMETGT
    + N_PART_LF
    + N_PART_MIN;

type R16 = [f32; N_CORE];
type E14 = [f32; N_ORD];
const NAN16: R16 = [f32::NAN; N_CORE];
const NAN5: [f32; 5] = [f32::NAN; 5];

// ============================================================================
// 以下工具函数逐字复制自 anneal_order_profile_v2_metrics.rs（订单聚合、退火核心、打乱对照、量结构、博弈子集）
// ============================================================================

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

/// 复用的聚合容器：哈希表只存 (订单号 -> 输出数组下标 u32)，新订单按首次出现位置 i 严格递增直接追加进输出切片，
/// 既消除 80 字节结构体的哈希桶扫描（values()），又天然满足按 (first_idx, order_id) 升序而无需二次排序。
struct OrderAgg {
    bid: FxHashMap<i64, u32>,
    ask: FxHashMap<i64, u32>,
}

impl OrderAgg {
    fn new(cap: usize) -> Self {
        Self {
            bid: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
            ask: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
        }
    }

    /// 聚合一段成交为买卖两侧订单（复用缓冲，不构造混合队列，供 237 分钟循环直接调用）
    fn build_bid_ask(
        &mut self,
        trades: &[TradeRecord],
        b: &mut Vec<RichOrder>,
        a: &mut Vec<RichOrder>,
    ) {
        use std::collections::hash_map::Entry;
        self.bid.clear();
        self.ask.clear();
        b.clear();
        a.clear();
        for (i, t) in trades.iter().enumerate() {
            match self.bid.entry(t.bid_order) {
                Entry::Occupied(o) => {
                    b[*o.get() as usize].add(i, t, t.flag == 66);
                }
                Entry::Vacant(v) => {
                    let idx = b.len() as u32;
                    v.insert(idx);
                    let mut ro = RichOrder::new(t.bid_order, i, t);
                    ro.add(i, t, t.flag == 66);
                    b.push(ro);
                }
            }
            match self.ask.entry(t.ask_order) {
                Entry::Occupied(o) => {
                    a[*o.get() as usize].add(i, t, t.flag == 83);
                }
                Entry::Vacant(v) => {
                    let idx = a.len() as u32;
                    v.insert(idx);
                    let mut ro = RichOrder::new(t.ask_order, i, t);
                    ro.add(i, t, t.flag == 83);
                    a.push(ro);
                }
            }
        }
    }

    /// 聚合一段成交为订单，返回按首次成交序排好的 [买, 卖, 买卖混合]
    fn build(&mut self, trades: &[TradeRecord]) -> [Vec<RichOrder>; 3] {
        let mut b = Vec::with_capacity(self.bid.capacity().min(trades.len()));
        let mut a = Vec::with_capacity(self.ask.capacity().min(trades.len()));
        self.build_bid_ask(trades, &mut b, &mut a);
        // b 与 a 已各自按 (first_idx, order_id) 升序，混合序键为 (first_idx, side_tag, order_id)
        // 其中买单 side_tag=0 < 卖单 side_tag=1，故双指针线性归并结果与元组排序逐元素完全一致
        let mut mixed = Vec::with_capacity(b.len() + a.len());
        let (mut ib, mut ia) = (0usize, 0usize);
        while ib < b.len() && ia < a.len() {
            if b[ib].first_idx <= a[ia].first_idx {
                mixed.push(b[ib]);
                ib += 1;
            } else {
                mixed.push(a[ia]);
                ia += 1;
            }
        }
        if ib < b.len() {
            mixed.extend_from_slice(&b[ib..]);
        }
        if ia < a.len() {
            mixed.extend_from_slice(&a[ia..]);
        }
        [b, a, mixed]
    }
}

/// f32 的全序键：整数大小顺序与 f32::total_cmp 完全一致
#[inline]
fn ord_key(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// 按坐标轴重排成交量。v2 的写法是整块复制订单结构再稳定排序；这里只排小的键元组，
/// 键末位附上原位置（等价于稳定排序的并列规则），所以顺序与 v2 逐位相同。
fn vol_by_axis(orders: &[RichOrder], axis: usize) -> Vec<f32> {
    let n = orders.len();
    match axis {
        1 => {
            let mut k: Vec<(usize, usize, i64, u32)> =
                orders.iter().enumerate().map(|(p, o)| (o.last_idx, o.first_idx, o.order_id, p as u32)).collect();
            k.sort_unstable();
            k.iter().map(|x| orders[x.3 as usize].vol).collect()
        }
        2 => {
            let mut k: Vec<(i64, usize, u32)> =
                orders.iter().enumerate().map(|(p, o)| (o.order_id, o.first_idx, p as u32)).collect();
            k.sort_unstable();
            k.iter().map(|x| orders[x.2 as usize].vol).collect()
        }
        3 => {
            let mut k: Vec<(u32, usize, i64, u32)> = orders
                .iter()
                .enumerate()
                .map(|(p, o)| (ord_key(o.vwap()), o.first_idx, o.order_id, p as u32))
                .collect();
            k.sort_unstable();
            k.iter().map(|x| orders[x.3 as usize].vol).collect()
        }
        _ => {
            let mut v = Vec::with_capacity(n);
            v.extend(orders.iter().map(|o| o.vol));
            v
        }
    }
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

fn anneal_presorted<const NEED_D: bool>(
    seq: &[f32],
    pre_sorted: &[f32],
    m_max: usize,
    mode: u8,
    keep_guess: bool,
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

    // 构造交错存储的 (guess_k, seq_k) 数组，把每步 4 次随机访存压成 2 次同缓存行读取，同时单趟算初始平方和 s
    buf.gs.clear();
    let mut s = 0.0f64;
    if mode == 0 || mode > 3 {
        buf.gs.reserve(n);
        for k in 0..n {
            let g = pre_sorted[k];
            let t = seq[k];
            let d = g as f64 - t as f64;
            s += d * d;
            buf.gs.push((g, t));
        }
    } else if mode == 1 {
        buf.gs.reserve(n);
        for k in 0..n {
            let g = pre_sorted[n - 1 - k];
            let t = seq[k];
            let d = g as f64 - t as f64;
            s += d * d;
            buf.gs.push((g, t));
        }
    } else if mode == 2 {
        // U型先验：大值放两端，小值放中间
        buf.gs.resize(n, (0.0, 0.0));
        let mut left = 0usize;
        let mut right = n - 1;
        for (idx, &val) in pre_sorted.iter().rev().enumerate() {
            if idx % 2 == 0 {
                buf.gs[left] = (val, seq[left]);
                left += 1;
            } else {
                buf.gs[right] = (val, seq[right]);
                right = right.saturating_sub(1);
            }
        }
        for &(g, t) in buf.gs.iter() {
            let d = g as f64 - t as f64;
            s += d * d;
        }
    } else {
        // 倒U型先验：小值放两端，大值放中间
        buf.gs.resize(n, (0.0, 0.0));
        let mut left = 0usize;
        let mut right = n - 1;
        for (idx, &val) in pre_sorted.iter().enumerate() {
            if idx % 2 == 0 {
                buf.gs[left] = (val, seq[left]);
                left += 1;
            } else {
                buf.gs[right] = (val, seq[right]);
                right = right.saturating_sub(1);
            }
        }
        for &(g, t) in buf.gs.iter() {
            let d = g as f64 - t as f64;
            s += d * d;
        }
    }
    let denom = 2.0 * sigma2 * n as f64;
    let inv_denom = 1.0 / denom;
    let r0 = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
    let hl_target = (1.0 + r0) * 0.5;
    let s_tol = (sigma2 * n as f64 * 1e-10).max(1e-12);
    let inv_m = 1.0 / (m_max as f64 - 1.0).max(1.0);

    let mut rng = XorShift64::new(SEED ^ ((mode as u64) << 32));
    let mut prev_r = r0;
    let mut peak_r = r0;
    let mut max_dd = 0.0f64;

    let mut a1_step = m_max;
    let mut a2_step = m_max;
    let mut a3_step = m_max;

    let mut inertia_sum = 0.0f64;
    let mut declines = 0u32;
    let mut dr_s1 = 0.0f64;
    let mut dr_s2 = 0.0f64;

    let s1_end = m_max / 3;
    let s2_end = 2 * (m_max / 3);
    let mut s1_s1 = 0.0f64;
    let mut s1_s2 = 0.0f64;
    let mut s2_decl = 0u32;
    let mut s3_s1 = 0.0f64;
    let mut s3_s2 = 0.0f64;
    let mut r_at_s2 = r0;

    let mut d_cnt = 0u32;
    let mut d_pos = 0u32;
    let mut d_flips = 0u32;
    let mut prev_d_sign: i8 = 0;

    let mut actual_steps = m_max;
    let gs = buf.gs.as_mut_slice();
    assert_eq!(gs.len(), n);

    for t in 0..m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = if i + 1 == n { 0 } else { i + 1 };
        }

        // SAFETY: n >= 3 且 next_index(n) 满足 ((u32 as u64) * n) >> 32 < n，j 亦严格小于 n
        let (gi, ti, gj, tj) = unsafe {
            let pi = *gs.get_unchecked(i);
            let pj = *gs.get_unchecked(j);
            (pi.0, pi.1, pj.0, pj.1)
        };
        if NEED_D && d_cnt < 2000 && gi != gj {
            let dv = tj - ti;
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

        let ds = 2.0f32 * (gi - gj) * (ti - tj);
        let accepted = (ds as f64) < 0.0
            || (ds as f64) < sigma2 * (1.0 - t as f64 * inv_m).max(0.0);
        if accepted {
            s += ds as f64;
            unsafe {
                gs.get_unchecked_mut(i).0 = gj;
                gs.get_unchecked_mut(j).0 = gi;
            }
        }

        // 当 t > 0 且本步未改变平方和 s 时，cr 与 prev_r 逐位相同、dv == +0.0，
        // 只需累加 inertia_sum 与记录 s2_end 截面值，其余统计与收敛判定均不变
        if t > 0 && (!accepted || ds == 0.0) {
            inertia_sum += 1.0 - prev_r;
            if t + 1 == s2_end {
                r_at_s2 = prev_r;
            }
            continue;
        }

        let cr = (1.0 - s * inv_denom).clamp(-1.0, 1.0);

        if t > 0 {
            let dv = cr - prev_r;
            dr_s1 += dv;
            dr_s2 += dv * dv;
            if cr < prev_r {
                declines += 1;
                if t >= s1_end && t < s2_end {
                    s2_decl += 1;
                }
            }
            if t < s1_end {
                s1_s1 += dv;
                s1_s2 += dv * dv;
            } else if t >= s2_end {
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
            actual_steps = t + 1;
            if t + 1 < s2_end {
                r_at_s2 = cr;
            }
            break;
        }
    }

    if keep_guess {
        buf.guess.clear();
        buf.guess.extend(buf.gs.iter().map(|p| p.0));
    }

    let final_r = prev_r;
    let dr_cnt = actual_steps.saturating_sub(1) as u32;
    let s1_cnt = actual_steps.min(s1_end).saturating_sub(1) as u32;
    let s2_cnt = actual_steps.min(s2_end).saturating_sub(s1_end.max(1)) as u32;
    let s3_cnt = actual_steps.saturating_sub(s2_end.max(1)) as u32;

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
    grp: Vec<f32>,
    ra: Vec<f32>,
    rb: Vec<f32>,
}

impl Ev {
    fn new() -> Self {
        Self {
            buf: FastAnnealBuf::new(),
            shuf: Vec::with_capacity(4096),
            grp: Vec::with_capacity(4096),
            ra: Vec::with_capacity(4096),
            rb: Vec::with_capacity(4096),
        }
    }
}

/// 平均秩（并列取平均名次，名次 0..n-1）。vals 必须是 sorted 的一个排列（真实序列、打乱序列、
/// 退火后的猜测序列都满足）。并列组按 == 在排好序的数组上划分、组内取平均名次，与 v2 的
/// 「下标间接排序再分组」结果逐位相同，但只需一次线性扫描加二分查找。
fn build_rank_groups(sorted: &[f32], grp: &mut Vec<f32>) {
    let n = sorted.len();
    grp.clear();
    grp.resize(n, 0.0);
    let mut s = 0usize;
    while s < n {
        let mut e = s + 1;
        while e < n && sorted[e] == sorted[s] {
            e += 1;
        }
        let r = (s + e - 1) as f32 * 0.5;
        for g in &mut grp[s..e] {
            *g = r;
        }
        s = e;
    }
}

#[inline]
fn map_ranks_from_groups(vals: &[f32], sorted: &[f32], grp: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.extend(vals.iter().map(|&v| grp[sorted.partition_point(|&y| y < v)]));
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
fn deep8(seq: &[f32], guess: &[f32], sorted: &[f32], grp: &mut Vec<f32>, ra: &mut Vec<f32>, rb: &mut Vec<f32>) -> [f32; 8] {
    let n = seq.len();
    if n < 10 || guess.len() != n {
        return NAN8;
    }
    build_rank_groups(sorted, grp);
    map_ranks_from_groups(seq, sorted, grp, ra);
    map_ranks_from_groups(guess, sorted, grp, rb);
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
        let c = anneal_presorted::<false>(&ev.shuf, sorted, m, 0, want_deep, &mut ev.buf);
        if !c[1].is_finite() {
            return (NAN_ORD, NAN8);
        }
        for k in 0..N_ORD {
            acc[k] += c[k] as f64;
        }
        if want_deep {
            let d = deep8(&ev.shuf, &ev.buf.guess, sorted, &mut ev.grp, &mut ev.ra, &mut ev.rb);
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
    let c = anneal_presorted::<true>(seq, sorted, m, mode, want_deep, &mut ev.buf);
    let d = if want_deep && c[1].is_finite() {
        deep8(seq, &ev.buf.guess, sorted, &mut ev.grp, &mut ev.ra, &mut ev.rb)
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

fn sorted_copy(seq: &[f32]) -> Vec<f32> {
    let mut v = seq.to_vec();
    v.sort_unstable_by(|a, b| a.total_cmp(b));
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
    let idx = (v.len() * num / den).min(v.len() - 1);
    let (_, val, _) = v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    Some(*val)
}

/// 12 个博弈子集的订单列表（slice 10/11 为「同时有主动和被动成交」的订单，仅 gametgt 使用）
fn game_order_slices(dirs: &[Vec<RichOrder>; 3]) -> [Vec<RichOrder>; 12] {
    let (b, a) = (&dirs[0], &dirs[1]);
    let bp70 = quantile_sorted(b.iter().map(|o| o.vol).collect(), 7, 10).unwrap_or(f32::INFINITY);
    let ap70 = quantile_sorted(a.iter().map(|o| o.vol).collect(), 7, 10).unwrap_or(f32::INFINITY);
    let med = quantile_sorted(dirs[2].iter().map(|o| o.vwap()).collect(), 1, 2).unwrap_or(f32::NAN);
    let mut out: [Vec<RichOrder>; 12] = std::array::from_fn(|_| Vec::new());
    for &o in b {
        if o.n_act > 0 && o.sub_cnt >= 2 {
            out[0].push(o);
        }
        if o.n_pas > 0 && o.sub_cnt >= 2 {
            out[2].push(o);
        }
        if o.vol > bp70 && o.sub_cnt <= 2 {
            out[4].push(o);
        }
        let vw = o.vwap();
        if vw > med {
            out[6].push(o);
        }
        if vw < med {
            out[8].push(o);
        }
        if o.n_act > 0 && o.n_pas > 0 {
            out[10].push(o);
        }
    }
    for &o in a {
        if o.n_act > 0 && o.sub_cnt >= 2 {
            out[1].push(o);
        }
        if o.n_pas > 0 && o.sub_cnt >= 2 {
            out[3].push(o);
        }
        if o.vol > ap70 && o.sub_cnt <= 2 {
            out[5].push(o);
        }
        let vw = o.vwap();
        if vw > med {
            out[7].push(o);
        }
        if vw < med {
            out[9].push(o);
        }
        if o.n_act > 0 && o.n_pas > 0 {
            out[11].push(o);
        }
    }
    out
}

/// 稳定按 time_us 排序（输入已有序时不动）
fn ensure_time_sorted(v: &mut Vec<TradeRecord>) {
    if !v.windows(2).all(|w| w[0].time_us <= w[1].time_us) {
        v.sort_by_key(|t| t.time_us);
    }
}

// ============================================================================
// v3 新增的小工具
// ============================================================================

/// 一条序列同时出 raw16（真实顺序退火）与 ex14（相对打乱对照的超额）
fn raw_and_excess(seq: &[f32], cap: usize, n_shuf: usize, ev: &mut Ev) -> (R16, E14) {
    if seq.len() < 3 {
        return (NAN16, NAN_ORD);
    }
    let sorted = sorted_copy(seq);
    let (base, _) = shuffle_baseline(&sorted, cap, n_shuf, false, ev);
    let (c, _) = run_true(seq, &sorted, cap, 0, false, ev);
    (c, excess(&c, &base))
}

fn diff16(a: &R16, b: &R16) -> R16 {
    let mut o = NAN16;
    for k in 0..N_CORE {
        o[k] = sd(a[k], b[k]);
    }
    o
}

#[inline]
fn push_re(out: &mut Vec<f32>, r: &R16, e: &E14) {
    out.extend_from_slice(r);
    out.extend_from_slice(e);
}

/// 成交活跃度 5 项：订单数、平均 / 中位单笔金额（取对数）、前 10% 大单金额占比、平均成交笔数
fn flow5(ords: &[RichOrder], buf: &mut Vec<f32>) -> [f32; 5] {
    let n = ords.len();
    if n < 3 {
        return NAN5;
    }
    buf.clear();
    buf.extend(ords.iter().map(|o| o.turnover.max(0.0)));
    let sum: f64 = buf.iter().map(|&x| x as f64).sum();
    if sum <= 0.0 {
        return NAN5;
    }
    let fills: f64 = ords.iter().map(|o| o.sub_cnt as f64).sum();
    buf.sort_unstable_by(|a, b| a.total_cmp(b));
    let median = buf[n / 2] as f64;
    let k = ((n as f64 * 0.1).ceil() as usize).clamp(1, n);
    let top: f64 = buf[n - k..].iter().map(|&x| x as f64).sum();
    [
        (n as f64).ln() as f32,
        (sum / n as f64 + 1.0).ln() as f32,
        (median + 1.0).ln() as f32,
        (top / sum) as f32,
        (fills / n as f64) as f32,
    ]
}

/// lot7 + flow5 拼成 12 维
fn lf12(lot: &[f32; 7], flow: &[f32; 5]) -> [f32; 12] {
    let mut o = [f32::NAN; 12];
    o[..7].copy_from_slice(lot);
    o[7..].copy_from_slice(flow);
    o
}

// ============================================================================
// 主计算
// ============================================================================

pub fn compute_anneal_order_profile_v3_from_records(
    trades: &[TradeRecord],
    trades_all: &[TradeRecord],
) -> Vec<f32> {
    if trades.len() < 10 {
        return vec![f32::NAN; EXPECTED_LEN];
    }
    let mut out: Vec<f32> = Vec::with_capacity(EXPECTED_LEN);
    let mask = crate::funnel_mask::snapshot();
    let need_deep = mask.any_active(&["s2v3_deep", "s2v3_deepdiff"]);
    let need_lf = mask.any_active(&["s2v3_lf", "s2v3_lfdiff"]);
    let mut need_axis = [false; 4];
    for ax in 1..4 {
        let m_direct = format!("s2v3_axis_{}", AXIS3_NAMES[ax - 1]);
        need_axis[ax] = mask.any_active(&["s2v3_axis", "s2v3_axisdiff", &m_direct])
            || AXIS_PAIRS.iter().any(|&(a, b, p)| {
                (a == ax || b == ax) && mask.is_active(&format!("s2v3_axisdiff_{p}"))
            });
    }
    let mut need_prior = [false; 4];
    for mode in 1..4usize {
        let m_direct = format!("s2v3_prior_{}", PRIOR3_NAMES[mode - 1]);
        need_prior[mode] = mask.any_active(&["s2v3_prior", "s2v3_priordiff", &m_direct])
            || PRIOR_PAIRS.iter().any(|&(a, b, p)| {
                (a == mode || b == mode) && mask.is_active(&format!("s2v3_priordiff_{p}"))
            });
    }
    let mut need_game = [false; 12];
    for sl in 0..12 {
        let m_direct = format!("s2v3_game_{}", GAME_SLICE12_NAMES[sl]);
        need_game[sl] = mask.any_active(&["s2v3_game", "s2v3_gamediff", &m_direct])
            || GAME_PAIRS.iter().any(|&(a, b, p)| {
                (a == sl || b == sl) && mask.is_active(&format!("s2v3_gamediff_{p}"))
            });
    }
    let need_game_from_tgt0 = GAME_PAIRS.iter().any(|&(a, _, p)| {
        (a == 100 || a == 101)
            && (mask.is_active("s2v3_gamediff") || mask.is_active(&format!("s2v3_gamediff_{p}")))
    });
    let mut need_tgt = [false; 5];
    for t in 0..5 {
        let m_direct = format!("s2v3_tgt_{}", TARGET5_NAMES[t]);
        need_tgt[t] = mask.any_active(&["s2v3_tgt", "s2v3_tgtdiff", &m_direct])
            || TARGET_PAIRS.iter().any(|&(a, b, p)| {
                (a == t || b == t) && mask.is_active(&format!("s2v3_tgtdiff_{p}"))
            });
    }
    if need_deep
        || need_game_from_tgt0
        || need_axis[1..].iter().any(|&x| x)
        || need_prior[1..].iter().any(|&x| x)
        || AXIS_PAIRS.iter().any(|&(a, b, p)| {
            (a == 0 || b == 0) && mask.is_active(&format!("s2v3_axisdiff_{p}"))
        })
        || PRIOR_PAIRS.iter().any(|&(a, b, p)| {
            (a == 0 || b == 0) && mask.is_active(&format!("s2v3_priordiff_{p}"))
        })
    {
        need_tgt[0] = true;
    }
    let mut need_gametgt = [false; 12];
    for sl in 0..12 {
        let m_direct = format!("s2v3_gametgt_{}", GAMETGT_SLICE12_NAMES[sl]);
        need_gametgt[sl] = mask.any_active(&["s2v3_gametgt", &m_direct]);
    }
    let mut need_min_range = [false; 3];
    for (ri, &(_, _, pfx)) in MIN_RANGES.iter().enumerate() {
        let m0 = format!("s2v3_{pfx}");
        let m1 = format!("s2v3_{pfx}_bid");
        let m2 = format!("s2v3_{pfx}_ask");
        need_min_range[ri] =
            mask.any_active(&[&m0, &m1, &m2, "s2v3_min", "s2v3_min_bid", "s2v3_min_ask"]);
    }
    let need_any_min = need_min_range.iter().any(|&x| x);
    let mut ev = Ev::new();
    let mut agg = OrderAgg::new(trades.len() / 2 + 1);
    let mut keys: Vec<u64> = Vec::with_capacity(4096);
    let mut fbuf: Vec<f32> = Vec::with_capacity(4096);

    // time_us 已含 +8 小时与下午前移 90 分钟，按本地日历日取 09:30 为基准
    let day_us = 86_400 * US;
    let t_open = trades[0].time_us.div_euclid(day_us) * day_us + (9 * 3600 + 30 * 60) * US;
    let win_range = |v: &[TradeRecord], lo_s: i64, hi_s: i64| -> (usize, usize) {
        let lo = v.partition_point(|t| t.time_us < t_open + lo_s * US);
        let hi = v.partition_point(|t| t.time_us < t_open + hi_s * US);
        (lo, hi)
    };

    let mut tgt_r = [[[NAN16; 3]; NW]; 5];
    let mut tgt_e = [[[NAN_ORD; 3]; NW]; 5];
    let mut axis_r = [[[NAN16; 3]; NW]; 4]; // axis 0 = tgt_vol
    let mut axis_e = [[[NAN_ORD; 3]; NW]; 4];
    let mut prior_r = [[[NAN16; 3]; NW]; 4]; // mode 0 = tgt_vol
    let mut prior_e = [[[NAN_ORD; 3]; NW]; 4];
    let mut deep_r = [[NAN8; 3]; NW];
    let mut deep_e = [[NAN8; 3]; NW];
    let mut lot = [[NAN7; 3]; NW];
    let mut flow = [[NAN5; 3]; NW];
    let mut game_r = [[NAN16; NW]; 12];
    let mut game_e = [[NAN_ORD; NW]; 12];
    let mut gt_r = [[[NAN16; 3]; 3]; 12]; // [slice][gt_win][target]
    let mut gt_e = [[[NAN_ORD; 3]; 3]; 12];

    for w in 0..NW {
        let (s_lo, s_hi) = WIN_BOUNDS_SEC[w];
        let (lo, hi) = win_range(trades, s_lo, s_hi);
        let (rlo, rhi) = win_range(trades_all, s_lo, s_hi);

        // 撤单序列（一侧订单号为 0 的撤单记录，按时间顺序）
        if need_game[10] || need_game[11] {
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
            if need_game[10] {
                (game_r[10][w], game_e[10][w]) = raw_and_excess(&ret_b, CAP_WIN, N_SHUF_WIN, &mut ev);
            }
            if need_game[11] {
                (game_r[11][w], game_e[11][w]) = raw_and_excess(&ret_a, CAP_WIN, N_SHUF_WIN, &mut ev);
            }
        }

        if hi < lo + 3 {
            continue;
        }
        let need_any_win_ord = need_lf
            || need_tgt.iter().any(|&x| x)
            || need_game[..10].iter().any(|&x| x)
            || (GT_WINS.contains(&w) && need_gametgt.iter().any(|&x| x));
        if !need_any_win_ord {
            continue;
        }
        let dirs = agg.build(&trades[lo..hi]);
        for d in 0..3 {
            let ords = &dirs[d];
            if need_lf {
                let vol_seq: Vec<f32> = ords.iter().map(|o| o.vol).collect();
                lot[w][d] = lot7(&vol_seq, &mut keys);
                flow[w][d] = flow5(ords, &mut fbuf);
            }
            for t in 0..5 {
                if !need_tgt[t] {
                    continue;
                }
                let seq: Vec<f32> = ords.iter().map(|o| o.target_val(t)).collect();
                let want_deep = t == 0 && need_deep;
                let sorted = sorted_copy(&seq);
                let (base, based) = shuffle_baseline(&sorted, CAP_WIN, N_SHUF_WIN, want_deep, &mut ev);
                let (c, dd) = run_true(&seq, &sorted, CAP_WIN, 0, want_deep, &mut ev);
                tgt_r[t][w][d] = c;
                tgt_e[t][w][d] = excess(&c, &base);
                if t == 0 {
                    if need_deep {
                        deep_r[w][d] = dd;
                        for k in 0..8 {
                            deep_e[w][d][k] = sd(dd[k], based[k]);
                        }
                    }
                    axis_r[0][w][d] = c;
                    axis_e[0][w][d] = tgt_e[0][w][d];
                    prior_r[0][w][d] = c;
                    prior_e[0][w][d] = tgt_e[0][w][d];
                    // 新坐标轴：同一批成交量换顺序，共用 vol 的对照
                    for ax in 1..4 {
                        if !need_axis[ax] {
                            continue;
                        }
                        let s2 = vol_by_axis(ords, ax);
                        let (c2, _) = run_true(&s2, &sorted, CAP_WIN, 0, false, &mut ev);
                        axis_r[ax][w][d] = c2;
                        axis_e[ax][w][d] = excess(&c2, &base);
                    }
                    // 新先验：同一步数上限，共用 vol 的对照
                    for mode in 1..4u8 {
                        if !need_prior[mode as usize] {
                            continue;
                        }
                        let (c3, _) = run_true(&seq, &sorted, CAP_WIN, mode, false, &mut ev);
                        prior_r[mode as usize][w][d] = c3;
                        prior_e[mode as usize][w][d] = excess(&c3, &base);
                    }
                }
            }
        }
        // 博弈子集（成交量序列）
        let need_slices = need_game[..10].iter().any(|&x| x)
            || (GT_WINS.contains(&w) && need_gametgt.iter().any(|&x| x));
        if need_slices {
            let slices = game_order_slices(&dirs);
            for sl in 0..10 {
                if !need_game[sl] {
                    continue;
                }
                let seq: Vec<f32> = slices[sl].iter().map(|o| o.vol).collect();
                (game_r[sl][w], game_e[sl][w]) = raw_and_excess(&seq, CAP_WIN, N_SHUF_WIN, &mut ev);
            }
            // 博弈子集 × 其他画像：全天、下午、尾盘
            if let Some(gi) = GT_WINS.iter().position(|&x| x == w) {
                for sl in 0..12 {
                    if !need_gametgt[sl] {
                        continue;
                    }
                    for (ti, &t) in GAMETGT_TARGETS.iter().enumerate() {
                        let seq: Vec<f32> = slices[sl].iter().map(|o| o.target_val(t)).collect();
                        (gt_r[sl][gi][ti], gt_e[sl][gi][ti]) = raw_and_excess(&seq, CAP_WIN, N_SHUF_WIN, &mut ev);
                    }
                }
            }
        }
    }

    // ---------------- 输出：画像 ----------------
    for t in 0..5 {
        for w in 0..NW {
            for d in 0..3 {
                push_re(&mut out, &tgt_r[t][w][d], &tgt_e[t][w][d]);
            }
        }
    }
    for &(a, b, _) in &TARGET_PAIRS {
        for w in 0..NW {
            for d in 0..3 {
                push_re(&mut out, &diff16(&tgt_r[a][w][d], &tgt_r[b][w][d]), &diff_ord(&tgt_e[a][w][d], &tgt_e[b][w][d]));
            }
        }
    }
    // ---------------- 输出：坐标轴（对照相同，ex 之差等于 raw 之差，差值只输出 raw） ----------------
    for ax in 1..4 {
        for w in 0..NW {
            for d in 0..3 {
                push_re(&mut out, &axis_r[ax][w][d], &axis_e[ax][w][d]);
            }
        }
    }
    for &(a, b, _) in &AXIS_PAIRS {
        for w in 0..NW {
            for d in 0..3 {
                out.extend_from_slice(&diff16(&axis_r[a][w][d], &axis_r[b][w][d]));
            }
        }
    }
    // ---------------- 输出：先验 ----------------
    for mode in 1..4 {
        for w in 0..NW {
            for d in 0..3 {
                push_re(&mut out, &prior_r[mode][w][d], &prior_e[mode][w][d]);
            }
        }
    }
    for &(a, b, _) in &PRIOR_PAIRS {
        for w in 0..NW {
            for d in 0..3 {
                out.extend_from_slice(&diff16(&prior_r[a][w][d], &prior_r[b][w][d]));
            }
        }
    }
    // ---------------- 输出：深度对齐 ----------------
    for w in 0..NW {
        for d in 0..3 {
            out.extend_from_slice(&deep_r[w][d]);
            out.extend_from_slice(&deep_e[w][d]);
        }
    }
    for w in 0..NW {
        for k in 0..8 {
            out.push(sd(deep_r[w][0][k], deep_r[w][1][k]));
        }
        for k in 0..8 {
            out.push(sd(deep_e[w][0][k], deep_e[w][1][k]));
        }
    }
    // ---------------- 输出：博弈 ----------------
    for sl in 0..12 {
        for w in 0..NW {
            push_re(&mut out, &game_r[sl][w], &game_e[sl][w]);
        }
    }
    for &(a, b, _) in &GAME_PAIRS {
        for w in 0..NW {
            let va = match a {
                100 => tgt_r[0][w][0],
                101 => tgt_r[0][w][1],
                _ => game_r[a][w],
            };
            out.extend_from_slice(&diff16(&va, &game_r[b][w]));
        }
    }
    for sl in 0..12 {
        for gi in 0..3 {
            for ti in 0..3 {
                push_re(&mut out, &gt_r[sl][gi][ti], &gt_e[sl][gi][ti]);
            }
        }
    }
    // ---------------- 输出：成交量结构 + 成交活跃度 ----------------
    let lf: Vec<[[f32; 12]; 3]> =
        (0..NW).map(|w| [lf12(&lot[w][0], &flow[w][0]), lf12(&lot[w][1], &flow[w][1]), lf12(&lot[w][2], &flow[w][2])]).collect();
    for w in 0..NW {
        for d in 0..3 {
            out.extend_from_slice(&lf[w][d]);
        }
    }
    for w in 0..NW {
        for k in 0..12 {
            out.push(sd(lf[w][0][k], lf[w][1][k]));
        }
    }
    for &(a, b, _) in &LF_CONTRASTS {
        for d in 0..3 {
            for k in 0..12 {
                out.push(sd(lf[a][d][k], lf[b][d][k]));
            }
        }
    }
    debug_assert_eq!(out.len(), EXPECTED_LEN - N_PART_MIN);

    // ---------------- 分钟矩阵：237 分钟 × 78 列，再分三段降维 ----------------
    if need_any_min {
        let mut min_mat = Array2::<f32>::from_elem((N_MINUTES, N_MIN_COLS), f32::NAN);
        let mut min_agg = OrderAgg::new(512);
        let mut min_dirs: [Vec<RichOrder>; 2] = [Vec::with_capacity(512), Vec::with_capacity(512)];
        let mut min_vol_sorted: [Vec<f32>; 2] = [Vec::with_capacity(512), Vec::with_capacity(512)];
        let mut min_seq: Vec<f32> = Vec::with_capacity(512);
        let mut min_sorted: Vec<f32> = Vec::with_capacity(512);
        for mi in 0..N_MINUTES {
            let (lo, hi) = win_range(trades, mi as i64 * 60, (mi as i64 + 1) * 60);
            if hi < lo + 3 {
                continue;
            }
            let (b_ords, a_ords) = min_dirs.split_at_mut(1);
            min_agg.build_bid_ask(&trades[lo..hi], &mut b_ords[0], &mut a_ords[0]);
            for d in 0..2 {
                min_vol_sorted[d].clear();
                min_vol_sorted[d].extend(min_dirs[d].iter().map(|o| o.vol));
                min_vol_sorted[d].sort_unstable_by(|a, b| a.total_cmp(b));
            }
            for ch in 0..8 {
                let side = ch % 2;
                let ords = &min_dirs[side];
                if ords.len() < 3 {
                    continue;
                }
                match ch / 2 {
                    0 => {
                        min_seq.clear();
                        min_seq.extend(ords.iter().map(|o| o.vol));
                    }
                    1 => {
                        min_seq.clear();
                        min_seq.extend(ords.iter().map(|o| o.target_val(1)));
                    }
                    2 => {
                        min_seq.clear();
                        min_seq.extend(ords.iter().map(|o| o.target_val(2)));
                    }
                    _ => {
                        min_seq = vol_by_axis(ords, 1);
                    }
                }
                // ch=0/1 (vol) 与 ch=6/7 (lastord vol) 的排序数组完全相同，直接复用 min_vol_sorted
                let sorted_ref: &[f32] = if ch / 2 == 0 || ch / 2 == 3 {
                    &min_vol_sorted[side]
                } else {
                    min_sorted.clear();
                    min_sorted.extend_from_slice(&min_seq);
                    min_sorted.sort_unstable_by(|a, b| a.total_cmp(b));
                    &min_sorted
                };
                let ex_pos = MIN_EX_CH.iter().position(|&x| x == ch);
                let base = match ex_pos {
                    Some(_) => shuffle_baseline(sorted_ref, CAP_MIN, N_SHUF_MIN, false, &mut ev).0,
                    None => NAN_ORD,
                };
                let (c, _) = run_true(&min_seq, sorted_ref, CAP_MIN, 0, false, &mut ev);
                for (j, &k) in MIN_RAW_IDX.iter().enumerate() {
                    min_mat[[mi, ch * 7 + j]] = c[k];
                }
                if let Some(p) = ex_pos {
                    let e = excess(&c, &base);
                    for (j, &k) in MIN_EX_IDX.iter().enumerate() {
                        min_mat[[mi, 56 + p * 3 + j]] = e[k];
                    }
                }
            }
            for d in 0..2 {
                let l = lot7(&min_vol_sorted[d], &mut keys);
                min_mat[[mi, 68 + d]] = l[6];
                min_mat[[mi, 70 + d]] = l[4];
                min_mat[[mi, 72 + d]] = l[0];
                let f = flow5(&min_dirs[d], &mut fbuf);
                min_mat[[mi, 74 + d]] = f[0];
                min_mat[[mi, 76 + d]] = f[1];
            }
        }
        const N_SINGLE_MIN_RANGE: usize = N_PART_MIN / 3;
        for (ri, &(r0, r1, _)) in MIN_RANGES.iter().enumerate() {
            if need_min_range[ri] {
                let view = min_mat.slice(s![r0..r1, ..]);
                let reduced = features::get_features_factors_rust_values_only(&view, false);
                out.extend_from_slice(&reduced);
            } else {
                out.resize(out.len() + N_SINGLE_MIN_RANGE, f32::NAN);
            }
        }
    } else {
        out.resize(out.len() + N_PART_MIN, f32::NAN);
    }

    if out.len() != EXPECTED_LEN {
        out.resize(EXPECTED_LEN, f32::NAN);
    }
    out
}

// ============================================================================
// 因子名（与输出顺序逐一对齐）
// ============================================================================

fn min_col_names(pfx: &str) -> Vec<String> {
    let mut v = Vec::with_capacity(N_MIN_COLS);
    for ch in &MIN_RAW_CH {
        for &k in &MIN_RAW_IDX {
            v.push(format!("s2v3_{}_{}_raw_{}", pfx, ch, CORE_NAMES[k]));
        }
    }
    for &ch in &MIN_EX_CH {
        for &k in &MIN_EX_IDX {
            v.push(format!("s2v3_{}_{}_ex_{}", pfx, MIN_RAW_CH[ch], CORE_NAMES[k]));
        }
    }
    for n in MIN_LOT_NAMES.iter().chain(MIN_FLOW_NAMES.iter()) {
        v.push(format!("s2v3_{}_{}", pfx, n));
    }
    v
}

fn push_re_names(v: &mut Vec<String>, stem: &str) {
    for m in &CORE_NAMES {
        v.push(format!("{}_raw_{}", stem, m));
    }
    for m in &CORE_NAMES[..N_ORD] {
        v.push(format!("{}_ex_{}", stem, m));
    }
}

fn push_raw_names(v: &mut Vec<String>, stem: &str) {
    for m in &CORE_NAMES {
        v.push(format!("{}_raw_{}", stem, m));
    }
}

pub fn anneal_order_profile_v3_names() -> Vec<String> {
    let mut v = Vec::with_capacity(EXPECTED_LEN);
    for t in &TARGET5_NAMES {
        for w in &WIN_NAMES {
            for d in &DIR3_NAMES {
                push_re_names(&mut v, &format!("s2v3_tgt_{}_{}_{}", t, w, d));
            }
        }
    }
    for (_, _, p) in &TARGET_PAIRS {
        for w in &WIN_NAMES {
            for d in &DIR3_NAMES {
                push_re_names(&mut v, &format!("s2v3_tgtdiff_{}_{}_{}", p, w, d));
            }
        }
    }
    for a in &AXIS3_NAMES {
        for w in &WIN_NAMES {
            for d in &DIR3_NAMES {
                push_re_names(&mut v, &format!("s2v3_axis_{}_{}_{}", a, w, d));
            }
        }
    }
    for (_, _, p) in &AXIS_PAIRS {
        for w in &WIN_NAMES {
            for d in &DIR3_NAMES {
                push_raw_names(&mut v, &format!("s2v3_axisdiff_{}_{}_{}", p, w, d));
            }
        }
    }
    for p in &PRIOR3_NAMES {
        for w in &WIN_NAMES {
            for d in &DIR3_NAMES {
                push_re_names(&mut v, &format!("s2v3_prior_{}_{}_{}", p, w, d));
            }
        }
    }
    for (_, _, p) in &PRIOR_PAIRS {
        for w in &WIN_NAMES {
            for d in &DIR3_NAMES {
                push_raw_names(&mut v, &format!("s2v3_priordiff_{}_{}_{}", p, w, d));
            }
        }
    }
    for w in &WIN_NAMES {
        for d in &DIR3_NAMES {
            for kind in ["raw", "ex"] {
                for m in &DEEP8_NAMES {
                    v.push(format!("s2v3_deep_{}_{}_{}_{}", w, d, kind, m));
                }
            }
        }
    }
    for w in &WIN_NAMES {
        for kind in ["raw", "ex"] {
            for m in &DEEP8_NAMES {
                v.push(format!("s2v3_deepdiff_{}_bid_minus_ask_{}_{}", w, kind, m));
            }
        }
    }
    for s in &GAME_SLICE12_NAMES {
        for w in &WIN_NAMES {
            push_re_names(&mut v, &format!("s2v3_game_{}_{}", s, w));
        }
    }
    for (_, _, p) in &GAME_PAIRS {
        for w in &WIN_NAMES {
            push_raw_names(&mut v, &format!("s2v3_gamediff_{}_{}", p, w));
        }
    }
    for s in &GAMETGT_SLICE12_NAMES {
        for &gw in &GT_WINS {
            for &t in &GAMETGT_TARGETS {
                push_re_names(&mut v, &format!("s2v3_gametgt_{}_{}_{}", s, WIN_NAMES[gw], TARGET5_NAMES[t]));
            }
        }
    }
    let lf_names: Vec<&str> = LOT7_NAMES.iter().chain(FLOW5_NAMES.iter()).copied().collect();
    for w in &WIN_NAMES {
        for d in &DIR3_NAMES {
            for m in &lf_names {
                v.push(format!("s2v3_lf_{}_{}_{}", w, d, m));
            }
        }
    }
    for w in &WIN_NAMES {
        for m in &lf_names {
            v.push(format!("s2v3_lfdiff_{}_bid_minus_ask_{}", w, m));
        }
    }
    for (_, _, p) in &LF_CONTRASTS {
        for d in &DIR3_NAMES {
            for m in &lf_names {
                v.push(format!("s2v3_lfdiff_{}_{}_{}", p, d, m));
            }
        }
    }
    let dummy = Array2::<f32>::ones((2, N_MIN_COLS));
    for &(_, _, pfx) in &MIN_RANGES {
        let (_, red) = features::get_features_factors_rust_full(&dummy.view(), &min_col_names(pfx), false);
        v.extend(red);
    }
    assert_eq!(v.len(), EXPECTED_LEN, "anneal_order_profile_v3 因子名数量不符");
    v
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

fn split_and_compute(mut all: Vec<TradeRecord>) -> Vec<f32> {
    ensure_time_sorted(&mut all);
    let trades: Vec<TradeRecord> = all.iter().copied().filter(|t| t.flag != 32).collect();
    compute_anneal_order_profile_v3_from_records(&trades, &all)
}

pub fn compute_anneal_order_profile_v3_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let all = read_trade_fast_inner(code, date, true, true, usize::MAX)?;
    Ok(split_and_compute(all))
}

/// 柜员游戏扩展系列2 第三版（单股单日 30045 列）
#[pyfunction]
pub fn py_anneal_order_profile_v3(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_order_profile_v3_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_anneal_order_profile_v3_names() -> Vec<String> {
    anneal_order_profile_v3_names()
}

#[pyfunction]
pub fn py_anneal_order_profile_v3_from_data(
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
