//! 柜员游戏拓展1 v3：原始可还原性 + 打乱对照超额 + 订单量结构（anneal_regime_raw_excess）。
//!
//! 由旧版 anneal_regime_diff（v1）与新版 anneal_regime_excess（v2）合并而来，取舍依据是两版
//! 全部派生因子的中性化 IC（2017-01~2026-05，gap5）：
//!   - v1 的原始退火指标（半衰期、到达 0.8 步数、未还原面积、最大回撤等）整体最强，
//!     其中一部分强度来自「还原度随订单数变化」，v3 原样保留 v1 的步数规则与公式；
//!   - v1 两个探测指标（posD_ratio / sign_flip_prob）实际反映订单量雷同程度，同样有效，保留；
//!   - v2 的 tie_ratio（订单量相同比例）是两版最强的单个指标，later_gt_ratio 次之，保留；
//!   - v2 的打乱对照超额整体偏弱，只保留较好的 4 个（最终相关、净增益、未还原面积、Δr 波动），
//!     并改用与原始指标相同的步数，和原始指标同一套随机交换序列，作为「去掉订单数影响后」的对照；
//!   - v1 的分钟矩阵两两相关整体强、v2 的（全是超额指标之间）弱：两两相关只对 16 个原始指标做；
//!   - v2 的修正全部保留：微秒精确时间、订单按首次成交时刻归属、30 秒分辨率、
//!     全天 / 下午 / 尾盘 30 分钟三段、事件锚点只在 09:45~14:42 之间找、尺度末段补全、去掉比值列。
//!
//! 每段订单 22 个指标（MET_NAMES），模块与维度：
//!   M1 12 个日内时段 × 7 方向 × 22                                  =  1,848
//!   M2 15 组时段对比（只做差）× 7 方向 × 22                          =  2,310
//!   M3a 时间格矩阵(7 通道 × 22 = 154 列)单列 21 统计 × 6 份(3 段 × 1分钟/30秒) = 19,404
//!   M3b 全天 1 分钟矩阵中 16 个原始指标 × 7 通道(112 列)两两相关      =  6,216
//!   M4 10 种行情状态 × 5 通道 × 22 ×（高低组差 + 状态相关，急拉急跌只做差）× 6 份 = 12,540
//!   M5 放缩量涨跌四象限 6 组 × 5 通道 × 22 × 全天(1分钟/30秒)          =  1,320
//!   M6 盘中最高/最低/最大量分钟前后 15 分钟（买/卖）                   =    528
//!   M7 30s/1m/5m/15m/30m/全天 跨尺度均值、差值与斜率                   =    528
//!   合计 44,694。
//! 全程确定性：固定种子 xorshift64、按成交先后排序、HashMap 只做查找不遍历，同股同日逐比特一致。

use crate::fast_csv_reader::{build_trade_record, read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// ============================================================================
// 常量
// ============================================================================

pub const N_MET: usize = 22;
const N_RAW: usize = 16;

pub const MET_NAMES: [&str; N_MET] = [
    // 0..16：v1 原始指标（公式与 v1 anneal_core16 一致）
    "r0_init", "final_r", "net_r_gain", "rel_r_eff", "half_life_norm", "steps_r80_norm",
    "steps_r90_norm", "inertia_norm", "decline_ratio", "max_drawdown", "dr_std", "mid_deter_ratio",
    "cold_gain", "hot_cold_std_ratio", "posD_ratio", "sign_flip_prob",
    // 16..20：真实顺序 − 打乱顺序（同步数、同随机交换序列）
    "final_r_ex", "net_r_gain_ex", "inertia_norm_ex", "dr_std_ex",
    // 20..22：订单量结构
    "later_gt_ratio", "tie_ratio",
];
/// 超额值对应的原始指标编号
const EX_FROM: [usize; 4] = [1, 2, 7, 10];

pub const SIDE7_NAMES: [&str; 7] = ["bid", "ask", "mixed", "actbid", "actask", "pasbid", "pasask"];
/// 矩阵 7 通道：前 5 个来自订单通道，后 2 个是差值
pub const MAT_CH_NAMES: [&str; 7] = ["bid", "ask", "mixed", "act", "pas", "bid_minus_ask", "act_minus_pas"];
const MAT_BASE_IDX: [usize; 5] = [0, 1, 2, 7, 8];
const N_MAT_COLS: usize = 7 * N_MET; // 154
/// 状态切割与四象限用的 5 个通道（与 v1 相同）：bid, ask, mixed, bid_minus_ask, act_minus_pas
const COND_CH: [usize; 5] = [0, 1, 2, 5, 6];

pub const FINE_WIN_NAMES: [&str; 12] = [
    "w0_open15", "w1_early15", "w2_morn_trend", "w3_morn_mid", "w4_pre_lunch", "w5_post_lunch",
    "w6_aft_early", "w7_aft_mid", "w8_late_warm", "w9_close12", "w10_morn2h", "w11_aft2h",
];
const FINE_WIN_BOUNDS: [(i64, i64); 12] = [
    (0, 900), (900, 1_800), (1_800, 3_600), (3_600, 5_400), (5_400, 7_200), (7_200, 9_000),
    (9_000, 10_800), (10_800, 12_600), (12_600, 13_500), (13_500, 14_220), (0, 7_200), (7_200, 14_220),
];
pub const WIN_CONTRAST_PAIRS: [(usize, usize, &str); 15] = [
    (9, 0, "close12_vs_open15"), (1, 0, "early15_vs_open15"), (2, 0, "morntrend_vs_open15"),
    (4, 2, "prelunch_vs_morntrend"), (5, 4, "postlunch_vs_prelunch"), (8, 7, "latewarm_vs_aftmid"),
    (9, 8, "close12_vs_latewarm"), (9, 7, "close12_vs_aftmid"), (11, 10, "aft2h_vs_morn2h"),
    (8, 1, "latewarm_vs_early15"), (5, 0, "postlunch_vs_open15"), (9, 4, "close12_vs_prelunch"),
    (3, 2, "mornmid_vs_morntrend"), (7, 5, "aftmid_vs_postlunch"), (6, 3, "aftearly_vs_mornmid"),
];
pub const REGIME10_NAMES: [&str; 10] = [
    "ret_sign", "ret_extreme", "ret_abs_move", "vol_level", "cnt_freq",
    "price_range", "act_buy_ratio", "net_inflow", "order_hhi", "mean_ord_size",
];
pub const QUAD_PAIR_NAMES: [&str; 6] = [
    "hv_up_vs_hv_dn", "lv_up_vs_lv_dn", "hv_up_vs_lv_up", "hv_dn_vs_lv_dn", "hv_up_vs_lv_dn", "hv_dn_vs_lv_up",
];
const QUAD_PAIRS: [(usize, usize); 6] = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];

/// 两种分辨率：(名字, 每格秒数, 格数)
const RESOLUTIONS: [(&str, i64, usize); 2] = [("m1", 60, 237), ("s30", 30, 474)];
const DAY_SEC: i64 = 14_220;
/// 时间分段：(名字前缀, 起始秒)；下午 13:00 起（前移后 7200 秒），尾盘为最后 30 分钟
const SEGMENTS: [(&str, i64); 3] = [("", 0), ("pm_", 7_200), ("tail30_", DAY_SEC - 1_800)];
const US: i64 = 1_000_000;

/// v1 的退火步数上限（步数 = max(6n², 600)，不超过上限）
const CAP_WIN: usize = 20_000;
const CAP_BIN: usize = 6_000;
const CAP_EVT: usize = 12_000;
const CAP_BLOCK: usize = 15_000;
const CAP_DAY: usize = 30_000;
const MAX_PROBE_PAIRS: usize = 2_000;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_NULL: u64 = 0xD1B5_4A32_D192_ED03;

const N_M1: usize = 12 * 7 * N_MET;
const N_M2: usize = 15 * 7 * N_MET;
const N_STATS_GROUPS_HEAD: usize = 18; // 降维输出：前 18 组单列统计，然后两两相关，最后 3 组单列统计
const N_STATS_GROUPS_TAIL: usize = 3;
const N_M3A_ONE: usize = (N_STATS_GROUPS_HEAD + N_STATS_GROUPS_TAIL) * N_MAT_COLS; // 3,234
const N_RAW_COLS: usize = 7 * N_RAW; // 112
const N_M3B: usize = N_RAW_COLS * (N_RAW_COLS - 1) / 2; // 6,216
const N_COND_COLS: usize = COND_CH.len() * N_MET; // 110
const N_M4_ONE: usize = 10 * N_COND_COLS + 9 * N_COND_COLS; // 2,090
const N_M5_ONE: usize = 6 * N_COND_COLS; // 660
const N_M6: usize = 12 * 2 * N_MET; // 528
const N_M7: usize = 4 * 3 * N_MET + 3 * 3 * N_MET + 3 * N_MET; // 528
pub const EXPECTED_LEN: usize =
    N_M1 + N_M2 + 6 * N_M3A_ONE + N_M3B + 6 * N_M4_ONE + 2 * N_M5_ONE + N_M6 + N_M7; // 44,694

// ============================================================================
// 确定性随机数
// ============================================================================

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0xDEAD_BEEF_DEAD_BEEF } else { seed } }
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
        ((self.next_u64() >> 32).wrapping_mul(n as u64) >> 32) as usize
    }
}

// ============================================================================
// 单序列退火
// ============================================================================

pub struct AnnealBuf {
    sorted: Vec<f32>,
    guess: Vec<f32>,
    null_target: Vec<f32>,
}

impl AnnealBuf {
    pub fn new() -> Self {
        Self {
            sorted: Vec::with_capacity(4096),
            guess: Vec::with_capacity(4096),
            null_target: Vec::with_capacity(4096),
        }
    }
}

/// v1 的步数规则
#[inline]
fn adaptive_steps(n: usize, cap: usize) -> usize {
    (n.saturating_mul(n).saturating_mul(6)).max(600).min(cap)
}

/// v1 anneal_core16 的退火过程与 16 个指标（升序初猜），结果与 v1 逐比特一致（sandbox/anneal_regime_v3_20261108 对拍 160 组）。
/// probe=false 时跳过两个探测指标（打乱对照用不到），对其余 14 个指标没有影响。
/// 提速：接受与否用条件选择代替分支，拒绝的一步对累计量是加 0，结果不变。
fn run_anneal_v1(target: &[f32], sorted: &[f32], guess: &mut Vec<f32>, sigma2: f64, m_max: usize, probe: bool) -> [f32; N_RAW] {
    let n = target.len();
    guess.clear();
    guess.extend_from_slice(sorted);
    let mut s = 0.0f64;
    for k in 0..n {
        let d = guess[k] as f64 - target[k] as f64;
        s += d * d;
    }
    let denom = 2.0 * sigma2 * n as f64;
    let inv_denom = 1.0 / denom;
    let r0 = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
    let hl_target = (1.0 + r0) * 0.5;
    let s_tol = (sigma2 * n as f64 * 1e-10).max(1e-12);
    let inv_m = 1.0 / (m_max as f64 - 1.0).max(1.0);

    let mut rng = XorShift64::new(SEED);
    let (mut prev_r, mut peak_r, mut max_dd) = (r0, r0, 0.0f64);
    let (mut a1, mut a2, mut a3) = (m_max, m_max, m_max);
    let mut inertia_sum = 0.0f64;
    let mut declines = 0u32;
    let (mut dr_s1, mut dr_s2) = (0.0f64, 0.0f64);
    let s1_end = m_max / 3;
    let s2_end = 2 * (m_max / 3);
    let (mut h_s1, mut h_s2) = (0.0f64, 0.0f64);
    let mut s2_decl = 0u32;
    let (mut c_s1, mut c_s2) = (0.0f64, 0.0f64);
    let mut r_at_s2 = r0;
    let (mut d_cnt, mut d_pos, mut d_flips) = (0u32, 0u32, 0u32);
    let mut prev_d_sign: i8 = 0;
    let mut actual = m_max;

    for t in 0..m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = (i + 1) % n;
        }
        let gi = guess[i];
        let gj = guess[j];
        if probe && d_cnt < 2000 && gi != gj {
            let dv = target[j] - target[i];
            d_cnt += 1;
            if dv > 0.0 {
                d_pos += 1;
            }
            let cur: i8 = if dv > 0.0 { 1 } else if dv < 0.0 { -1 } else { 0 };
            if prev_d_sign != 0 && cur != 0 && cur != prev_d_sign {
                d_flips += 1;
            }
            if cur != 0 {
                prev_d_sign = cur;
            }
        }
        let ds = 2.0f32 * (gi - gj) * (target[i] - target[j]);
        let ct = sigma2 * (1.0 - t as f64 * inv_m).max(0.0);
        let acc = (ds as f64) < ct; // v1 的「ds<0 或 ds<ct」在 ct≥0 时等价
        s += if acc { ds as f64 } else { 0.0 };
        guess[i] = if acc { gj } else { gi };
        guess[j] = if acc { gi } else { gj };
        let cr = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
        if t > 0 {
            let dv = cr - prev_r;
            let dv2 = dv * dv;
            dr_s1 += dv;
            dr_s2 += dv2;
            let dec = (cr < prev_r) as u32;
            declines += dec;
            if t < s1_end {
                h_s1 += dv;
                h_s2 += dv2;
            } else if t < s2_end {
                s2_decl += dec;
            } else {
                c_s1 += dv;
                c_s2 += dv2;
            }
        }
        if a1 == m_max && cr >= hl_target {
            a1 = t;
        }
        if a2 == m_max && cr >= 0.80 {
            a2 = t;
        }
        if a3 == m_max && cr >= 0.90 {
            a3 = t;
        }
        inertia_sum += 1.0 - cr;
        peak_r = peak_r.max(cr);
        max_dd = max_dd.max(peak_r - cr);
        if t + 1 == s2_end {
            r_at_s2 = cr;
        }
        prev_r = cr;
        if s <= s_tol {
            if t + 1 < s2_end {
                r_at_s2 = cr;
            }
            actual = t + 1;
            break;
        }
    }
    let final_r = prev_r;
    let dr_cnt = actual.saturating_sub(1) as u32;
    let s1_cnt = actual.min(s1_end).saturating_sub(1) as u32;
    let s2_cnt = actual.min(s2_end).saturating_sub(s1_end) as u32;
    let s3_cnt = actual.saturating_sub(s2_end) as u32;

    let sd = |c: u32, a: f64, b: f64| -> f64 {
        if c >= 2 {
            let nd = c as f64;
            ((b - a * a / nd) / (nd - 1.0)).max(0.0).sqrt()
        } else {
            0.0
        }
    };
    let m_f = m_max as f32;
    let act_f = actual.max(1) as f32;
    let r0_f = r0 as f32;
    let fin_f = final_r as f32;
    let hot = sd(s1_cnt, h_s1, h_s2);
    let cold = sd(s3_cnt, c_s1, c_s2);
    [
        r0_f,
        fin_f,
        fin_f - r0_f.abs(),
        (fin_f - r0_f) / (1.0 - r0_f + 1e-4),
        a1 as f32 / m_f,
        a2 as f32 / m_f,
        a3 as f32 / m_f,
        (inertia_sum / act_f as f64) as f32,
        declines as f32 / act_f,
        max_dd as f32,
        sd(dr_cnt, dr_s1, dr_s2) as f32,
        if s2_cnt > 0 { s2_decl as f32 / s2_cnt as f32 } else { 0.0 },
        (final_r - r_at_s2) as f32,
        (hot / (cold + 1e-6)) as f32,
        if d_cnt > 0 { d_pos as f32 / d_cnt as f32 } else { 0.5 },
        if d_cnt > 1 { d_flips as f32 / (d_cnt - 1) as f32 } else { 0.5 },
    ]
}

/// 订单量的时间趋势与雷同度：n 小时穷举全部 (先, 后) 对，n 大时抽 MAX_PROBE_PAIRS 对（与 v2 相同）
fn probe_order_pairs(seq: &[f32]) -> (f32, f32) {
    let n = seq.len();
    if n < 2 {
        return (f32::NAN, f32::NAN);
    }
    let (mut gt, mut tie, mut tot) = (0u32, 0u32, 0u32);
    let mut tally = |a: f32, b: f32| {
        tot += 1;
        if a == b {
            tie += 1;
        } else if b > a {
            gt += 1;
        }
    };
    if n * (n - 1) / 2 <= MAX_PROBE_PAIRS {
        for a in 0..n {
            for b in a + 1..n {
                tally(seq[a], seq[b]);
            }
        }
    } else {
        let mut rng = XorShift64::new(SEED ^ 0x5555);
        for _ in 0..MAX_PROBE_PAIRS {
            let i = rng.next_index(n);
            let mut j = rng.next_index(n);
            if j == i {
                j = (i + 1) % n;
            }
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            tally(seq[a], seq[b]);
        }
    }
    let later = if tot > tie { gt as f32 / (tot - tie) as f32 } else { 0.5 };
    (later, tie as f32 / tot as f32)
}

/// 一条订单量序列 → 22 指标
pub fn anneal_metrics22(seq: &[f32], cap: usize, buf: &mut AnnealBuf) -> [f32; N_MET] {
    let mut out = [f32::NAN; N_MET];
    let n = seq.len();
    if n < 3 {
        return out;
    }
    let (later, tie) = probe_order_pairs(seq);
    out[20] = later;
    out[21] = tie;
    let mut sum = 0.0f64;
    for &v in seq {
        if !v.is_finite() {
            return [f32::NAN; N_MET];
        }
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let mut sq = 0.0f64;
    for &v in seq {
        let d = v as f64 - mean;
        sq += d * d;
    }
    let sigma2 = sq / n as f64;
    if sigma2 <= 1e-20 {
        return out; // 全部订单量相同：退火指标为空，订单量结构照常输出
    }
    let m = adaptive_steps(n, cap);

    buf.sorted.clear();
    buf.sorted.extend_from_slice(seq);
    buf.sorted.sort_by(|a, b| a.total_cmp(b));
    let raw = run_anneal_v1(seq, &buf.sorted, &mut buf.guess, sigma2, m, true);
    out[..N_RAW].copy_from_slice(&raw);

    buf.null_target.clear();
    buf.null_target.extend_from_slice(seq);
    let mut rng = XorShift64::new(SEED_NULL);
    for i in (1..n).rev() {
        let j = rng.next_index(i + 1);
        buf.null_target.swap(i, j);
    }
    let null = run_anneal_v1(&buf.null_target, &buf.sorted, &mut buf.guess, sigma2, m, false);
    for (e, &k) in EX_FROM.iter().enumerate() {
        out[N_RAW + e] = raw[k] - null[k];
    }
    out
}

// ============================================================================
// 订单聚合：全天一次，每个订单归属首次成交时刻（与 v2 相同）
// ============================================================================

/// 0 bid, 1 ask, 2 mixed, 3 actbid, 4 actask, 5 pasbid, 6 pasask, 7 act(=3∪4), 8 pas(=5∪6)
struct OrderChannels {
    vol: [Vec<f32>; 9],
    t_us: [Vec<i64>; 9],
}

impl OrderChannels {
    #[inline]
    fn slice(&self, ch: usize, lo_us: i64, hi_us: i64) -> &[f32] {
        let t = &self.t_us[ch];
        let lo = t.partition_point(|&x| x < lo_us);
        let hi = t.partition_point(|&x| x < hi_us);
        &self.vol[ch][lo..hi.max(lo)]
    }
}

#[derive(Clone, Copy)]
struct Agg {
    vol: f32,
    n_act: u32,
    n_pas: u32,
    first_idx: u32,
    first_t: i64,
}

type OrdTuple = (u32, u8, f32, i64);

fn merge_by_idx(a: &[OrdTuple], b: &[OrdTuple]) -> (Vec<f32>, Vec<i64>) {
    let mut v = Vec::with_capacity(a.len() + b.len());
    let mut t = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let take_a = j >= b.len() || (i < a.len() && (a[i].0, a[i].1) <= (b[j].0, b[j].1));
        let x = if take_a {
            i += 1;
            a[i - 1]
        } else {
            j += 1;
            b[j - 1]
        };
        v.push(x.2);
        t.push(x.3);
    }
    (v, t)
}

fn build_order_channels(trades: &[TradeRecord]) -> OrderChannels {
    let mut idx_bid: FxHashMap<i64, u32> = FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());
    let mut idx_ask: FxHashMap<i64, u32> = FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());
    let mut bids: Vec<Agg> = Vec::with_capacity(trades.len() / 2 + 1);
    let mut asks: Vec<Agg> = Vec::with_capacity(trades.len() / 2 + 1);
    for (i, t) in trades.iter().enumerate() {
        let new = Agg { vol: 0.0, n_act: 0, n_pas: 0, first_idx: i as u32, first_t: t.time_us };
        let bi = *idx_bid.entry(t.bid_order).or_insert_with(|| {
            bids.push(new);
            (bids.len() - 1) as u32
        }) as usize;
        let b = &mut bids[bi];
        b.vol += t.volume;
        if t.flag == 66 {
            b.n_act += 1;
        } else if t.flag == 83 {
            b.n_pas += 1;
        }
        let ai = *idx_ask.entry(t.ask_order).or_insert_with(|| {
            asks.push(new);
            (asks.len() - 1) as u32
        }) as usize;
        let a = &mut asks[ai];
        a.vol += t.volume;
        if t.flag == 83 {
            a.n_act += 1;
        } else if t.flag == 66 {
            a.n_pas += 1;
        }
    }
    let to_tuple = |v: &[Agg], side: u8, pick: &dyn Fn(&Agg) -> bool| -> Vec<OrdTuple> {
        v.iter().filter(|e| pick(e)).map(|e| (e.first_idx, side, e.vol, e.first_t)).collect()
    };
    let all = |_: &Agg| true;
    let act_only = |e: &Agg| e.n_act > 0 && e.n_pas == 0;
    let pas_only = |e: &Agg| e.n_pas > 0 && e.n_act == 0;
    let tb = to_tuple(&bids, 0, &all);
    let ta = to_tuple(&asks, 1, &all);
    let tab = to_tuple(&bids, 0, &act_only);
    let taa = to_tuple(&asks, 1, &act_only);
    let tpb = to_tuple(&bids, 0, &pas_only);
    let tpa = to_tuple(&asks, 1, &pas_only);
    let split = |x: &[OrdTuple]| -> (Vec<f32>, Vec<i64>) {
        (x.iter().map(|e| e.2).collect(), x.iter().map(|e| e.3).collect())
    };
    let (v0, t0) = split(&tb);
    let (v1, t1) = split(&ta);
    let (v2, t2) = merge_by_idx(&tb, &ta);
    let (v3, t3) = split(&tab);
    let (v4, t4) = split(&taa);
    let (v5, t5) = split(&tpb);
    let (v6, t6) = split(&tpa);
    let (v7, t7) = merge_by_idx(&tab, &taa);
    let (v8, t8) = merge_by_idx(&tpb, &tpa);
    OrderChannels { vol: [v0, v1, v2, v3, v4, v5, v6, v7, v8], t_us: [t0, t1, t2, t3, t4, t5, t6, t7, t8] }
}

// ============================================================================
// 辅助统计
// ============================================================================

#[inline]
fn sdiff(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() { a - b } else { f32::NAN }
}

fn mean_mask(vals: &[f32], mask: &[bool]) -> f32 {
    let (mut s, mut c) = (0.0f64, 0usize);
    for (v, &m) in vals.iter().zip(mask) {
        if m && v.is_finite() {
            s += *v as f64;
            c += 1;
        }
    }
    if c == 0 { f32::NAN } else { (s / c as f64) as f32 }
}

fn mean_finite(vals: &[f32]) -> f32 {
    let (mut s, mut c) = (0.0f64, 0usize);
    for &v in vals {
        if v.is_finite() {
            s += v as f64;
            c += 1;
        }
    }
    if c == 0 { f32::NAN } else { (s / c as f64) as f32 }
}

/// 两列共同有效行上的皮尔逊相关；有效行 < 5 为空，某侧无波动记 0
fn corr_finite(a: &[f32], b: &[f32]) -> f32 {
    let (mut n, mut sa, mut sb) = (0usize, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            n += 1;
            sa += x as f64;
            sb += y as f64;
        }
    }
    if n < 5 {
        return f32::NAN;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut cov, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            let dx = x as f64 - ma;
            let dy = y as f64 - mb;
            cov += dx * dy;
            va += dx * dx;
            vb += dy * dy;
        }
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    (cov / (va.sqrt() * vb.sqrt())) as f32
}

fn top_bot_20_masks(state: &[f32]) -> (Vec<bool>, Vec<bool>) {
    let n = state.len();
    let mut valid: Vec<(f32, usize)> =
        state.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, &v)| (v, i)).collect();
    valid.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let (mut hi, mut lo) = (vec![false; n], vec![false; n]);
    let m = valid.len();
    if m < 5 {
        return (hi, lo);
    }
    let k = m / 5;
    for &(_, i) in &valid[..k] {
        lo[i] = true;
    }
    for &(_, i) in &valid[m - k..] {
        hi[i] = true;
    }
    (hi, lo)
}

// ============================================================================
// 时间格矩阵
// ============================================================================

struct BinMatrix {
    n_rows: usize,
    /// 列优先：cols[c][row]，c = 通道 × 22 + 指标
    cols: Vec<Vec<f32>>,
    states: [Vec<f32>; 10],
    high: Vec<f32>,
    low: Vec<f32>,
    vol: Vec<f32>,
}

fn build_bin_matrix(
    trades: &[TradeRecord],
    oc: &OrderChannels,
    t0: i64,
    width_sec: i64,
    n_rows: usize,
    buf: &mut AnnealBuf,
) -> BinMatrix {
    let mut cols = vec![vec![f32::NAN; n_rows]; N_MAT_COLS];
    let nan = || vec![f32::NAN; n_rows];
    let (mut ret, mut rng, mut vol, mut cnt) = (nan(), nan(), nan(), nan());
    let (mut abr, mut net, mut hhi, mut msz) = (nan(), nan(), nan(), nan());
    let (mut high, mut low) = (nan(), nan());
    let mut prev_close = f32::NAN;
    let bounds: Vec<usize> = (0..=n_rows)
        .map(|b| {
            let edge = t0 + ((b as i64 * width_sec).min(DAY_SEC)) * US;
            trades.partition_point(|t| t.time_us < edge)
        })
        .collect();

    for r in 0..n_rows {
        let (lo, hi) = (bounds[r], bounds[r + 1]);
        if hi <= lo + 2 {
            continue;
        }
        let sl = &trades[lo..hi];
        let p_open = sl[0].price;
        let p_close = sl[sl.len() - 1].price;
        let (mut ph, mut pl, mut vs, mut vb) = (p_open, p_open, 0.0f64, 0.0f64);
        for t in sl {
            ph = ph.max(t.price);
            pl = pl.min(t.price);
            vs += t.volume as f64;
            if t.flag == 66 {
                vb += t.volume as f64;
            }
        }
        let base = if prev_close.is_finite() { prev_close } else { p_open };
        if base > 0.0 {
            ret[r] = p_close / base - 1.0;
        }
        if p_open > 0.0 {
            rng[r] = (ph - pl) / p_open;
        }
        prev_close = p_close;
        high[r] = ph;
        low[r] = pl;
        vol[r] = vs as f32;
        cnt[r] = sl.len() as f32;
        if vs > 0.0 {
            abr[r] = (vb / vs) as f32;
        }
        net[r] = (2.0 * vb - vs) as f32;

        let lo_us = t0 + r as i64 * width_sec * US;
        let hi_us = t0 + ((r as i64 + 1) * width_sec).min(DAY_SEC) * US;
        let mix = oc.slice(2, lo_us, hi_us);
        let tot: f64 = mix.iter().map(|&v| v as f64).sum();
        if tot > 0.0 {
            hhi[r] = mix.iter().map(|&v| (v as f64 / tot).powi(2)).sum::<f64>() as f32;
            msz[r] = (tot / mix.len() as f64) as f32;
        }
        let mut ch = [[f32::NAN; N_MET]; 5];
        for (c, &oc_idx) in MAT_BASE_IDX.iter().enumerate() {
            ch[c] = anneal_metrics22(oc.slice(oc_idx, lo_us, hi_us), CAP_BIN, buf);
            for k in 0..N_MET {
                cols[c * N_MET + k][r] = ch[c][k];
            }
        }
        for k in 0..N_MET {
            cols[5 * N_MET + k][r] = sdiff(ch[0][k], ch[1][k]);
            cols[6 * N_MET + k][r] = sdiff(ch[3][k], ch[4][k]);
        }
    }
    let abs_ret: Vec<f32> = ret.iter().map(|v| v.abs()).collect();
    BinMatrix {
        n_rows,
        cols,
        states: [ret.clone(), ret, abs_ret, vol.clone(), cnt, rng, abr, net, hhi, msz],
        high,
        low,
        vol,
    }
}

fn tail_rows(bm: &BinMatrix, start: usize) -> BinMatrix {
    let cut = |v: &Vec<f32>| v[start..].to_vec();
    BinMatrix {
        n_rows: bm.n_rows - start,
        cols: bm.cols.iter().map(cut).collect(),
        states: [
            cut(&bm.states[0]), cut(&bm.states[1]), cut(&bm.states[2]), cut(&bm.states[3]), cut(&bm.states[4]),
            cut(&bm.states[5]), cut(&bm.states[6]), cut(&bm.states[7]), cut(&bm.states[8]), cut(&bm.states[9]),
        ],
        high: cut(&bm.high),
        low: cut(&bm.low),
        vol: cut(&bm.vol),
    }
}

#[inline]
fn is_raw_col(c: usize) -> bool {
    c % N_MET < N_RAW
}

/// M3：对 154 列做降维；单列统计写入 stats_out；want_pairs 时把「两列都是原始指标」的两两相关写入 pairs_out
fn reduce_matrix(bm: &BinMatrix, stats_out: &mut Vec<f32>, pairs_out: Option<&mut Vec<f32>>) {
    let mut mat = Array2::<f32>::from_elem((bm.n_rows, N_MAT_COLS), f32::NAN);
    for c in 0..N_MAT_COLS {
        for r in 0..bm.n_rows {
            mat[[r, c]] = bm.cols[c][r];
        }
    }
    let vals = features::get_features_factors_rust_values_only(&mat.view(), false);
    let n_pairs_all = N_MAT_COLS * (N_MAT_COLS - 1) / 2;
    let head = N_STATS_GROUPS_HEAD * N_MAT_COLS;
    if vals.len() != head + n_pairs_all + N_STATS_GROUPS_TAIL * N_MAT_COLS {
        stats_out.extend(std::iter::repeat(f32::NAN).take(N_M3A_ONE));
        if let Some(p) = pairs_out {
            p.extend(std::iter::repeat(f32::NAN).take(N_M3B));
        }
        return;
    }
    stats_out.extend_from_slice(&vals[..head]);
    stats_out.extend_from_slice(&vals[head + n_pairs_all..]);
    if let Some(p) = pairs_out {
        let mut idx = head;
        for i in 0..N_MAT_COLS {
            for j in i + 1..N_MAT_COLS {
                if is_raw_col(i) && is_raw_col(j) {
                    p.push(vals[idx]);
                }
                idx += 1;
            }
        }
    }
}

/// M4：10 种行情状态 × 5 通道 × 22：高低组差 +（急拉急跌除外）状态相关
fn regime_block(bm: &BinMatrix, out: &mut Vec<f32>) {
    for (ri, st) in bm.states.iter().enumerate() {
        let (hi, lo) = if ri == 0 {
            (
                st.iter().map(|&v| v.is_finite() && v > 0.0).collect::<Vec<_>>(),
                st.iter().map(|&v| v.is_finite() && v < 0.0).collect::<Vec<_>>(),
            )
        } else {
            top_bot_20_masks(st)
        };
        for &ch in &COND_CH {
            for k in 0..N_MET {
                let col = &bm.cols[ch * N_MET + k];
                out.push(sdiff(mean_mask(col, &hi), mean_mask(col, &lo)));
                if ri != 1 {
                    out.push(corr_finite(col, st));
                }
            }
        }
    }
}

/// M5：按成交量中位数 × 涨跌分四象限，6 组两两对比
fn quad_block(bm: &BinMatrix, out: &mut Vec<f32>) {
    let ret = &bm.states[0];
    let mut vv: Vec<f32> = bm.vol.iter().copied().filter(|x| x.is_finite()).collect();
    vv.sort_by(|a, b| a.total_cmp(b));
    let med = if vv.is_empty() { f32::NAN } else { vv[vv.len() / 2] };
    let mut q = vec![vec![false; bm.n_rows]; 4];
    if med.is_finite() {
        for r in 0..bm.n_rows {
            let (v, x) = (bm.vol[r], ret[r]);
            if !(v.is_finite() && x.is_finite()) || x == 0.0 {
                continue;
            }
            let k = match (v >= med, x > 0.0) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            };
            q[k][r] = true;
        }
    }
    for &(a, b) in &QUAD_PAIRS {
        for &ch in &COND_CH {
            for k in 0..N_MET {
                let col = &bm.cols[ch * N_MET + k];
                out.push(sdiff(mean_mask(col, &q[a]), mean_mask(col, &q[b])));
            }
        }
    }
}

// ============================================================================
// 主计算
// ============================================================================

pub fn compute_anneal_regime_raw_excess_from_records(trades: &[TradeRecord]) -> Vec<f32> {
    if trades.len() < 10 {
        return vec![f32::NAN; EXPECTED_LEN];
    }
    let mut out = Vec::with_capacity(EXPECTED_LEN);
    let mut buf = AnnealBuf::new();
    let day_us = 86_400 * US;
    let first = trades[0].time_us;
    let t0 = first - first.rem_euclid(day_us) + (9 * 3600 + 30 * 60) * US;
    let at = |sec: i64| t0 + sec * US;
    let oc = build_order_channels(trades);

    // ---- M1 ----
    let mut win = [[[f32::NAN; N_MET]; 7]; 12];
    for (w, &(lo, hi)) in FINE_WIN_BOUNDS.iter().enumerate() {
        for s in 0..7 {
            win[w][s] = anneal_metrics22(oc.slice(s, at(lo), at(hi)), CAP_WIN, &mut buf);
            out.extend_from_slice(&win[w][s]);
        }
    }
    // ---- M2 ----
    for &(a, b, _) in &WIN_CONTRAST_PAIRS {
        for s in 0..7 {
            for k in 0..N_MET {
                out.push(sdiff(win[a][s][k], win[b][s][k]));
            }
        }
    }
    // ---- 时间格矩阵：全天各算一次，分段取行 ----
    let mats: Vec<BinMatrix> = RESOLUTIONS
        .iter()
        .map(|&(_, width, n_rows)| build_bin_matrix(trades, &oc, t0, width, n_rows, &mut buf))
        .collect();
    let mut seg_mats: Vec<BinMatrix> = Vec::with_capacity(6);
    for &(_, start_sec) in &SEGMENTS {
        for (ri, &(_, width, _)) in RESOLUTIONS.iter().enumerate() {
            seg_mats.push(tail_rows(&mats[ri], (start_sec / width) as usize));
        }
    }
    // ---- M3a 单列统计（6 份）+ M3b 两两相关（全天 1 分钟，原始指标）----
    let mut pairs = Vec::with_capacity(N_M3B);
    for (i, bm) in seg_mats.iter().enumerate() {
        reduce_matrix(bm, &mut out, if i == 0 { Some(&mut pairs) } else { None });
    }
    out.extend_from_slice(&pairs);
    // ---- M4（6 份）----
    for bm in &seg_mats {
        regime_block(bm, &mut out);
    }
    // ---- M5（全天 1 分钟、全天 30 秒）----
    for bm in &seg_mats[..2] {
        quad_block(bm, &mut out);
    }

    // ---- M6 事件锚点（第 15~221 分钟里找）----
    let m1 = &mats[0];
    let pick = |vals: &[f32], want_max: bool| -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for r in 15..=221 {
            let v = vals[r];
            if !v.is_finite() {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, b)) => if want_max { v > b } else { v < b },
            };
            if better {
                best = Some((r, v));
            }
        }
        best.map(|x| x.0)
    };
    let centers = [pick(&m1.high, true), pick(&m1.low, false), pick(&m1.vol, true)];
    let mut evt = [[[f32::NAN; N_MET]; 2]; 6];
    for (e, c) in centers.iter().enumerate() {
        if let Some(c) = *c {
            let c = c as i64;
            for (sub, &(lo_m, hi_m)) in [(c - 15, c), (c + 1, c + 16)].iter().enumerate() {
                for d in 0..2 {
                    evt[e * 2 + sub][d] = anneal_metrics22(oc.slice(d, at(lo_m * 60), at(hi_m * 60)), CAP_EVT, &mut buf);
                }
            }
        }
    }
    for w in 0..6 {
        for d in 0..2 {
            out.extend_from_slice(&evt[w][d]);
        }
    }
    const EVT_CONTRASTS: [(usize, usize); 6] = [(1, 0), (3, 2), (5, 4), (0, 2), (1, 3), (4, 0)];
    for &(a, b) in &EVT_CONTRASTS {
        for d in 0..2 {
            for k in 0..N_MET {
                out.push(sdiff(evt[a][d][k], evt[b][d][k]));
            }
        }
    }

    // ---- M7 跨尺度（bid/ask/mixed）----
    let mut scale = [[[f32::NAN; N_MET]; 3]; 6];
    for (si, bm) in [(0usize, &mats[1]), (1usize, &mats[0])] {
        for d in 0..3 {
            for k in 0..N_MET {
                scale[si][d][k] = mean_finite(&bm.cols[d * N_MET + k]);
            }
        }
    }
    for (si, width) in [(2usize, 300i64), (3, 900), (4, 1800)] {
        let n_blocks = ((DAY_SEC + width - 1) / width) as usize;
        let mut acc = [[0.0f64; N_MET]; 3];
        let mut cnt = [[0usize; N_MET]; 3];
        for b in 0..n_blocks as i64 {
            let (lo, hi) = (b * width, ((b + 1) * width).min(DAY_SEC));
            for d in 0..3 {
                let v = anneal_metrics22(oc.slice(d, at(lo), at(hi)), CAP_BLOCK, &mut buf);
                for k in 0..N_MET {
                    if v[k].is_finite() {
                        acc[d][k] += v[k] as f64;
                        cnt[d][k] += 1;
                    }
                }
            }
        }
        for d in 0..3 {
            for k in 0..N_MET {
                if cnt[d][k] > 0 {
                    scale[si][d][k] = (acc[d][k] / cnt[d][k] as f64) as f32;
                }
            }
        }
    }
    for d in 0..3 {
        scale[5][d] = anneal_metrics22(oc.slice(d, at(0), at(DAY_SEC)), CAP_DAY, &mut buf);
    }
    for si in 2..6 {
        for d in 0..3 {
            out.extend_from_slice(&scale[si][d]);
        }
    }
    for &(a, b) in &[(1usize, 3usize), (2, 5), (0, 2)] {
        for d in 0..3 {
            for k in 0..N_MET {
                out.push(sdiff(scale[a][d][k], scale[b][d][k]));
            }
        }
    }
    let xs: [f64; 6] = [30f64.ln(), 60f64.ln(), 300f64.ln(), 900f64.ln(), 1800f64.ln(), (DAY_SEC as f64).ln()];
    for d in 0..3 {
        for k in 0..N_MET {
            let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for si in 0..6 {
                let y = scale[si][d][k];
                if y.is_finite() {
                    let (x, y) = (xs[si], y as f64);
                    n += 1;
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    sxy += x * y;
                }
            }
            let slope = if n >= 3 {
                let den = sxx - sx * sx / n as f64;
                if den.abs() > 1e-12 { ((sxy - sx * sy / n as f64) / den) as f32 } else { 0.0 }
            } else {
                f32::NAN
            };
            out.push(slope);
        }
    }
    debug_assert_eq!(out.len(), EXPECTED_LEN);
    out
}

// ============================================================================
// 因子名（与输出顺序逐一对齐）
// ============================================================================

fn mat_col_names(tag: &str) -> Vec<String> {
    let mut v = Vec::with_capacity(N_MAT_COLS);
    for ch in &MAT_CH_NAMES {
        for m in &MET_NAMES {
            v.push(format!("s3_{}_{}_{}", tag, ch, m));
        }
    }
    v
}

pub fn anneal_regime_raw_excess_names() -> Vec<String> {
    let mut names = Vec::with_capacity(EXPECTED_LEN);
    for w in &FINE_WIN_NAMES {
        for s in &SIDE7_NAMES {
            for m in &MET_NAMES {
                names.push(format!("s3_{}_{}_{}", w, s, m));
            }
        }
    }
    for &(_, _, p) in &WIN_CONTRAST_PAIRS {
        for s in &SIDE7_NAMES {
            for m in &MET_NAMES {
                names.push(format!("s3_wdiff_{}_{}_{}", p, s, m));
            }
        }
    }
    let tags: Vec<String> = SEGMENTS
        .iter()
        .flat_map(|&(seg, _)| RESOLUTIONS.iter().map(move |&(res, _, _)| format!("{}{}", seg, res)))
        .collect();
    let n_pairs_all = N_MAT_COLS * (N_MAT_COLS - 1) / 2;
    let head = N_STATS_GROUPS_HEAD * N_MAT_COLS;
    let mut pair_names = Vec::new();
    for (ti, tag) in tags.iter().enumerate() {
        let cn = mat_col_names(tag);
        let dummy = Array2::<f32>::ones((3, N_MAT_COLS));
        let (_, red) = features::get_features_factors_rust_full(&dummy.view(), &cn, false);
        assert_eq!(red.len(), head + n_pairs_all + N_STATS_GROUPS_TAIL * N_MAT_COLS);
        names.extend_from_slice(&red[..head]);
        names.extend_from_slice(&red[head + n_pairs_all..]);
        if ti == 0 {
            let mut idx = head;
            for i in 0..N_MAT_COLS {
                for j in i + 1..N_MAT_COLS {
                    if is_raw_col(i) && is_raw_col(j) {
                        pair_names.push(red[idx].clone());
                    }
                    idx += 1;
                }
            }
        }
    }
    names.extend(pair_names);
    let cond_ch_names: Vec<&str> = COND_CH.iter().map(|&c| MAT_CH_NAMES[c]).collect();
    for tag in &tags {
        for (ri, r) in REGIME10_NAMES.iter().enumerate() {
            for ch in &cond_ch_names {
                for m in &MET_NAMES {
                    names.push(format!("s3_cond_{}_{}_{}_{}_hi_lo_diff", tag, r, ch, m));
                    if ri != 1 {
                        names.push(format!("s3_cond_{}_{}_{}_{}_state_corr", tag, r, ch, m));
                    }
                }
            }
        }
    }
    for tag in &tags[..2] {
        for q in &QUAD_PAIR_NAMES {
            for ch in &cond_ch_names {
                for m in &MET_NAMES {
                    names.push(format!("s3_quad_{}_{}_{}_{}_diff", tag, q, ch, m));
                }
            }
        }
    }
    for ew in &["pre_high15", "post_high15", "pre_low15", "post_low15", "pre_maxvol15", "post_maxvol15"] {
        for d in &["bid", "ask"] {
            for m in &MET_NAMES {
                names.push(format!("s3_evt_{}_{}_{}", ew, d, m));
            }
        }
    }
    for ed in &[
        "post_vs_pre_high", "post_vs_pre_low", "post_vs_pre_maxvol",
        "pre_high_vs_pre_low", "post_high_vs_post_low", "pre_maxvol_vs_pre_high",
    ] {
        for d in &["bid", "ask"] {
            for m in &MET_NAMES {
                names.push(format!("s3_evtdiff_{}_{}_{}", ed, d, m));
            }
        }
    }
    for sc in &["scale5m", "scale15m", "scale30m", "scalefullday"] {
        for d in &["bid", "ask", "mixed"] {
            for m in &MET_NAMES {
                names.push(format!("s3_{}_{}_{}", sc, d, m));
            }
        }
    }
    for p in &["scale1m_vs_15m", "scale5m_vs_fullday", "scale30s_vs_5m"] {
        for d in &["bid", "ask", "mixed"] {
            for m in &MET_NAMES {
                names.push(format!("s3_{}_{}_{}_diff", p, d, m));
            }
        }
    }
    for d in &["bid", "ask", "mixed"] {
        for m in &MET_NAMES {
            names.push(format!("s3_scale_slope_{}_{}", d, m));
        }
    }
    assert_eq!(names.len(), EXPECTED_LEN);
    names
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

pub fn compute_anneal_regime_raw_excess_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let trades = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    Ok(compute_anneal_regime_raw_excess_from_records(&trades))
}

/// 单股单日计算（读盘），返回 44694 个因子值
#[pyfunction]
pub fn py_anneal_regime_raw_excess(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_regime_raw_excess_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_anneal_regime_raw_excess_names() -> Vec<String> {
    anneal_regime_raw_excess_names()
}

/// 传入逐笔数组计算（与读盘版共用同一内核）
#[pyfunction]
pub fn py_anneal_regime_raw_excess_from_data(
    exchtime_us: PyReadonlyArray1<i64>,
    price: PyReadonlyArray1<f32>,
    volume: PyReadonlyArray1<f32>,
    turnover: PyReadonlyArray1<f32>,
    flag: PyReadonlyArray1<i32>,
    bid_order: PyReadonlyArray1<i64>,
    ask_order: PyReadonlyArray1<i64>,
) -> PyResult<Vec<f32>> {
    let (t, p, v, tn, f, bo, ao) = (
        exchtime_us.as_array(), price.as_array(), volume.as_array(), turnover.as_array(),
        flag.as_array(), bid_order.as_array(), ask_order.as_array(),
    );
    let n = t.len();
    if [p.len(), v.len(), tn.len(), f.len(), bo.len(), ao.len()].iter().any(|&l| l != n) {
        return Err(pyo3::exceptions::PyValueError::new_err("所有输入数组长度必须一致"));
    }
    let trades: Vec<TradeRecord> = (0..n)
        .filter(|&i| f[i] != 32)
        .filter_map(|i| build_trade_record(t[i], p[i], v[i], tn[i], f[i], bo[i], ao[i], true))
        .collect();
    Ok(compute_anneal_regime_raw_excess_from_records(&trades))
}
