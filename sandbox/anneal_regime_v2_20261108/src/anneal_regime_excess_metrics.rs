//! 柜员游戏拓展1 新版：时空尺度与行情条件下的「超额可还原性」（anneal_regime_excess）。
//!
//! 与旧版 anneal_regime_diff（保持不动）相比，修了五类问题：
//!   1. 时间精度：旧版用 f32 秒（当前日期附近只精确到 128 秒）切分钟，237 个分钟里一半为空。
//!      新版一律用 time_us（i64 微秒）切窗口，分钟/30 秒边界精确。
//!   2. 订单数干扰：旧版退火步数 = min(6n², 上限)，订单一多就触顶，还原度机械地随订单数下降。
//!      新版步数 = 每个订单固定 SWEEPS 次尝试（大窗口才有上限），并对同一段订单做一次
//!      「随机打乱顺序」的对照退火（同一批数值、同一套随机交换序列），输出「真实 − 打乱」的超额值，
//!      只保留订单先后顺序本身带来的那部分可还原性。
//!   3. 两个探测指标：旧版随机抽两个位置比大小，期望恒为 0.5，实际只反映「订单量相同」的比例。
//!      新版改成两个含义明确的指标：later_gt_ratio（后来的订单比先来的大的比例，衡量规模的时间趋势）
//!      与 tie_ratio（两笔订单量完全相同的比例，衡量整数单/雷同单）。
//!   4. 爆值与重复：去掉所有「比值」类输出（分母可为负或接近 0）；热冷期波动比改为有界的占比；
//!      「涨跌分钟」与「急拉急跌分钟」的状态相关完全相同，只保留一份。
//!   5. 覆盖与空值：5/15/30 分钟尺度最后一段不足长度也保留；事件锚点只在 09:45~14:42 之间找，
//!      保证事件前后 15 分钟窗口都完整。
//! 另新增 30 秒分辨率的整套矩阵降维、行情状态切割与四象限对比；
//! 并且这三块（M3/M4/M5，原本用全天所有时间格）再按「只用下午」「只用尾盘 30 分钟」各算一遍，
//! 时间格的退火结果全天只算一次，分段只是取其中的行，不增加退火开销。
//!
//! 订单归属：每个订单按「首次成交时刻」归属到窗口，体积用全天累计成交量（旧版是在每个窗口内
//! 重新聚合，跨窗口的订单会被切成碎片）。
//!
//! 模块与维度：
//!   M1 12 个日内时段 × 7 方向 × 16 指标                       = 1,344
//!   M2 15 组时段对比（只做差）× 7 方向 × 16 指标                = 1,680
//!   以下 M3/M4/M5 各有 3 段（全天 / 下午 / 尾盘30分钟）× 2 分辨率（1 分钟 / 30 秒）= 6 份：
//!   M3 时间格矩阵(行×80)降维 4,840 × 6                           = 29,040
//!   M4 10 种行情状态（高低组差 + 状态相关）1,520 × 6             =  9,120
//!   M5 6 组放缩量涨跌四象限对比 480 × 6                          =  2,880
//!   M6 盘中最高/最低/最大量分钟前后 15 分钟                      =    384
//!   M7 30s/1m/5m/15m/30m/全天 跨尺度均值、差值与斜率             =    384
//!   合计 44,832。
//! 全程确定性：固定种子 xorshift64、按成交先后排序、无 HashMap 遍历，同股同日逐比特一致。

use crate::fast_csv_reader::{build_trade_record, read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// ============================================================================
// 常量
// ============================================================================

pub const N_CORE: usize = 16;

/// 每条订单序列输出的 16 个指标。带 _ex 的是「真实顺序 − 打乱顺序」的超额值。
pub const CORE_NAMES: [&str; N_CORE] = [
    "r0_init",            // 0: 升序初猜与真实顺序的相关（真实值）
    "final_r_ex",         // 1: 最终还原相关
    "net_r_gain_ex",      // 2: 最终相关 − |初始相关|
    "rel_r_eff_ex",       // 3: (最终 − 初始) / (1 − 初始)
    "half_life_norm_ex",  // 4: 走完一半还原所需步数占比
    "steps_r80_norm_ex",  // 5: 到达 r=0.80 的步数占比
    "steps_r90_norm_ex",  // 6: 到达 r=0.90 的步数占比
    "inertia_norm_ex",    // 7: 平均未还原程度 mean(1 − r_t)
    "decline_ratio_ex",   // 8: 恶化步数占比
    "max_drawdown_ex",    // 9: 退火过程最大回撤
    "dr_std_ex",          // 10: 单步 Δr 标准差
    "mid_deter_ratio_ex", // 11: 中段恶化步数占比
    "cold_gain_ex",       // 12: 后 1/3 净增益
    "hot_std_share_ex",   // 13: 前段 Δr 波动 /（前段 + 后段）
    "later_gt_ratio",     // 14: 后来订单量 > 先来订单量的比例（不含相等）
    "tie_ratio",          // 15: 两笔订单量完全相等的比例
];

/// 7 个方向（M1/M2 用）在订单通道里的编号
pub const SIDE7_NAMES: [&str; 7] = ["bid", "ask", "mixed", "actbid", "actask", "pasbid", "pasask"];
/// 矩阵通道：bid, ask, mixed, act, pas → 订单通道编号
pub const MAT_CH_NAMES: [&str; 5] = ["bid", "ask", "mixed", "act", "pas"];
const MAT_CH_IDX: [usize; 5] = [0, 1, 2, 7, 8];
pub const N_MAT_COLS: usize = 5 * N_CORE; // 80

pub const FINE_WIN_NAMES: [&str; 12] = [
    "w0_open15", "w1_early15", "w2_morn_trend", "w3_morn_mid", "w4_pre_lunch", "w5_post_lunch",
    "w6_aft_early", "w7_aft_mid", "w8_late_warm", "w9_close12", "w10_morn2h", "w11_aft2h",
];
/// 相对 09:30:00 的秒数（下午已前移 90 分钟，13:00 → 7200 秒）
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

/// 两种分辨率：(名字, 每格秒数, 格数)。14220 秒 = 237 分钟 = 474 个 30 秒。
const RESOLUTIONS: [(&str, i64, usize); 2] = [("m1", 60, 237), ("s30", 30, 474)];

const DAY_SEC: i64 = 14_220;

/// M3/M4/M5 的时间分段：(名字前缀, 起始秒)。下午从 13:00（前移后 7200 秒）起，尾盘为最后 30 分钟（14:27 起）。
const SEGMENTS: [(&str, i64); 3] = [("", 0), ("pm_", 7_200), ("tail30_", DAY_SEC - 1_800)];
const US: i64 = 1_000_000;

/// 每个订单分到的退火尝试次数；步数 = SWEEPS × n，下限 MIN_STEPS，大窗口另设上限。
const SWEEPS: usize = 20;
const MIN_STEPS: usize = 200;
const CAP_BIN: usize = 10_000;
const CAP_WIN: usize = 60_000;
const CAP_BLOCK: usize = 30_000;
const CAP_DAY: usize = 100_000;
const MAX_PROBE_PAIRS: usize = 2_000;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_NULL: u64 = 0xD1B5_4A32_D192_ED03;

const N_M1: usize = 12 * 7 * N_CORE;
const N_M2: usize = 15 * 7 * N_CORE;
const N_M3_ONE: usize = 21 * N_MAT_COLS + N_MAT_COLS * (N_MAT_COLS - 1) / 2; // 4,840
const N_M4_ONE: usize = 10 * 5 * N_CORE + 9 * 5 * N_CORE; // 1,520
const N_M5_ONE: usize = 6 * 5 * N_CORE; // 480
const N_M6: usize = 12 * 2 * N_CORE; // 384
const N_M7: usize = 4 * 3 * N_CORE + 3 * 3 * N_CORE + 3 * N_CORE; // 384
pub const EXPECTED_LEN: usize =
    N_M1 + N_M2 + 6 * (N_M3_ONE + N_M4_ONE + N_M5_ONE) + N_M6 + N_M7; // 44,832

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
// 单序列退火：真实顺序与打乱顺序各跑一次，输出 16 指标
// ============================================================================

/// 复用缓冲区，避免每条序列都重新分配
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

#[inline]
fn steps_for(n: usize, cap: usize) -> usize {
    (SWEEPS * n).max(MIN_STEPS).min(cap.max(MIN_STEPS))
}

/// 从升序初猜出发，用阈值接受的随机交换把初猜逐步还原成 target，返回 14 个过程指标（0..13 号）。
/// 交换序列只由 seed 决定，真实与打乱两次调用共用同一 seed（公共随机数，减小差值噪声）。
fn run_anneal(target: &[f32], sorted: &[f32], guess: &mut Vec<f32>, sigma2: f64, m: usize) -> [f64; 14] {
    let n = target.len();
    guess.clear();
    guess.extend_from_slice(sorted);

    let mut s = 0.0f64;
    for k in 0..n {
        let d = guess[k] as f64 - target[k] as f64;
        s += d * d;
    }
    let inv_denom = 1.0 / (2.0 * sigma2 * n as f64);
    let r0 = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
    let hl_target = (1.0 + r0) * 0.5;
    let s_tol = (sigma2 * n as f64 * 1e-10).max(1e-12);
    let inv_m = 1.0 / (m as f64 - 1.0).max(1.0);

    let mut rng = XorShift64::new(SEED);
    let (mut prev_r, mut peak_r, mut max_dd) = (r0, r0, 0.0f64);
    let (mut a1, mut a2, mut a3) = (m, m, m);
    let mut inertia_sum = 0.0f64;
    let mut declines = 0u32;
    let (mut dr_s1, mut dr_s2) = (0.0f64, 0.0f64);
    let s1_end = m / 3;
    let s2_end = 2 * (m / 3);
    let (mut h_s1, mut h_s2) = (0.0f64, 0.0f64);
    let mut mid_decl = 0u32;
    let (mut c_s1, mut c_s2) = (0.0f64, 0.0f64);
    let mut r_at_s2 = r0;
    let mut actual = m;

    // 提速说明：接受与否用条件选择代替分支（随机接受的分支几乎无法预测）；
    // 拒绝的一步 Δr=0，对累计量是加 0，结果与逐步分支写法逐比特相同。
    for t in 0..m {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = (i + 1) % n;
        }
        let gi = guess[i];
        let gj = guess[j];
        let ds = 2.0f64 * (gi as f64 - gj as f64) * (target[i] as f64 - target[j] as f64);
        let ct = sigma2 * (1.0 - t as f64 * inv_m).max(0.0);
        let acc = ds < ct;
        s += if acc { ds } else { 0.0 };
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
                mid_decl += dec;
            } else {
                c_s1 += dv;
                c_s2 += dv2;
            }
        }
        if a1 == m && cr >= hl_target {
            a1 = t;
        }
        if a2 == m && cr >= 0.80 {
            a2 = t;
        }
        if a3 == m && cr >= 0.90 {
            a3 = t;
        }
        peak_r = peak_r.max(cr);
        max_dd = max_dd.max(peak_r - cr);
        prev_r = cr;
        inertia_sum += 1.0 - cr;
        if t + 1 == s2_end {
            r_at_s2 = cr;
        }
        if s <= s_tol {
            if t + 1 < s2_end {
                r_at_s2 = cr;
            }
            actual = t + 1;
            break;
        }
    }
    let final_r = prev_r;
    let dr_cnt = (actual - 1) as u32;
    let h_cnt = actual.min(s1_end).saturating_sub(1) as u32;
    let mid_cnt = actual.min(s2_end).saturating_sub(s1_end) as u32;
    let c_cnt = actual.saturating_sub(s2_end) as u32;

    let sd = |c: u32, s1: f64, s2: f64| -> f64 {
        if c >= 2 {
            let nd = c as f64;
            ((s2 - s1 * s1 / nd) / (nd - 1.0)).max(0.0).sqrt()
        } else {
            0.0
        }
    };
    let hot = sd(h_cnt, h_s1, h_s2);
    let cold = sd(c_cnt, c_s1, c_s2);
    let mf = m as f64;
    let act = actual.max(1) as f64;
    [
        r0,
        final_r,
        final_r - r0.abs(),
        // 初猜已几乎完全还原（序列本身单调递增）时效率按 1 计，避免除以接近 0 的数
        if 1.0 - r0 > 1e-3 { (final_r - r0) / (1.0 - r0) } else { 1.0 },
        a1 as f64 / mf,
        a2 as f64 / mf,
        a3 as f64 / mf,
        inertia_sum / act,
        declines as f64 / act,
        max_dd,
        sd(dr_cnt, dr_s1, dr_s2),
        if mid_cnt > 0 { mid_decl as f64 / mid_cnt as f64 } else { 0.0 },
        final_r - r_at_s2,
        // 有界占比：前段与后段都没有波动时记 0.5
        if hot + cold > 0.0 { hot / (hot + cold) } else { 0.5 },
    ]
}

/// 订单量的时间趋势与雷同度：n 小时穷举全部 (先, 后) 对，n 大时抽 MAX_PROBE_PAIRS 对。
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

/// 一条订单量序列 → 16 指标（真实顺序与打乱顺序各退火一次取差）
pub fn anneal_excess16(seq: &[f32], m: usize, buf: &mut AnnealBuf) -> [f32; N_CORE] {
    let mut out = [f32::NAN; N_CORE];
    let n = seq.len();
    if n < 3 {
        return out;
    }
    let (later, tie) = probe_order_pairs(seq);
    out[14] = later;
    out[15] = tie;

    let mut sum = 0.0f64;
    for &v in seq {
        if !v.is_finite() {
            return [f32::NAN; N_CORE];
        }
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let sigma2 = seq.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    if sigma2 <= 1e-20 {
        return out; // 全部订单量相同：顺序无从谈起，退火指标为空，探测指标照常输出
    }

    buf.sorted.clear();
    buf.sorted.extend_from_slice(seq);
    buf.sorted.sort_by(|a, b| a.total_cmp(b));

    // 打乱对照：同一批数值，Fisher-Yates 固定种子洗牌
    buf.null_target.clear();
    buf.null_target.extend_from_slice(seq);
    let mut rng = XorShift64::new(SEED_NULL);
    for i in (1..n).rev() {
        let j = rng.next_index(i + 1);
        buf.null_target.swap(i, j);
    }

    let real = run_anneal(seq, &buf.sorted, &mut buf.guess, sigma2, m);
    let null = run_anneal(&buf.null_target, &buf.sorted, &mut buf.guess, sigma2, m);
    out[0] = real[0] as f32;
    for k in 1..14 {
        out[k] = (real[k] - null[k]) as f32;
    }
    out
}

// ============================================================================
// 订单聚合：全天一次，每个订单归属首次成交时刻
// ============================================================================

/// 9 条按成交先后排好的订单通道：
/// 0 bid, 1 ask, 2 mixed, 3 actbid, 4 actask, 5 pasbid, 6 pasask, 7 act(=3∪4), 8 pas(=5∪6)
struct OrderChannels {
    vol: [Vec<f32>; 9],
    t_us: [Vec<i64>; 9],
}

impl OrderChannels {
    /// 通道 ch 中首次成交时刻落在 [lo_us, hi_us) 的订单量切片
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

/// (first_idx, side, vol, t) 序列按 (first_idx, side) 两路归并
fn merge_by_idx(a: &[(u32, u8, f32, i64)], b: &[(u32, u8, f32, i64)]) -> (Vec<f32>, Vec<i64>) {
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
    // 订单号只做查找不遍历：插入顺序即首次成交顺序，结果确定
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
    let to_tuple = |v: &[Agg], side: u8, pick: &dyn Fn(&Agg) -> bool| -> Vec<(u32, u8, f32, i64)> {
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
    let split = |x: &[(u32, u8, f32, i64)]| -> (Vec<f32>, Vec<i64>) {
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

/// 两列共同有效行上的皮尔逊相关；有效行 < 5 为空，某侧无波动记 0（与 features 降维口径一致）
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
// 时间格矩阵：每格 5 通道 × 16 指标，以及 10 个行情状态
// ============================================================================

struct BinMatrix {
    n_rows: usize,
    /// 列优先：cols[c][row]
    cols: Vec<Vec<f32>>,
    /// 10 个行情状态，顺序同 REGIME10_NAMES
    states: [Vec<f32>; 10],
    /// 每格最高价、最低价、成交量（事件锚点用）
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

    // 成交按时间有序：边界一次性二分
    let bounds: Vec<usize> = (0..=n_rows)
        .map(|b| {
            let edge = t0 + ((b as i64 * width_sec).min(DAY_SEC)) * US;
            trades.partition_point(|t| t.time_us < edge)
        })
        .collect();

    for r in 0..n_rows {
        let (lo, hi) = (bounds[r], bounds[r + 1]);
        if hi <= lo + 2 {
            continue; // 少于 3 笔成交：本格留空
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
        // 收益用「本格收盘 / 上一有成交格收盘」，首格用本格开盘
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
        for (c, &ch) in MAT_CH_IDX.iter().enumerate() {
            let seq = oc.slice(ch, lo_us, hi_us);
            let v = anneal_excess16(seq, steps_for(seq.len(), CAP_BIN), buf);
            for k in 0..N_CORE {
                cols[c * N_CORE + k][r] = v[k];
            }
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

/// 取从第 start 行到末尾的子矩阵（分段用；行情状态的高低 20% 在段内重新划分）
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

/// M3：矩阵降维（21 统计量 + 两两相关）
fn reduce_matrix(bm: &BinMatrix, out: &mut Vec<f32>) {
    let mut mat = Array2::<f32>::from_elem((bm.n_rows, N_MAT_COLS), f32::NAN);
    for c in 0..N_MAT_COLS {
        for r in 0..bm.n_rows {
            mat[[r, c]] = bm.cols[c][r];
        }
    }
    let vals = features::get_features_factors_rust_values_only(&mat.view(), false);
    // 全空矩阵时降维返回空，按固定长度补空值
    if vals.len() == N_M3_ONE {
        out.extend_from_slice(&vals);
    } else {
        out.extend(std::iter::repeat(f32::NAN).take(N_M3_ONE));
    }
}

/// M4：10 种行情状态 × 5 通道 × 16 指标：高低组差 +（除急拉急跌外）状态相关
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
        for c in 0..N_MAT_COLS {
            let col = &bm.cols[c];
            out.push(sdiff(mean_mask(col, &hi), mean_mask(col, &lo)));
            if ri != 1 {
                out.push(corr_finite(col, st));
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
        for c in 0..N_MAT_COLS {
            let col = &bm.cols[c];
            out.push(sdiff(mean_mask(col, &q[a]), mean_mask(col, &q[b])));
        }
    }
}

// ============================================================================
// 主计算
// ============================================================================

pub fn compute_anneal_regime_excess_from_records(trades: &[TradeRecord]) -> Vec<f32> {
    if trades.len() < 10 {
        return vec![f32::NAN; EXPECTED_LEN];
    }
    let mut out = Vec::with_capacity(EXPECTED_LEN);
    let mut buf = AnnealBuf::new();

    // time_us 已含 +8h 与下午前移，按当地日界取 09:30:00
    let day_us = 86_400 * US;
    let first = trades[0].time_us;
    let t0 = first - first.rem_euclid(day_us) + (9 * 3600 + 30 * 60) * US;
    let at = |sec: i64| t0 + sec * US;

    let oc = build_order_channels(trades);

    // ---- M1：12 时段 × 7 方向 ----
    let mut win = [[[f32::NAN; N_CORE]; 7]; 12];
    for (w, &(lo, hi)) in FINE_WIN_BOUNDS.iter().enumerate() {
        for s in 0..7 {
            let seq = oc.slice(s, at(lo), at(hi));
            win[w][s] = anneal_excess16(seq, steps_for(seq.len(), CAP_WIN), &mut buf);
            out.extend_from_slice(&win[w][s]);
        }
    }

    // ---- M2：时段对比（只做差）----
    for &(a, b, _) in &WIN_CONTRAST_PAIRS {
        for s in 0..7 {
            for k in 0..N_CORE {
                out.push(sdiff(win[a][s][k], win[b][s][k]));
            }
        }
    }

    // ---- M3/M4/M5：1 分钟与 30 秒两种分辨率 ----
    let mut mats: Vec<BinMatrix> = Vec::with_capacity(2);
    for &(_, width, n_rows) in &RESOLUTIONS {
        mats.push(build_bin_matrix(trades, &oc, t0, width, n_rows, &mut buf));
    }
    // 分段：段序在外、分辨率在内，与名字生成顺序一致
    let mut seg_mats: Vec<BinMatrix> = Vec::with_capacity(6);
    for &(_, start_sec) in &SEGMENTS {
        for (ri, &(_, width, _)) in RESOLUTIONS.iter().enumerate() {
            seg_mats.push(tail_rows(&mats[ri], (start_sec / width) as usize));
        }
    }
    for bm in &seg_mats {
        reduce_matrix(bm, &mut out);
    }
    for bm in &seg_mats {
        regime_block(bm, &mut out);
    }
    for bm in &seg_mats {
        quad_block(bm, &mut out);
    }

    // ---- M6：事件锚点（只在第 15~221 分钟里找，保证前后 15 分钟完整）----
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
    let mut evt = [[[f32::NAN; N_CORE]; 2]; 6];
    for (e, c) in centers.iter().enumerate() {
        if let Some(c) = *c {
            let c = c as i64;
            let wins = [(c - 15, c), (c + 1, c + 16)];
            for (sub, &(lo_m, hi_m)) in wins.iter().enumerate() {
                for d in 0..2 {
                    let seq = oc.slice(d, at(lo_m * 60), at(hi_m * 60));
                    evt[e * 2 + sub][d] = anneal_excess16(seq, steps_for(seq.len(), CAP_WIN), &mut buf);
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
            for k in 0..N_CORE {
                out.push(sdiff(evt[a][d][k], evt[b][d][k]));
            }
        }
    }

    // ---- M7：跨尺度 30s / 1m / 5m / 15m / 30m / 全天，方向 bid/ask/mixed ----
    let mut scale = [[[f32::NAN; N_CORE]; 3]; 6];
    for (si, bm) in [(0usize, &mats[1]), (1usize, &mats[0])] {
        for d in 0..3 {
            for k in 0..N_CORE {
                scale[si][d][k] = mean_finite(&bm.cols[d * N_CORE + k]);
            }
        }
    }
    for (si, width) in [(2usize, 300i64), (3, 900), (4, 1800)] {
        let n_blocks = ((DAY_SEC + width - 1) / width) as usize; // 末段不足长度也保留
        let mut acc = [[0.0f64; N_CORE]; 3];
        let mut cnt = [[0usize; N_CORE]; 3];
        for b in 0..n_blocks as i64 {
            let (lo, hi) = (b * width, ((b + 1) * width).min(DAY_SEC));
            for d in 0..3 {
                let seq = oc.slice(d, at(lo), at(hi));
                let v = anneal_excess16(seq, steps_for(seq.len(), CAP_BLOCK), &mut buf);
                for k in 0..N_CORE {
                    if v[k].is_finite() {
                        acc[d][k] += v[k] as f64;
                        cnt[d][k] += 1;
                    }
                }
            }
        }
        for d in 0..3 {
            for k in 0..N_CORE {
                if cnt[d][k] > 0 {
                    scale[si][d][k] = (acc[d][k] / cnt[d][k] as f64) as f32;
                }
            }
        }
    }
    for d in 0..3 {
        let seq = oc.slice(d, at(0), at(DAY_SEC));
        scale[5][d] = anneal_excess16(seq, steps_for(seq.len(), CAP_DAY), &mut buf);
    }
    for si in 2..6 {
        for d in 0..3 {
            out.extend_from_slice(&scale[si][d]);
        }
    }
    for &(a, b) in &[(1usize, 3usize), (2, 5), (0, 2)] {
        for d in 0..3 {
            for k in 0..N_CORE {
                out.push(sdiff(scale[a][d][k], scale[b][d][k]));
            }
        }
    }
    let xs: [f64; 6] = [30f64.ln(), 60f64.ln(), 300f64.ln(), 900f64.ln(), 1800f64.ln(), (DAY_SEC as f64).ln()];
    for d in 0..3 {
        for k in 0..N_CORE {
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

fn mat_col_names(res: &str) -> Vec<String> {
    let mut v = Vec::with_capacity(N_MAT_COLS);
    for ch in &MAT_CH_NAMES {
        for m in &CORE_NAMES {
            v.push(format!("s2_{}_{}_{}", res, ch, m));
        }
    }
    v
}

pub fn anneal_regime_excess_names() -> Vec<String> {
    let mut names = Vec::with_capacity(EXPECTED_LEN);
    for w in &FINE_WIN_NAMES {
        for s in &SIDE7_NAMES {
            for m in &CORE_NAMES {
                names.push(format!("s2_{}_{}_{}", w, s, m));
            }
        }
    }
    for &(_, _, p) in &WIN_CONTRAST_PAIRS {
        for s in &SIDE7_NAMES {
            for m in &CORE_NAMES {
                names.push(format!("s2_wdiff_{}_{}_{}", p, s, m));
            }
        }
    }
    let tags: Vec<String> = SEGMENTS
        .iter()
        .flat_map(|&(seg, _)| RESOLUTIONS.iter().map(move |&(res, _, _)| format!("{}{}", seg, res)))
        .collect();
    for res in &tags {
        let dummy = Array2::<f32>::ones((3, N_MAT_COLS));
        let (_, red) = features::get_features_factors_rust_full(&dummy.view(), &mat_col_names(res), false);
        assert_eq!(red.len(), N_M3_ONE);
        names.extend(red);
    }
    for res in &tags {
        for (ri, r) in REGIME10_NAMES.iter().enumerate() {
            for ch in &MAT_CH_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_cond_{}_{}_{}_{}_hi_lo_diff", res, r, ch, m));
                    if ri != 1 {
                        names.push(format!("s2_cond_{}_{}_{}_{}_state_corr", res, r, ch, m));
                    }
                }
            }
        }
    }
    for res in &tags {
        for q in &QUAD_PAIR_NAMES {
            for ch in &MAT_CH_NAMES {
                for m in &CORE_NAMES {
                    names.push(format!("s2_quad_{}_{}_{}_{}_diff", res, q, ch, m));
                }
            }
        }
    }
    for ew in &["pre_high15", "post_high15", "pre_low15", "post_low15", "pre_maxvol15", "post_maxvol15"] {
        for d in &["bid", "ask"] {
            for m in &CORE_NAMES {
                names.push(format!("s2_evt_{}_{}_{}", ew, d, m));
            }
        }
    }
    for ed in &[
        "post_vs_pre_high", "post_vs_pre_low", "post_vs_pre_maxvol",
        "pre_high_vs_pre_low", "post_high_vs_post_low", "pre_maxvol_vs_pre_high",
    ] {
        for d in &["bid", "ask"] {
            for m in &CORE_NAMES {
                names.push(format!("s2_evtdiff_{}_{}_{}", ed, d, m));
            }
        }
    }
    for sc in &["scale5m", "scale15m", "scale30m", "scalefullday"] {
        for d in &["bid", "ask", "mixed"] {
            for m in &CORE_NAMES {
                names.push(format!("s2_{}_{}_{}", sc, d, m));
            }
        }
    }
    for p in &["scale1m_vs_15m", "scale5m_vs_fullday", "scale30s_vs_5m"] {
        for d in &["bid", "ask", "mixed"] {
            for m in &CORE_NAMES {
                names.push(format!("s2_{}_{}_{}_diff", p, d, m));
            }
        }
    }
    for d in &["bid", "ask", "mixed"] {
        for m in &CORE_NAMES {
            names.push(format!("s2_scale_slope_{}_{}", d, m));
        }
    }
    assert_eq!(names.len(), EXPECTED_LEN);
    names
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

pub fn compute_anneal_regime_excess_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let trades = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    Ok(compute_anneal_regime_excess_from_records(&trades))
}

/// 单股单日计算（读盘），返回 44832 个因子值
#[pyfunction]
pub fn py_anneal_regime_excess(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_regime_excess_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_anneal_regime_excess_names() -> Vec<String> {
    anneal_regime_excess_names()
}

/// 传入逐笔数组计算（与读盘版共用同一内核）
#[pyfunction]
pub fn py_anneal_regime_excess_from_data(
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
    Ok(compute_anneal_regime_excess_from_records(&trades))
}

/// 调试：单条序列给定步数，返回 (真实 14 指标, 打乱 14 指标, 最终 16 指标)
#[pyfunction]
pub fn py_anneal_seq_real_null_debug(seq: Vec<f32>, steps: usize) -> (Vec<f64>, Vec<f64>, Vec<f32>) {
    let mut buf = AnnealBuf::new();
    let out = anneal_excess16(&seq, steps, &mut buf);
    let n = seq.len();
    if n < 3 {
        return (vec![], vec![], out.to_vec());
    }
    let mean = seq.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let sigma2 = seq.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    if sigma2 <= 1e-20 {
        return (vec![], vec![], out.to_vec());
    }
    let mut sorted = seq.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut nt = seq.clone();
    let mut rng = XorShift64::new(SEED_NULL);
    for i in (1..n).rev() {
        let j = rng.next_index(i + 1);
        nt.swap(i, j);
    }
    let mut g = Vec::new();
    let real = run_anneal(&seq, &sorted, &mut g, sigma2, steps);
    let null = run_anneal(&nt, &sorted, &mut g, sigma2, steps);
    (real.to_vec(), null.to_vec(), out.to_vec())
}
