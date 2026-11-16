//! 柜员游戏扩展系列1：时空尺度与行情条件可还原性差异（anneal_regime_diff）。
//!
//! 核心逻辑：
//!   1. 12 个精细日内解剖时段 × 7 种买卖/主被动方向 × 16 个核心可还原性指标 = 1,344 维
//!   2. 15 组关键时段跃迁对比 × 7 方向 × 2 算子(diff/ratio) × 16 指标 = 3,360 维
//!   3. 237 分钟单分钟可还原性矩阵（112 列）经 get_features_factors_rust_full 降维 = 8,568 维
//!   4. 237 分钟 × 10 种日内行情状态条件切割（涨跌/放缩量/宽窄振幅/主买主卖/集中度等）= 2,160 维
//!   5. 放量涨/放量跌/缩量涨/缩量跌 2D 四象限交叉条件可还原性对比 = 384 维
//!   6. 日内事件锚点窗口（最高价/最低价/最大成交量脉冲前后 15 分钟）可还原性突变 = 384 维
//!   7. 跨时间尺度（1m / 5m / 15m / 30m / 全天）可还原性缩放比率与斜率 = 316 维
//!
//! 合计固定输出：16,516 维。
//! 全程确定性：固定种子 xorshift64 + 时间序排序 + 单线程退火，同股同日逐比特一致。

use crate::fast_csv_reader::{build_trade_record, read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// ============================================================================
// 常量定义
// ============================================================================

/// 核心可还原性指标数（16 维）
pub const N_CORE: usize = 16;

/// 16 个核心退火可还原性指标名
pub const CORE_NAMES: [&str; N_CORE] = [
    "r0_init",           // 0: 初始升序相关系数 r0
    "final_r",           // 1: 最终还原相关系数 r_final
    "net_r_gain",        // 2: 净还原增益 final_r - |r0|
    "rel_r_eff",         // 3: 相对还原效率 (final_r - r0) / (1 - r0 + 1e-6)
    "half_life_norm",    // 4: 归一化半衰期步数
    "steps_r80_norm",    // 5: 归一化到达 r=0.80 步数
    "steps_r90_norm",    // 6: 归一化到达 r=0.90 步数
    "inertia_norm",      // 7: 归一化惯性面积 mean(1 - r_t)
    "decline_ratio",     // 8: 退火恶化步数占比
    "max_drawdown",      // 9: 退火过程最大回撤
    "dr_std",            // 10: 单步 Δr 标准差
    "mid_deter_ratio",   // 11: 中段(1/3~2/3)恶化步数占比
    "cold_gain",         // 12: 冷期(后1/3)净增益
    "hot_cold_std_ratio",// 13: 热期与冷期 Δr 标准差之比
    "posD_ratio",        // 14: 探测对真值差 > 0 比例
    "sign_flip_prob",    // 15: 探测对真值差符号翻转概率
];

/// 12 个精细日内解剖时段名称
pub const FINE_WIN_NAMES: [&str; 12] = [
    "w0_open15",       // 09:30-09:45 隔夜信息宣泄期
    "w1_early15",      // 09:45-10:00 早盘首波分歧期
    "w2_morn_trend",   // 10:00-10:30 早盘机构建仓期
    "w3_morn_mid",     // 10:30-11:00 上午盘中平稳期
    "w4_pre_lunch",    // 11:00-11:30 午休前避险/偷袭期
    "w5_post_lunch",   // 13:00-13:30 午后情绪重启期
    "w6_aft_early",    // 13:30-14:00 下午前段
    "w7_aft_mid",      // 14:00-14:30 下午中段潜伏期
    "w8_late_warm",    // 14:30-14:45 尾盘抢筹预热期
    "w9_close12",      // 14:45-14:57 收盘前最后决战期
    "w10_morn2h",      // 09:30-11:30 整个上午
    "w11_aft2h",       // 13:00-14:57 整个下午
];

/// 12 个精细日内解剖时段的秒数边界（相对于 09:30:00，下午已前移 90 分钟）
pub const FINE_WIN_BOUNDS: [(f32, f32); 12] = [
    (0.0, 900.0),         // 09:30-09:45
    (900.0, 1_800.0),     // 09:45-10:00
    (1_800.0, 3_600.0),   // 10:00-10:30
    (3_600.0, 5_400.0),   // 10:30-11:00
    (5_400.0, 7_200.0),   // 11:00-11:30
    (7_200.0, 9_000.0),   // 13:00-13:30 (前移后 11:30-12:00)
    (9_000.0, 10_800.0),  // 13:30-14:00
    (10_800.0, 12_600.0), // 14:00-14:30
    (12_600.0, 13_500.0), // 14:30-14:45
    (13_500.0, 14_220.0), // 14:45-14:57
    (0.0, 7_200.0),       // 上午 2h
    (7_200.0, 14_220.0),  // 下午 2h
];

/// 15 组关键跨时段对比对 (win_a, win_b, pair_name)
pub const WIN_CONTRAST_PAIRS: [(usize, usize, &str); 15] = [
    (9, 0, "close12_vs_open15"),      // 收盘决战 vs 开盘宣泄
    (1, 0, "early15_vs_open15"),      // 早盘后段 vs 早盘前段
    (2, 0, "morntrend_vs_open15"),    // 早盘建仓 vs 开盘宣泄
    (4, 2, "prelunch_vs_morntrend"),  // 午休前 vs 早盘建仓
    (5, 4, "postlunch_vs_prelunch"),  // 午休后重启 vs 午休前
    (8, 7, "latewarm_vs_aftmid"),     // 尾盘预热 vs 下午平淡
    (9, 8, "close12_vs_latewarm"),    // 收盘决战 vs 尾盘预热
    (9, 7, "close12_vs_aftmid"),      // 收盘决战 vs 下午平淡
    (11, 10, "aft2h_vs_morn2h"),      // 下午整体 vs 上午整体
    (8, 1, "latewarm_vs_early15"),    // 尾盘预热 vs 早盘分歧
    (5, 0, "postlunch_vs_open15"),    // 下午开盘 vs 上午开盘
    (9, 4, "close12_vs_prelunch"),    // 下午收盘 vs 上午收盘
    (3, 2, "mornmid_vs_morntrend"),   // 上午平稳 vs 早盘趋势
    (7, 5, "aftmid_vs_postlunch"),    // 下午中段 vs 午后重启
    (6, 3, "aftearly_vs_mornmid"),    // 下午前段 vs 上午前段
];

/// 7 种核心方向名称
pub const SIDE7_NAMES: [&str; 7] = [
    "bid", "ask", "mixed", "actbid", "actask", "pasbid", "pasask",
];

/// 逐分钟 7 个方向/差值通道名
pub const MIN_CH_NAMES: [&str; 7] = [
    "bid",
    "ask",
    "mixed",
    "act",
    "pas",
    "bid_minus_ask",
    "act_minus_pas",
];

/// 10 种日内分钟行情状态名称
pub const REGIME10_NAMES: [&str; 10] = [
    "ret_sign",       // 0: 上涨分钟(ret>0) vs 下跌分钟(ret<0)
    "ret_extreme",    // 1: 急拉分钟(Top20%) vs 急跌分钟(Bot20%)
    "ret_abs_move",   // 2: 大幅变盘分钟(|ret| Top20%) vs 横盘分钟(|ret| Bot20%)
    "vol_level",      // 3: 放量分钟(Top20%) vs 缩量分钟(Bot20%)
    "cnt_freq",       // 4: 高频笔数分钟(Top20%) vs 冷清分钟(Bot20%)
    "price_range",    // 5: 宽振幅博弈分钟(Top20%) vs 窄振幅分钟(Bot20%)
    "act_buy_ratio",  // 6: 主买强攻分钟(Top20%) vs 主卖砸盘分钟(Bot20%)
    "net_inflow",     // 7: 净流入最大分钟(Top20%) vs 净流出最大分钟(Bot20%)
    "order_hhi",      // 8: 大单集中主导分钟(Top20%) vs 散户分散分钟(Bot20%)
    "mean_ord_size",  // 9: 均单规模最大分钟(Top20%) vs 均单最小分钟(Bot20%)
];

/// 6 个二维四象限对比对名称
pub const QUAD_PAIR_NAMES: [&str; 6] = [
    "hv_up_vs_hv_dn", // 放量上涨 vs 放量下跌
    "lv_up_vs_lv_dn", // 缩量上涨 vs 缩量下跌
    "hv_up_vs_lv_up", // 放量上涨 vs 缩量上涨
    "hv_dn_vs_lv_dn", // 放量下跌 vs 缩量下跌
    "hv_up_vs_lv_dn", // 放量上涨 vs 缩量下跌
    "hv_dn_vs_lv_up", // 放量下跌 vs 缩量上涨
];

pub const N_MINUTES: usize = 237;
pub const N_MIN_COLS: usize = 7 * N_CORE; // 112
pub const N_PART1: usize = 12 * 7 * N_CORE; // 1,344
pub const N_PART2: usize = 15 * 7 * 2 * N_CORE; // 3,360
pub const N_PART3: usize = 21 * N_MIN_COLS + N_MIN_COLS * (N_MIN_COLS - 1) / 2; // 2352 + 6216 = 8,568
pub const N_PART4: usize = 10 * 5 * N_CORE * 3 - 240; // 10状态 × 4核心通道(bid/ask/mixed/bid_minus_ask) × 16指标 × 3(diff/ratio/corr) + 10状态 × act_minus_pas × 16指标 × 1.5 = 2,160
pub const N_PART5: usize = 6 * 4 * N_CORE; // 384
pub const N_PART6: usize = 12 * 2 * N_CORE; // 384
pub const N_PART7: usize = 316;

/// 总输出维度：1,344 + 3,360 + 8,568 + 2,160 + 384 + 384 + 316 = 16,516
pub const EXPECTED_LEN: usize =
    N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5 + N_PART6 + N_PART7;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const EPS: f32 = 1e-6;

// ============================================================================
// 确定性随机数生成器 XorShift64
// ============================================================================

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

// ============================================================================
// 核心退火计算（16 维归一化可还原性指标）
// ============================================================================

pub struct FastAnnealBuf {
    pub guess: Vec<f32>,
    pub gs: Vec<(f32, f32)>,
}

impl FastAnnealBuf {
    pub fn new() -> Self {
        Self {
            guess: Vec::with_capacity(4096),
            gs: Vec::with_capacity(4096),
        }
    }
}

/// 自适应退火步数：保证不同长度序列都能充分退火，同时通过除以 m_max 归一化消除步数尺度影响
#[inline]
pub fn adaptive_steps(n: usize, cap: usize) -> usize {
    (n.saturating_mul(n).saturating_mul(6)).max(600).min(cap)
}

/// 对任意实数序列运行确定性模拟退火（初始先验为升序），返回 16 个核心可还原性指标。
pub fn anneal_core16(seq: &[f32], m_max: usize, buf: &mut FastAnnealBuf) -> [f32; N_CORE] {
    anneal_core16_with_mode(seq, m_max, 0, buf)
}

/// 支持多种初始猜测先验模式：
/// mode=0: 升序 (Ascending)
/// mode=1: 降序 (Descending)
/// mode=2: U型 (两头大中间小)
/// mode=3: 倒U型 (中间大两头小)
pub fn anneal_core16_with_mode(
    seq: &[f32],
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
    buf.guess.extend_from_slice(seq);
    buf.guess.sort_by(|a, b| a.total_cmp(b));
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

    for t in 0..m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = if i + 1 == n { 0 } else { i + 1 };
        }

        let gi = buf.guess[i];
        let gj = buf.guess[j];
        if d_cnt < 2000 && gi != gj {
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

        let ti = seq[i];
        let tj = seq[j];
        let ds = 2.0f32 * (gi - gj) * (ti - tj);
        let accepted = (ds as f64) < 0.0
            || (ds as f64) < sigma2 * (1.0 - t as f64 * inv_m).max(0.0);
        if accepted {
            s += ds as f64;
            buf.guess.swap(i, j);
        }

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
// 订单聚合与 7 方向序列提取
// ============================================================================

#[derive(Clone, Copy)]
struct SimpleAggOrder {
    volume: f32,
    n_act: u32,
    n_pas: u32,
    first_idx: usize,
}

fn aggregate_orders_slice(
    trades: &[TradeRecord],
) -> (
    FxHashMap<i64, SimpleAggOrder>,
    FxHashMap<i64, SimpleAggOrder>,
) {
    let mut bid_map: FxHashMap<i64, SimpleAggOrder> =
        FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());
    let mut ask_map: FxHashMap<i64, SimpleAggOrder> =
        FxHashMap::with_capacity_and_hasher(trades.len() / 2 + 1, Default::default());

    for (i, t) in trades.iter().enumerate() {
        let be = bid_map.entry(t.bid_order).or_insert(SimpleAggOrder {
            volume: 0.0,
            n_act: 0,
            n_pas: 0,
            first_idx: i,
        });
        be.volume += t.volume;
        if t.flag == 66 {
            be.n_act += 1;
        } else if t.flag == 83 {
            be.n_pas += 1;
        }

        let ae = ask_map.entry(t.ask_order).or_insert(SimpleAggOrder {
            volume: 0.0,
            n_act: 0,
            n_pas: 0,
            first_idx: i,
        });
        ae.volume += t.volume;
        if t.flag == 83 {
            ae.n_act += 1;
        } else if t.flag == 66 {
            ae.n_pas += 1;
        }
    }
    (bid_map, ask_map)
}

/// 按 first_idx 严格排序提取 7 种方向序列：
/// 0=bid, 1=ask, 2=mixed, 3=actbid, 4=actask, 5=pasbid, 6=pasask
fn extract_7_sides(
    bid_map: &FxHashMap<i64, SimpleAggOrder>,
    ask_map: &FxHashMap<i64, SimpleAggOrder>,
) -> [Vec<f32>; 7] {
    let mut bid_v: Vec<(usize, u8, f32, bool, bool)> = bid_map
        .values()
        .map(|e| (e.first_idx, 0u8, e.volume, e.n_act > 0 && e.n_pas == 0, e.n_pas > 0 && e.n_act == 0))
        .collect();
    bid_v.sort_by_key(|x| (x.0, x.1));

    let mut ask_v: Vec<(usize, u8, f32, bool, bool)> = ask_map
        .values()
        .map(|e| (e.first_idx, 1u8, e.volume, e.n_act > 0 && e.n_pas == 0, e.n_pas > 0 && e.n_act == 0))
        .collect();
    ask_v.sort_by_key(|x| (x.0, x.1));

    let s_bid: Vec<f32> = bid_v.iter().map(|x| x.2).collect();
    let s_ask: Vec<f32> = ask_v.iter().map(|x| x.2).collect();
    let s_actbid: Vec<f32> = bid_v.iter().filter(|x| x.3).map(|x| x.2).collect();
    let s_actask: Vec<f32> = ask_v.iter().filter(|x| x.3).map(|x| x.2).collect();
    let s_pasbid: Vec<f32> = bid_v.iter().filter(|x| x.4).map(|x| x.2).collect();
    let s_pasask: Vec<f32> = ask_v.iter().filter(|x| x.4).map(|x| x.2).collect();

    let mut mix_v = Vec::with_capacity(bid_v.len() + ask_v.len());
    mix_v.extend(bid_v.iter().map(|x| (x.0, x.1, x.2)));
    mix_v.extend(ask_v.iter().map(|x| (x.0, x.1, x.2)));
    mix_v.sort_by_key(|x| (x.0, x.1));
    let s_mix: Vec<f32> = mix_v.into_iter().map(|x| x.2).collect();

    [s_bid, s_ask, s_mix, s_actbid, s_actask, s_pasbid, s_pasask]
}

/// 提取分钟级 5 个基础方向序列：0=bid, 1=ask, 2=mixed, 3=act(actbid+actask), 4=pas(pasbid+pasask)
fn extract_5_min_sides(
    bid_map: &FxHashMap<i64, SimpleAggOrder>,
    ask_map: &FxHashMap<i64, SimpleAggOrder>,
) -> [Vec<f32>; 5] {
    let mut bid_v: Vec<(usize, u8, f32, bool, bool)> = bid_map
        .values()
        .map(|e| (e.first_idx, 0u8, e.volume, e.n_act > 0 && e.n_pas == 0, e.n_pas > 0 && e.n_act == 0))
        .collect();
    bid_v.sort_by_key(|x| (x.0, x.1));

    let mut ask_v: Vec<(usize, u8, f32, bool, bool)> = ask_map
        .values()
        .map(|e| (e.first_idx, 1u8, e.volume, e.n_act > 0 && e.n_pas == 0, e.n_pas > 0 && e.n_act == 0))
        .collect();
    ask_v.sort_by_key(|x| (x.0, x.1));

    let s_bid: Vec<f32> = bid_v.iter().map(|x| x.2).collect();
    let s_ask: Vec<f32> = ask_v.iter().map(|x| x.2).collect();

    let mut mix_v = Vec::with_capacity(bid_v.len() + ask_v.len());
    let mut act_v = Vec::new();
    let mut pas_v = Vec::new();
    for x in &bid_v {
        mix_v.push((x.0, x.1, x.2));
        if x.3 {
            act_v.push((x.0, x.1, x.2));
        }
        if x.4 {
            pas_v.push((x.0, x.1, x.2));
        }
    }
    for x in &ask_v {
        mix_v.push((x.0, x.1, x.2));
        if x.3 {
            act_v.push((x.0, x.1, x.2));
        }
        if x.4 {
            pas_v.push((x.0, x.1, x.2));
        }
    }
    mix_v.sort_by_key(|x| (x.0, x.1));
    act_v.sort_by_key(|x| (x.0, x.1));
    pas_v.sort_by_key(|x| (x.0, x.1));

    [
        s_bid,
        s_ask,
        mix_v.into_iter().map(|x| x.2).collect(),
        act_v.into_iter().map(|x| x.2).collect(),
        pas_v.into_iter().map(|x| x.2).collect(),
    ]
}

// ============================================================================
// 辅助统计函数
// ============================================================================

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

#[inline]
fn mean_of_mask(vals: &[f32], mask: &[bool]) -> f32 {
    let mut s = 0.0f64;
    let mut c = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if m && vals[i].is_finite() {
            s += vals[i] as f64;
            c += 1;
        }
    }
    if c == 0 {
        f32::NAN
    } else {
        (s / c as f64) as f32
    }
}

/// 根据分位数选出 Top 20% 与 Bot 20% 布尔掩码
fn top_bot_20_masks(state: &[f32]) -> (Vec<bool>, Vec<bool>) {
    let n = state.len();
    let mut valid: Vec<(f32, usize)> = state
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (v, i))
        .collect();
    valid.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut hi_mask = vec![false; n];
    let mut lo_mask = vec![false; n];
    let m = valid.len();
    if m < 5 {
        return (hi_mask, lo_mask);
    }
    let k = (m / 5).max(1);
    for &(_, idx) in &valid[..k] {
        lo_mask[idx] = true;
    }
    for &(_, idx) in &valid[m - k..] {
        hi_mask[idx] = true;
    }
    (hi_mask, lo_mask)
}

// ============================================================================
// 纯计算核心：compute_anneal_regime_diff_from_records
// ============================================================================

pub fn compute_anneal_regime_diff_from_records(trades: &[TradeRecord]) -> Vec<f32> {
    if trades.len() < 10 {
        return vec![f32::NAN; EXPECTED_LEN];
    }

    let mut out = Vec::with_capacity(EXPECTED_LEN);
    let mut buf = FastAnnealBuf::new();

    // 以首笔成交所在交易日 09:30:00 为基准起点（防首笔晚于 09:30 偏移）
    let first_sec = trades[0].time_sec;
    let day_base = ((first_sec as i64) - ((first_sec as i64).rem_euclid(86400))) as f32;
    let t_open = day_base + 9.0 * 3600.0 + 30.0 * 60.0;
    let t_base = if (first_sec - t_open).abs() < 7200.0 {
        t_open
    } else {
        first_sec
    };

    // ------------------------------------------------------------------------
    // 模块 1：12 个精细日内解剖时段 × 7 方向 × 16 指标 = 1,344 维
    // ------------------------------------------------------------------------
    let mut win_metrics = [[[f32::NAN; N_CORE]; 7]; 12];
    for (w_idx, &(sec_lo, sec_hi)) in FINE_WIN_BOUNDS.iter().enumerate() {
        let lo = trades.partition_point(|t| t.time_sec < t_base + sec_lo);
        let hi = trades.partition_point(|t| t.time_sec < t_base + sec_hi);
        if hi > lo + 2 {
            let (bid_map, ask_map) = aggregate_orders_slice(&trades[lo..hi]);
            let sides = extract_7_sides(&bid_map, &ask_map);
            for s_idx in 0..7 {
                let seq = &sides[s_idx];
                let m_steps = adaptive_steps(seq.len(), 20_000);
                win_metrics[w_idx][s_idx] = anneal_core16(seq, m_steps, &mut buf);
            }
        }
        for s_idx in 0..7 {
            out.extend_from_slice(&win_metrics[w_idx][s_idx]);
        }
    }
    debug_assert_eq!(out.len(), N_PART1);

    // ------------------------------------------------------------------------
    // 模块 2：15 组跨时段跃迁对比 × 7 方向 × 2(diff/ratio) × 16 指标 = 3,360 维
    // ------------------------------------------------------------------------
    for &(wa, wb, _) in &WIN_CONTRAST_PAIRS {
        for s_idx in 0..7 {
            for k in 0..N_CORE {
                out.push(safe_diff(win_metrics[wa][s_idx][k], win_metrics[wb][s_idx][k]));
            }
            for k in 0..N_CORE {
                out.push(safe_ratio(win_metrics[wa][s_idx][k], win_metrics[wb][s_idx][k]));
            }
        }
    }
    debug_assert_eq!(out.len(), N_PART1 + N_PART2);

    // ------------------------------------------------------------------------
    // 模块 3：237 分钟单分钟可还原性矩阵（237 × 112）及每分钟市场状态计算
    // ------------------------------------------------------------------------
    let mut min_mat = Array2::<f32>::from_elem((N_MINUTES, N_MIN_COLS), f32::NAN);
    let mut min_ret = vec![f32::NAN; N_MINUTES];
    let mut min_vol = vec![f32::NAN; N_MINUTES];
    let mut min_cnt = vec![f32::NAN; N_MINUTES];
    let mut min_rng = vec![f32::NAN; N_MINUTES];
    let mut min_abr = vec![f32::NAN; N_MINUTES];
    let mut min_net = vec![f32::NAN; N_MINUTES];
    let mut min_hhi = vec![f32::NAN; N_MINUTES];
    let mut min_msz = vec![f32::NAN; N_MINUTES];

    let mut high_min_idx = 0usize;
    let mut high_prc_val = f32::NEG_INFINITY;
    let mut low_min_idx = 0usize;
    let mut low_prc_val = f32::INFINITY;
    let mut max_vol_min_idx = 0usize;
    let mut max_vol_val = -1.0f32;

    for m_idx in 0..N_MINUTES {
        let lo = trades.partition_point(|t| t.time_sec < t_base + m_idx as f32 * 60.0);
        let hi = trades.partition_point(|t| t.time_sec < t_base + (m_idx + 1) as f32 * 60.0);
        if hi <= lo + 2 {
            continue;
        }
        let slice = &trades[lo..hi];
        let p_open = slice.first().unwrap().price;
        let p_close = slice.last().unwrap().price;
        let mut p_high = p_open;
        let mut p_low = p_open;
        let mut v_sum = 0.0f32;
        let mut v_buy = 0.0f32;
        for t in slice {
            if t.price > p_high {
                p_high = t.price;
            }
            if t.price < p_low {
                p_low = t.price;
            }
            v_sum += t.volume;
            if t.flag == 66 {
                v_buy += t.volume;
            }
        }
        if p_high > high_prc_val {
            high_prc_val = p_high;
            high_min_idx = m_idx;
        }
        if p_low < low_prc_val {
            low_prc_val = p_low;
            low_min_idx = m_idx;
        }
        if v_sum > max_vol_val {
            max_vol_val = v_sum;
            max_vol_min_idx = m_idx;
        }

        if p_open > 0.0 {
            min_ret[m_idx] = (p_close - p_open) / p_open;
            min_rng[m_idx] = (p_high - p_low) / p_open;
        }
        min_vol[m_idx] = v_sum;
        min_cnt[m_idx] = slice.len() as f32;
        if v_sum > 0.0 {
            min_abr[m_idx] = v_buy / v_sum;
        }
        min_net[m_idx] = 2.0 * v_buy - v_sum;

        let (bid_map, ask_map) = aggregate_orders_slice(slice);
        let sides5 = extract_5_min_sides(&bid_map, &ask_map);

        // 订单集中度 HHI 与均单规模（基于 mixed 序列）
        let mix_seq = &sides5[2];
        if !mix_seq.is_empty() {
            let tot_m: f32 = mix_seq.iter().sum();
            if tot_m > 0.0 {
                let hhi: f32 = mix_seq.iter().map(|&v| (v / tot_m).powi(2)).sum();
                min_hhi[m_idx] = hhi;
                min_msz[m_idx] = tot_m / mix_seq.len() as f32;
            }
        }

        // 计算 5 个基础方向的 16 指标
        let mut ch_vals = [[f32::NAN; N_CORE]; 5];
        for c in 0..5 {
            let seq = &sides5[c];
            let m_steps = adaptive_steps(seq.len(), 6_000);
            ch_vals[c] = anneal_core16(seq, m_steps, &mut buf);
            for k in 0..N_CORE {
                min_mat[[m_idx, c * N_CORE + k]] = ch_vals[c][k];
            }
        }
        // 通道 5: bid_minus_ask
        for k in 0..N_CORE {
            min_mat[[m_idx, 5 * N_CORE + k]] = safe_diff(ch_vals[0][k], ch_vals[1][k]);
        }
        // 通道 6: act_minus_pas
        for k in 0..N_CORE {
            min_mat[[m_idx, 6 * N_CORE + k]] = safe_diff(ch_vals[3][k], ch_vals[4][k]);
        }
    }

    let reduced = features::get_features_factors_rust_values_only(&min_mat.view(), false);
    out.extend_from_slice(&reduced);
    debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3);

    // ------------------------------------------------------------------------
    // 模块 4：237 分钟 × 10 种日内行情状态条件切割 = 2,160 维
    // 对前 4 个核心通道 (bid=0, ask=1, mixed=2, bid_minus_ask=5) 输出 diff/ratio/corr (10×4×16×3 = 1,920)
    // 对第 5 个通道 (act_minus_pas=6) 输出 diff 与 corr，对前5状态额外输出 ratio (240) → 合计 2,160 维
    // ------------------------------------------------------------------------
    let min_abs_ret: Vec<f32> = min_ret.iter().map(|v| v.abs()).collect();
    let state_vecs: [&[f32]; 10] = [
        &min_ret,     // 0: ret_sign
        &min_ret,     // 1: ret_extreme
        &min_abs_ret, // 2: ret_abs_move
        &min_vol,     // 3: vol_level
        &min_cnt,     // 4: cnt_freq
        &min_rng,     // 5: price_range
        &min_abr,     // 6: act_buy_ratio
        &min_net,     // 7: net_inflow
        &min_hhi,     // 8: order_hhi
        &min_msz,     // 9: mean_ord_size
    ];

    let cond_channels: [usize; 4] = [0, 1, 2, 5]; // bid, ask, mixed, bid_minus_ask
    for (r_idx, &st_vec) in state_vecs.iter().enumerate() {
        let (hi_mask, lo_mask) = if r_idx == 0 {
            // ret_sign: 正收益分钟 vs 负收益分钟
            let hi: Vec<bool> = st_vec.iter().map(|&v| v.is_finite() && v > 0.0).collect();
            let lo: Vec<bool> = st_vec.iter().map(|&v| v.is_finite() && v < 0.0).collect();
            (hi, lo)
        } else {
            top_bot_20_masks(st_vec)
        };

        for &ch in &cond_channels {
            for k in 0..N_CORE {
                let col_idx = ch * N_CORE + k;
                let col: Vec<f32> = (0..N_MINUTES).map(|m| min_mat[[m, col_idx]]).collect();
                let m_hi = mean_of_mask(&col, &hi_mask);
                let m_lo = mean_of_mask(&col, &lo_mask);
                out.push(safe_diff(m_hi, m_lo));
                out.push(safe_ratio(m_hi, m_lo));
                out.push(features::corr_pair(&col, st_vec));
            }
        }
        // 第 5 通道 act_minus_pas (ch=6): 输出 diff（所有 10 状态）+ corr（前 5 状态）
        for k in 0..N_CORE {
            let col_idx = 6 * N_CORE + k;
            let col: Vec<f32> = (0..N_MINUTES).map(|m| min_mat[[m, col_idx]]).collect();
            let m_hi = mean_of_mask(&col, &hi_mask);
            let m_lo = mean_of_mask(&col, &lo_mask);
            out.push(safe_diff(m_hi, m_lo));
            if r_idx < 5 {
                out.push(features::corr_pair(&col, st_vec));
            }
        }
    }
    debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3 + N_PART4);

    // ------------------------------------------------------------------------
    // 模块 5：二维四象限（放量涨 / 放量跌 / 缩量涨 / 缩量跌）交叉对比 = 384 维
    // ------------------------------------------------------------------------
    let vol_med = {
        let mut vv: Vec<f32> = min_vol.iter().copied().filter(|x| x.is_finite()).collect();
        vv.sort_by(|a, b| a.total_cmp(b));
        if vv.is_empty() {
            f32::NAN
        } else {
            vv[vv.len() / 2]
        }
    };
    let mut q_masks = [
        vec![false; N_MINUTES], // 0: 放量涨 (HV_UP)
        vec![false; N_MINUTES], // 1: 放量跌 (HV_DN)
        vec![false; N_MINUTES], // 2: 缩量涨 (LV_UP)
        vec![false; N_MINUTES], // 3: 缩量跌 (LV_DN)
    ];
    if vol_med.is_finite() {
        for m in 0..N_MINUTES {
            let v = min_vol[m];
            let r = min_ret[m];
            if v.is_finite() && r.is_finite() {
                if v >= vol_med && r > 0.0 {
                    q_masks[0][m] = true;
                } else if v >= vol_med && r < 0.0 {
                    q_masks[1][m] = true;
                } else if v < vol_med && r > 0.0 {
                    q_masks[2][m] = true;
                } else if v < vol_med && r < 0.0 {
                    q_masks[3][m] = true;
                }
            }
        }
    }
    let quad_pairs: [(usize, usize); 6] = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
    for &(qa, qb) in &quad_pairs {
        for &ch in &cond_channels {
            for k in 0..N_CORE {
                let col_idx = ch * N_CORE + k;
                let col: Vec<f32> = (0..N_MINUTES).map(|m| min_mat[[m, col_idx]]).collect();
                let ma = mean_of_mask(&col, &q_masks[qa]);
                let mb = mean_of_mask(&col, &q_masks[qb]);
                out.push(safe_diff(ma, mb));
            }
        }
    }
    debug_assert_eq!(out.len(), N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5);

    // ------------------------------------------------------------------------
    // 模块 6：事件触发窗口（最高价/最低价/最大量脉冲前后 15 分钟）可还原性突变 = 384 维
    // 6 个事件子窗口：[pre_high, post_high, pre_low, post_low, pre_maxvol, post_maxvol]
    // ------------------------------------------------------------------------
    let event_centers = [high_min_idx, low_min_idx, max_vol_min_idx];
    let mut evt_metrics = [[[f32::NAN; N_CORE]; 2]; 6]; // 6窗口 × 2方向(bid=0, ask=1)
    for (e_i, &c_min) in event_centers.iter().enumerate() {
        let pre_lo_sec = (c_min.saturating_sub(15) as f32) * 60.0;
        let pre_hi_sec = (c_min as f32) * 60.0;
        let post_lo_sec = ((c_min + 1) as f32) * 60.0;
        let post_hi_sec = ((c_min + 16).min(N_MINUTES) as f32) * 60.0;

        for (sub_i, (s_lo, s_hi)) in [(pre_lo_sec, pre_hi_sec), (post_lo_sec, post_hi_sec)]
            .iter()
            .enumerate()
        {
            let lo = trades.partition_point(|t| t.time_sec < t_base + s_lo);
            let hi = trades.partition_point(|t| t.time_sec < t_base + s_hi);
            if hi > lo + 2 {
                let (bid_map, ask_map) = aggregate_orders_slice(&trades[lo..hi]);
                let sides = extract_5_min_sides(&bid_map, &ask_map);
                for d_i in 0..2 {
                    let seq = &sides[d_i];
                    evt_metrics[e_i * 2 + sub_i][d_i] =
                        anneal_core16(seq, adaptive_steps(seq.len(), 12_000), &mut buf);
                }
            }
        }
    }
    // 6 个事件窗口原始值 (6 × 2 × 16 = 192)
    for w in 0..6 {
        for d_i in 0..2 {
            out.extend_from_slice(&evt_metrics[w][d_i]);
        }
    }
    // 6 组事件前后与顶底对比差值 (6 × 2 × 16 = 192)
    let evt_contrasts: [(usize, usize); 6] = [
        (1, 0), // post_high - pre_high
        (3, 2), // post_low - pre_low
        (5, 4), // post_maxvol - pre_maxvol
        (0, 2), // pre_high - pre_low
        (1, 3), // post_high - post_low
        (4, 0), // pre_maxvol - pre_high
    ];
    for &(ea, eb) in &evt_contrasts {
        for d_i in 0..2 {
            for k in 0..N_CORE {
                out.push(safe_diff(evt_metrics[ea][d_i][k], evt_metrics[eb][d_i][k]));
            }
        }
    }
    debug_assert_eq!(
        out.len(),
        N_PART1 + N_PART2 + N_PART3 + N_PART4 + N_PART5 + N_PART6
    );

    // ------------------------------------------------------------------------
    // 模块 7：跨时间尺度（1m / 5m / 15m / 30m / 全天）可还原性缩放与衰减 = 316 维
    // ------------------------------------------------------------------------
    // 计算 5m (47段)、15m (15段)、30m (8段)、全天 (1段) 在 3 个方向 (bid/ask/mixed) 上的平均可还原性
    let mut scale_means = [[[f32::NAN; N_CORE]; 3]; 5]; // 5尺度(1m,5m,15m,30m,fullday) × 3方向 × 16指标

    // 尺度 0: 1m 均值（直接从 min_mat 前 3 个通道取列均值）
    for d_i in 0..3 {
        for k in 0..N_CORE {
            let col: Vec<f32> = (0..N_MINUTES)
                .map(|m| min_mat[[m, d_i * N_CORE + k]])
                .collect();
            scale_means[0][d_i][k] = features::col_mean(&col);
        }
    }

    // 尺度 1(5m=300s), 尺度 2(15m=900s), 尺度 3(30m=1800s)
    let scale_secs: [(usize, f32, usize); 3] = [(1, 300.0, 47), (2, 900.0, 15), (3, 1800.0, 8)];
    for &(sc_idx, step_sec, n_blocks) in &scale_secs {
        let mut acc = [[0.0f64; N_CORE]; 3];
        let mut cnt = [[0usize; N_CORE]; 3];
        for b in 0..n_blocks {
            let s_lo = b as f32 * step_sec;
            let s_hi = ((b + 1) as f32 * step_sec).min(14_220.0);
            let lo = trades.partition_point(|t| t.time_sec < t_base + s_lo);
            let hi = trades.partition_point(|t| t.time_sec < t_base + s_hi);
            if hi <= lo + 2 {
                continue;
            }
            let (bid_map, ask_map) = aggregate_orders_slice(&trades[lo..hi]);
            let sides5 = extract_5_min_sides(&bid_map, &ask_map);
            for d_i in 0..3 {
                let seq = &sides5[d_i];
                let m_steps = adaptive_steps(seq.len(), 15_000);
                let vals = anneal_core16(seq, m_steps, &mut buf);
                for k in 0..N_CORE {
                    if vals[k].is_finite() {
                        acc[d_i][k] += vals[k] as f64;
                        cnt[d_i][k] += 1;
                    }
                }
            }
        }
        for d_i in 0..3 {
            for k in 0..N_CORE {
                if cnt[d_i][k] > 0 {
                    scale_means[sc_idx][d_i][k] = (acc[d_i][k] / cnt[d_i][k] as f64) as f32;
                }
            }
        }
    }

    // 尺度 4: 全天 (0..14220s)
    {
        let lo = trades.partition_point(|t| t.time_sec < t_base);
        let hi = trades.partition_point(|t| t.time_sec < t_base + 14_220.0);
        if hi > lo + 2 {
            let (bid_map, ask_map) = aggregate_orders_slice(&trades[lo..hi]);
            let sides5 = extract_5_min_sides(&bid_map, &ask_map);
            for d_i in 0..3 {
                let seq = &sides5[d_i];
                scale_means[4][d_i] =
                    anneal_core16(seq, adaptive_steps(seq.len(), 30_000), &mut buf);
            }
        }
    }

    // 7a: 5m / 15m / 30m / 全天 的均值本身 (4 × 3 × 16 = 192 维)
    for sc_idx in 1..5 {
        for d_i in 0..3 {
            out.extend_from_slice(&scale_means[sc_idx][d_i]);
        }
    }
    // 7b: 跨尺度比值与差值 (1m vs 5m, 1m vs 15m, 5m vs 30m, 15m vs fullday)
    // 2 组核心尺度对比 (1m_vs_15m, 5m_vs_fullday) × 3 方向 × 16 指标 = 96 维
    for &(sa, sb) in &[(0usize, 2usize), (1usize, 4usize)] {
        for d_i in 0..3 {
            for k in 0..N_CORE {
                out.push(safe_diff(scale_means[sa][d_i][k], scale_means[sb][d_i][k]));
            }
        }
    }
    // 7c: 5 个尺度上的缩放斜率 (bid / ask 前 14 个核心指标: 2 × 14 = 28 维) -> 192 + 96 + 28 = 316 维
    let log_scales = [0.0f32, 5.0f32.ln(), 15.0f32.ln(), 30.0f32.ln(), 237.0f32.ln()];
    for d_i in 0..2 {
        for k in 0..14 {
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            let mut sxx = 0.0f32;
            let mut sxy = 0.0f32;
            let mut n_v = 0usize;
            for sc in 0..5 {
                let y = scale_means[sc][d_i][k];
                if y.is_finite() {
                    let x = log_scales[sc];
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    sxy += x * y;
                    n_v += 1;
                }
            }
            let slope = if n_v >= 3 {
                let nf = n_v as f32;
                let denom = sxx - sx * sx / nf;
                if denom.abs() > 1e-8 {
                    (sxy - sx * sy / nf) / denom
                } else {
                    0.0
                }
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
// 因子名生成（与 compute_anneal_regime_diff_from_records 严格逐一对齐）
// ============================================================================

fn build_min_col_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_MIN_COLS);
    for ch in &MIN_CH_NAMES {
        for m in &CORE_NAMES {
            names.push(format!("s1_min_{}_{}", ch, m));
        }
    }
    names
}

pub fn anneal_regime_diff_names() -> Vec<String> {
    let mut names = Vec::with_capacity(EXPECTED_LEN);

    // 模块 1: 1,344
    for w in &FINE_WIN_NAMES {
        for s in &SIDE7_NAMES {
            for m in &CORE_NAMES {
                names.push(format!("s1_{}_{}_{}", w, s, m));
            }
        }
    }

    // 模块 2: 3,360
    for &(_, _, pair_name) in &WIN_CONTRAST_PAIRS {
        for s in &SIDE7_NAMES {
            for m in &CORE_NAMES {
                names.push(format!("s1_wdiff_{}_{}_{}", pair_name, s, m));
            }
            for m in &CORE_NAMES {
                names.push(format!("s1_wratio_{}_{}_{}", pair_name, s, m));
            }
        }
    }

    // 模块 3: 8,568
    let min_col_names = build_min_col_names();
    let dummy = Array2::<f32>::ones((2, N_MIN_COLS));
    let (_, red_names) =
        features::get_features_factors_rust_full(&dummy.view(), &min_col_names, false);
    names.extend(red_names);

    // 模块 4: 2,160
    let cond_ch_names = ["bid", "ask", "mixed", "bid_minus_ask"];
    for (r_idx, r_name) in REGIME10_NAMES.iter().enumerate() {
        for ch_name in &cond_ch_names {
            for m in &CORE_NAMES {
                names.push(format!("s1_cond_{}_{}_{}_hi_lo_diff", r_name, ch_name, m));
                names.push(format!("s1_cond_{}_{}_{}_hi_lo_ratio", r_name, ch_name, m));
                names.push(format!("s1_cond_{}_{}_{}_state_corr", r_name, ch_name, m));
            }
        }
        for m in &CORE_NAMES {
            names.push(format!("s1_cond_{}_act_minus_pas_{}_hi_lo_diff", r_name, m));
            if r_idx < 5 {
                names.push(format!("s1_cond_{}_act_minus_pas_{}_state_corr", r_name, m));
            }
        }
    }

    // 模块 5: 384
    for q_name in &QUAD_PAIR_NAMES {
        for ch_name in &cond_ch_names {
            for m in &CORE_NAMES {
                names.push(format!("s1_quad_{}_{}_{}_diff", q_name, ch_name, m));
            }
        }
    }

    // 模块 6: 384
    let evt_win_names = [
        "pre_high15",
        "post_high15",
        "pre_low15",
        "post_low15",
        "pre_maxvol15",
        "post_maxvol15",
    ];
    for ew in &evt_win_names {
        for d_name in &["bid", "ask"] {
            for m in &CORE_NAMES {
                names.push(format!("s1_evt_{}_{}_{}", ew, d_name, m));
            }
        }
    }
    let evt_diff_names = [
        "post_vs_pre_high",
        "post_vs_pre_low",
        "post_vs_pre_maxvol",
        "pre_high_vs_pre_low",
        "post_high_vs_post_low",
        "pre_maxvol_vs_pre_high",
    ];
    for ed in &evt_diff_names {
        for d_name in &["bid", "ask"] {
            for m in &CORE_NAMES {
                names.push(format!("s1_evtdiff_{}_{}_{}", ed, d_name, m));
            }
        }
    }

    // 模块 7: 316
    for sc_name in &["scale5m", "scale15m", "scale30m", "scalefullday"] {
        for d_name in &["bid", "ask", "mixed"] {
            for m in &CORE_NAMES {
                names.push(format!("s1_{}_{}_{}", sc_name, d_name, m));
            }
        }
    }
    for pair_sc in &["scale1m_vs_15m", "scale5m_vs_fullday"] {
        for d_name in &["bid", "ask", "mixed"] {
            for m in &CORE_NAMES {
                names.push(format!("s1_{}_{}_{}_diff", pair_sc, d_name, m));
            }
        }
    }
    for d_name in &["bid", "ask"] {
        for k in 0..14 {
            names.push(format!("s1_scale_slope_{}_{}", d_name, CORE_NAMES[k]));
        }
    }

    assert_eq!(names.len(), EXPECTED_LEN);
    names
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

pub fn compute_anneal_regime_diff_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let trades = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    Ok(compute_anneal_regime_diff_from_records(&trades))
}

#[pyfunction]
pub fn py_anneal_regime_diff(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_regime_diff_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_anneal_regime_diff_names() -> Vec<String> {
    anneal_regime_diff_names()
}

#[pyfunction]
pub fn py_anneal_regime_diff_from_data(
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
    let trades: Vec<TradeRecord> = (0..n)
        .filter_map(|i| {
            if f[i] == 32 {
                return None;
            }
            build_trade_record(t[i], p[i], v[i], tn[i], f[i], bo[i], ao[i], true)
        })
        .collect();
    Ok(compute_anneal_regime_diff_from_records(&trades))
}
