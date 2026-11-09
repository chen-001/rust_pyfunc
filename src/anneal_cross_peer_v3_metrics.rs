//! 柜员游戏扩展系列3 v3（anneal_cross_peer_v3）：以跨股票信息为主的同伴与近邻可还原性因子。
//!
//! 设计依据（v1/v2 评估）：分钟矩阵类因子与拓展1高度重复、挤占入选席位；真正独有的信号在同伴群的
//! 「组内排名 − 全市场排名」；收益率不是性质好的相似度指标。因此 v3：
//!   1. 分钟矩阵瘦身：60 秒保留单列统计量，两两相关只留 8 个高产指标（496 对）；30 秒只留单列统计量。
//!   2. 同伴群扩到 26 个分组维度（行业、市值、订单结构、时间结构、订单编号、盘口流动性、波动结构等）。
//!   3. 近邻：过去 60 日 8 个视角（成交额/波动/日内量形态/交易节奏/订单结构/挂单等待/可还原性/收益）
//!      + 共识近邻；当天 4 个视角（收益/成交量/笔数/平均单笔）；复杂度近邻只保留对收益与波动的溢出。
//!   4. 新增缺口模块（收益、量、笔数、大单、主买、波动六个度量）与 30 秒突变同步模块。
//!   5. 去掉模板退火模块（两版都没有因子入选）。
//!
//! 组合 pipeline（两轮）：
//!   第一轮 `anneal_cross_peer_v3_profile`：每天算每股 271 列中间数据（34 个画像字段 + 237 分钟成交量占比），
//!     当作普通因子存储（研究期存因子库，生产期经 write_selected_factor_base 写成 h5）。
//!   第二轮 `anneal_cross_peer_v3`：读中间数据过去 60 个交易日，算正式因子。第二轮运行前由
//!     rust_pyfunc.prepare_anneal_cross_peer_v3_history_* 把中间数据取到临时目录（环境变量 ACP3_PROFILE_DIR），
//!     Rust 只读临时目录，只读严格早于当天的日期；窗口内任何一天缺失都直接报错。
//! 时间口径与 v2 相同：微秒时间、读数据时下午前移，交易时段 [0, 14220] 秒。算不出来的值一律 NaN。
//!
//! 每只股票 42,268 维，顺序：
//!   [m60 全天/下午/尾盘30] 每块 5,200 = 大盘联动 1,152 + 降维 2,512 + 大盘状态 1,536
//!   [s30 全天/下午/尾盘30] 每块 2,016
//!   同伴群 11,232（组内排名 / 组内减全局 / 扣掉组别水平的纯净版）+ 近邻对比 7,020 + 溢出 1,504 + 缺口 216 + 同步 648

use crate::anneal_cross_peer_metrics::list_codes;
use crate::anneal_regime_diff_metrics::{
    adaptive_steps, anneal_core16, FastAnnealBuf, CORE_NAMES, N_CORE,
};
use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use crate::features;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ============================================================================
// 常量
// ============================================================================

pub const N_CH: usize = 3 * N_CORE;
pub const DIR3_NAMES: [&str; 3] = ["bid", "ask", "bid_minus_ask"];
const SESSION_SEC: i64 = 14_220;
const US: i64 = 1_000_000;
const NB60: usize = 237;
const NB30: usize = 474;
const N_SCALE: usize = 2;
const SCALE_SEC: [i64; N_SCALE] = [60, 30];
const SCALE_TAGS: [&str; N_SCALE] = ["m60", "s30"];
const N_SEG: usize = 3;
const SEG_START_SEC: [i64; N_SEG] = [0, 7_200, 12_420];
const SEG_TAGS: [&str; N_SEG] = ["full", "pm", "late30"];
const SEG_EDGE_SEC: [i64; N_SEG] = [1_800, 1_800, 600];
/// 整段窗口（同伴群 / 近邻对比）：全天、下午、尾盘 30 分钟
const N_WIN: usize = 3;
const WIN_START_SEC: [i64; N_WIN] = [0, 7_200, 12_420];
const WIN_TAGS: [&str; N_WIN] = ["full", "pm", "late30"];
/// 缺口模块窗口：全天、上午、下午、尾盘 30 分钟（左闭右开，秒）
const N_W4: usize = 4;
const W4_BOUNDS: [(i64, i64); N_W4] = [
    (0, SESSION_SEC + 1),
    (0, 7_200),
    (7_200, SESSION_SEC + 1),
    (12_420, SESSION_SEC + 1),
];
const W4_BIN: [(usize, usize); N_W4] = [(0, NB60), (0, 120), (120, NB60), (207, NB60)];
/// 六个度量（收益 / 成交额异动 / 笔数异动 / 大单占比 / 主买占比 / 分钟波动）
const N_GAP_M: usize = 6;

// 画像字段下标
pub const P_LOG_AMT: usize = 0;
pub const P_RVOL: usize = 1;
pub const P_IVOL: usize = 2;
pub const P_LOG_CNT: usize = 3;
pub const P_INTERVAL_CV: usize = 4;
pub const P_ACTIVE_SEC: usize = 5;
pub const P_OPEN30: usize = 6;
pub const P_LATE30: usize = 7;
pub const P_MEAN_ORD: usize = 8;
pub const P_LARGE: usize = 9;
pub const P_HHI: usize = 10;
pub const P_PAS_SPLIT: usize = 11;
pub const P_ACT_SPLIT: usize = 12;
pub const P_WAIT_MED: usize = 13;
pub const P_WAIT_DISP: usize = 14;
pub const P_IMPACT: usize = 15;
pub const P_JUMP: usize = 16;
pub const P_DOWN: usize = 17;
pub const P_TICK: usize = 18;
pub const P_ABR: usize = 19;
pub const P_DAY_RET: usize = 20;
pub const P_FINAL_R: usize = 21;
pub const P_SPREAD: usize = 22;
pub const P_DEPTH: usize = 23;
pub const P_IMB_STD: usize = 24;
pub const P_MV: usize = 25;
pub const P_TURN: usize = 26;
pub const P_IND: usize = 27;
pub const P_AMT_AM: usize = 28;
pub const P_AMT_PM: usize = 29;
pub const P_AMT_LATE: usize = 30;
pub const P_CNT_AM: usize = 31;
pub const P_CNT_PM: usize = 32;
pub const P_CNT_LATE: usize = 33;
pub const N_PROF: usize = 34;
const N_HIST_DAYS: usize = 60;
const N_HIST_MEAN: usize = 20;
pub const N_PROFILE_COLS: usize = N_PROF + NB60;
/// 日内量形态视角至少需要的有效天数
const MIN_SHAPE_DAYS: usize = 30;

/// 溢出目标
pub const N_TARGET: usize = 14;
pub const TARGET_NAMES: [&str; N_TARGET] = [
    "day_ret", "late30_ret", "early30_ret", "log_amt", "ivol_cc", "ivol_inbin", "abr",
    "mean_ord", "log_cnt", "amt_surprise", "vol_surprise", "large_share", "wait_med", "order_hhi",
];
const T_DAY_RET: usize = 0;
const T_LATE30: usize = 1;
const T_EARLY30: usize = 2;
const T_LOG_AMT: usize = 3;
const T_IVOL_CC: usize = 4;
const T_IVOL_IN: usize = 5;
const T_ABR: usize = 6;
const T_MEAN_ORD: usize = 7;
const T_LOG_CNT: usize = 8;
const T_AMT_SUR: usize = 9;
const T_VOL_SUR: usize = 10;
const T_LARGE: usize = 11;
const T_WAIT: usize = 12;
const T_HHI: usize = 13;
const COMP_TARGETS: [usize; 3] = [T_DAY_RET, T_IVOL_CC, T_IVOL_IN];

/// 模块 2 两两相关的 8 个高产指标：steps_r80 / steps_r90 / max_drawdown / half_life / dr_std / final_r / sign_flip / posD
const PAIR_CORE: [usize; 8] = [5, 6, 9, 4, 10, 1, 15, 14];
/// 近邻对比用的 18 个通道：6 核心指标 × 3 方向
const XCH_CORE: [usize; 6] = [1, 3, 7, 9, 14, 15];
const N_XCH: usize = 3 * XCH_CORE.len();
/// 30 秒突变同步用的 6 个通道
const SYNC_CH: [usize; 6] = [1, 3, 15, 17, 19, 31];
/// 突变同步的近邻网下标（共识、成交额、波动、订单结构、挂单等待、可还原性）
const SYNC_NETS: [usize; 6] = [8, 0, 1, 4, 5, 6];

const N_STABLE: usize = 9;
const N_TODAY: usize = 4;
const N_NETS: usize = N_STABLE + N_TODAY;
pub const NET_NAMES: [&str; N_NETS] = [
    "st_amt", "st_rvol", "st_shape", "st_rhythm", "st_order", "st_wait", "st_final_r", "st_ret",
    "st_consensus", "td_ret", "td_vol", "td_cnt", "td_ordsize",
];
const COMP_NET_CHS: [usize; 2] = [1, 17];
const COMP_NET_NAMES: [&str; 2] = ["cnet_bid_r", "cnet_ask_r"];

/// 分组维度（共 26 个）
pub const N_GROUP: usize = 26;
pub const GROUP_NAMES: [&str; N_GROUP] = [
    "ind", "float_mv", "turn", "ind_size", "ord_mean", "ord_large", "ord_hhi", "pas_split",
    "iv_cv", "act_sec", "open30", "late30", "wait_med", "wait_disp", "act_split", "spread",
    "depth", "imb_std", "impact", "rvol", "ivol_inbin", "jump", "down", "tick", "abr", "final_r",
];

pub const N_P1: usize = N_CH * 24;
pub const N_RED_COLS: usize = 2 * N_CH;
pub const N_SINGLE: usize = 21 * N_RED_COLS;
pub const N_PAIR_KEEP: usize = 496;
pub const N_P2_60: usize = N_SINGLE + N_PAIR_KEEP;
pub const N_P6: usize = 16 * N_CH * 2;
pub const N_BLOCK60: usize = N_P1 + N_P2_60 + N_P6;
pub const N_BLOCK30: usize = N_SINGLE;
pub const N_SCALE_ALL: usize = N_SEG * (N_BLOCK60 + N_BLOCK30);
pub const N_GRP: usize = N_GROUP * N_WIN * N_CH * 3;
pub const N_NET_CMP: usize = N_NETS * N_WIN * N_XCH * 10;
pub const N_SPILL: usize = N_NETS * N_TARGET * 8 + 2 * COMP_TARGETS.len() * 8;
pub const N_GAP: usize = N_STABLE * N_GAP_M * 4;
pub const N_SYNC: usize = SYNC_NETS.len() * SYNC_CH.len() * N_SEG * 6;
pub const N_FACTORS: usize = N_SCALE_ALL + N_GRP + N_NET_CMP + N_SPILL + N_GAP + N_SYNC;

const EPS: f32 = 1e-6;
const MIN_PAIRS: usize = 10;
const K_NN: usize = 15;
const K_TOP: usize = 30;

fn profile_dir() -> std::io::Result<PathBuf> {
    std::env::var("ACP3_PROFILE_DIR").map(PathBuf::from).map_err(|_| {
        Error::new(
            ErrorKind::NotFound,
            "未设置 ACP3_PROFILE_DIR：第二轮前要先调用 prepare_anneal_cross_peer_v3_history_*",
        )
    })
}

#[inline]
fn safe_ln(v: f64) -> f32 {
    if v > 0.0 && v.is_finite() {
        v.ln() as f32
    } else {
        f32::NAN
    }
}

// ============================================================================
// 从 v2 复用的小工具
// ============================================================================

#[inline]
fn nanmean(v: &[f32]) -> f32 {
    let mut s = 0.0f64;
    let mut c = 0usize;
    for &x in v {
        if x.is_finite() {
            s += x as f64;
            c += 1;
        }
    }
    if c == 0 {
        f32::NAN
    } else {
        (s / c as f64) as f32
    }
}

/// x 对 y 的共同有效点上的 (相关系数, x 对 y 的回归斜率 beta, 截距 alpha)。
fn pair_reg(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
    let (mut n, mut sx, mut sy) = (0usize, 0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            n += 1;
            sx += a as f64;
            sy += b as f64;
        }
    }
    if n < MIN_PAIRS {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    let (mx, my) = (sx / n as f64, sy / n as f64);
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            let (da, db) = (a as f64 - mx, b as f64 - my);
            sxy += da * db;
            sxx += da * da;
            syy += db * db;
        }
    }
    let tiny = 1e-18 * n as f64;
    let corr = if sxx > tiny && syy > tiny {
        (sxy / (sxx.sqrt() * syy.sqrt())) as f32
    } else {
        f32::NAN
    };
    let (beta, alpha) = if syy > tiny {
        let b = sxy / syy;
        (b as f32, (mx - b * my) as f32)
    } else {
        (f32::NAN, f32::NAN)
    };
    (corr, beta, alpha)
}

#[inline]
fn pair_corr(x: &[f32], y: &[f32]) -> f32 {
    pair_reg(x, y).0
}

/// 错位相关：corr(a[t], b[t+k])
#[inline]
fn lag_corr(a: &[f32], b: &[f32], k: usize) -> f32 {
    if a.len() <= k {
        return f32::NAN;
    }
    pair_corr(&a[..a.len() - k], &b[k..])
}

/// 横截面百分位排名 [0,1]；有效值少于 2 个时全部 NaN。order_buf 为复用缓冲。
fn rank_pct_into(vals: &[f32], out: &mut [f32], order_buf: &mut Vec<(f32, u32)>) {
    order_buf.clear();
    for (i, &v) in vals.iter().enumerate() {
        if v.is_finite() {
            order_buf.push((v, i as u32));
        }
    }
    out.iter_mut().for_each(|o| *o = f32::NAN);
    let m = order_buf.len();
    if m < 2 {
        return;
    }
    order_buf.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let inv = 1.0 / (m - 1) as f32;
    for (r, &(_, i)) in order_buf.iter().enumerate() {
        out[i as usize] = r as f32 * inv;
    }
}

/// 在有效值中取最高 / 最低 20% 的位置；有效值少于 5 个时两组都为空。
fn extreme_masks(vals: &[f32]) -> (Vec<bool>, Vec<bool>) {
    let n = vals.len();
    let mut idx: Vec<(f32, usize)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (v, i))
        .collect();
    let mut hi = vec![false; n];
    let mut lo = vec![false; n];
    let m = idx.len();
    if m < 5 {
        return (hi, lo);
    }
    idx.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let k = m / 5;
    for &(_, i) in &idx[..k] {
        lo[i] = true;
    }
    for &(_, i) in &idx[m - k..] {
        hi[i] = true;
    }
    (hi, lo)
}

/// 8 路独立累加的点积，便于编译器向量化
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// 按首次出现顺序累加某一方向（买单号或卖单号）的订单成交量；哈希表与输出缓冲复用。
struct OrdAgg {
    pos: FxHashMap<i64, u32>,
    vols: Vec<f32>,
}

impl OrdAgg {
    fn new() -> Self {
        Self {
            pos: FxHashMap::default(),
            vols: Vec::new(),
        }
    }
    fn run(&mut self, trades: &[TradeRecord], idx: &[u32], bid_side: bool) {
        self.pos.clear();
        self.vols.clear();
        for &k in idx {
            let t = &trades[k as usize];
            let id = if bid_side { t.bid_order } else { t.ask_order };
            match self.pos.entry(id) {
                Entry::Occupied(e) => self.vols[*e.get() as usize] += t.volume,
                Entry::Vacant(e) => {
                    e.insert(self.vols.len() as u32);
                    self.vols.push(t.volume);
                }
            }
        }
    }
}

#[inline]
fn write_ch(dst: &mut [f32], b: &[f32; N_CORE], a: &[f32; N_CORE]) {
    for k in 0..N_CORE {
        dst[k] = b[k];
        dst[N_CORE + k] = a[k];
        dst[2 * N_CORE + k] = if b[k].is_finite() && a[k].is_finite() {
            b[k] - a[k]
        } else {
            f32::NAN
        };
    }
}

/// 剔除本股后的群均值
#[inline]
fn group_mean_excl(sum: f64, cnt: u32, self_in: bool, x: f32) -> f32 {
    if self_in && x.is_finite() {
        if cnt > 1 {
            ((sum - x as f64) / (cnt - 1) as f64) as f32
        } else {
            f32::NAN
        }
    } else if cnt > 0 {
        (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

/// 去均值 + 单位长度；有效点少于一半或方差为 0 时不合格
fn normalize_into(v: &[f32], dst: &mut [f32]) -> bool {
    let (mut s, mut c) = (0.0f64, 0usize);
    for &x in v {
        if x.is_finite() {
            s += x as f64;
            c += 1;
        }
    }
    dst.iter_mut().for_each(|d| *d = 0.0);
    if c * 2 < v.len() || c < 5 {
        return false;
    }
    let mean = s / c as f64;
    let mut ss = 0.0f64;
    for &x in v {
        if x.is_finite() {
            ss += (x as f64 - mean).powi(2);
        }
    }
    let norm = ss.sqrt();
    if norm < 1e-9 {
        return false;
    }
    for (d, &x) in dst.iter_mut().zip(v) {
        if x.is_finite() {
            *d = ((x as f64 - mean) / norm) as f32;
        }
    }
    true
}

/// 近邻上的均值 / 加权均值 / 标准差 / 计数 / 最大 / 最小 / 被本股超过的个数
struct NbrStat {
    mean: f32,
    wmean: f32,
    std: f32,
    max: f32,
    min: f32,
    cnt: usize,
    beat: usize,
}

fn nbr_stat(nbrs: &[(u32, f32)], val: impl Fn(usize) -> f32, x: f32) -> NbrStat {
    let (mut s, mut sq, mut ws, mut wt, mut c, mut beat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
    let (mut mx, mut mn) = (f32::NEG_INFINITY, f32::INFINITY);
    for &(j, sim) in nbrs {
        let v = val(j as usize);
        if v.is_finite() {
            s += v as f64;
            sq += (v as f64) * (v as f64);
            let w = sim.max(0.05) as f64;
            ws += v as f64 * w;
            wt += w;
            c += 1;
            mx = mx.max(v);
            mn = mn.min(v);
            if x.is_finite() && x > v {
                beat += 1;
            }
        }
    }
    if c == 0 {
        return NbrStat {
            mean: f32::NAN,
            wmean: f32::NAN,
            std: f32::NAN,
            max: f32::NAN,
            min: f32::NAN,
            cnt: 0,
            beat: 0,
        };
    }
    let nf = c as f64;
    NbrStat {
        mean: (s / nf) as f32,
        wmean: (ws / wt) as f32,
        std: if c >= 2 {
            ((sq - s * s / nf) / (nf - 1.0)).max(0.0).sqrt() as f32
        } else {
            f32::NAN
        },
        max: mx,
        min: mn,
        cnt: c,
        beat,
    }
}

/// 去均值 z 分数（只用有效点）；有效点少于一半或方差 0 时全部置 NaN 并返回 false
fn zscore_in_place(v: &mut [f32]) -> bool {
    let (mut s, mut c) = (0.0f64, 0usize);
    for &x in v.iter() {
        if x.is_finite() {
            s += x as f64;
            c += 1;
        }
    }
    if c * 2 < v.len() || c < 5 {
        v.iter_mut().for_each(|x| *x = f32::NAN);
        return false;
    }
    let mean = s / c as f64;
    let mut ss = 0.0f64;
    for &x in v.iter() {
        if x.is_finite() {
            ss += (x as f64 - mean).powi(2);
        }
    }
    let sd = (ss / c as f64).sqrt();
    if sd < 1e-9 {
        v.iter_mut().for_each(|x| *x = f32::NAN);
        return false;
    }
    for x in v.iter_mut() {
        if x.is_finite() {
            *x = ((*x as f64 - mean) / sd) as f32;
        }
    }
    true
}

// ============================================================================
// 单股统计（不含退火）：时间、订单结构、订单编号等待、时间格序列
// ============================================================================

struct Basic {
    order: Vec<u32>,
    offs: Vec<i64>,
    prof: [f32; N_PROF],
    b_ret: Vec<f32>,
    b_ret_in: Vec<f32>,
    b_amt: Vec<f32>,
    b_vol: Vec<f32>,
    b_cnt: Vec<f32>,
    b_buy: Vec<f32>,
    b_large: Vec<f32>,
    b_wait: Vec<f32>,
    b_ordsize: Vec<f32>,
    shape: Vec<f32>,
    /// [窗口][度量]：收益、log 成交额、log 笔数、大单占比、主买占比、分钟波动
    w4: [[f32; N_GAP_M]; N_W4],
    early_ret: f32,
    late_ret: f32,
    ivol_cc: f32,
    ivol_in: f32,
}

#[inline]
fn is_session(o: i64) -> bool {
    o >= 0 && o < (SESSION_SEC + 1) * US
}

fn quantile_sel(v: &mut [f32], q: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

fn basic_stats(trades: &[TradeRecord]) -> Option<Basic> {
    if trades.len() < 15 {
        return None;
    }
    let mut order: Vec<u32> = (0..trades.len() as u32).collect();
    order.sort_by_key(|&k| trades[k as usize].time_us);
    let t0 = trades[order[0] as usize].time_us;
    let open_us = t0 - t0.rem_euclid(86_400 * US) + (9 * 3600 + 30 * 60) * US;
    order.retain(|&k| is_session(trades[k as usize].time_us - open_us));
    let n = order.len();
    if n < 15 {
        return None;
    }
    let offs: Vec<i64> = order
        .iter()
        .map(|&k| trades[k as usize].time_us - open_us)
        .collect();

    let (mut total_vol, mut total_amt, mut buy_amt) = (0.0f64, 0.0f64, 0.0f64);
    let mut a_map: FxHashMap<i64, u32> = FxHashMap::with_capacity_and_hasher(n / 2 + 1, Default::default());
    let mut p_set: FxHashSet<i64> = FxHashSet::with_capacity_and_hasher(n / 2 + 1, Default::default());
    let mut a_amt: Vec<f32> = Vec::with_capacity(n / 2 + 1);
    let mut a_seen_bin: Vec<u16> = Vec::with_capacity(n / 2 + 1);
    let mut act_pos: Vec<u32> = Vec::with_capacity(n);
    let mut act_ids: Vec<i64> = Vec::with_capacity(n);
    let mut act_offs: Vec<i64> = Vec::with_capacity(n);
    let mut b_ord = vec![0.0f32; NB60];
    let mut valid_tr = 0usize;
    for (k, &ti) in order.iter().enumerate() {
        let t = &trades[ti as usize];
        total_vol += t.volume as f64;
        total_amt += t.turnover as f64;
        let (a, p) = match t.flag {
            66 => {
                buy_amt += t.turnover as f64;
                (t.bid_order, t.ask_order)
            }
            83 => (t.ask_order, t.bid_order),
            _ => (0, 0),
        };
        let b = ((offs[k] / (60 * US)) as usize).min(NB60 - 1) as u16;
        if a > 0 {
            valid_tr += 1;
            let idx = match a_map.entry(a) {
                Entry::Occupied(e) => {
                    let id = *e.get();
                    a_amt[id as usize] += t.turnover;
                    id
                }
                Entry::Vacant(e) => {
                    let id = a_amt.len() as u32;
                    e.insert(id);
                    a_amt.push(t.turnover);
                    a_seen_bin.push(u16::MAX);
                    id
                }
            };
            act_pos.push(idx);
            if a_seen_bin[idx as usize] != b {
                a_seen_bin[idx as usize] = b;
                b_ord[b as usize] += 1.0;
            }
            act_ids.push(a);
            act_offs.push(offs[k]);
            if p > 0 {
                p_set.insert(p);
            }
        } else {
            act_pos.push(u32::MAX);
        }
    }
    if total_vol <= 0.0 || total_amt <= 0.0 {
        return None;
    }

    // ---- 订单结构 ----
    let na = a_amt.len();
    let (large_thr, ord_mean, hhi) = if na >= 5 {
        let mut c = a_amt.clone();
        let thr = quantile_sel(&mut c, 0.9);
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for &x in &a_amt {
            s += x as f64;
            s2 += (x as f64).powi(2);
        }
        (thr, safe_ln(s / na as f64), if s > 0.0 { (s2 / (s * s)) as f32 } else { f32::NAN })
    } else {
        (f32::NAN, f32::NAN, f32::NAN)
    };

    // ---- 挂单等待：用本股主动单「单号 → 时间」的对应关系把被动单号换算成挂单时刻（秒） ----
    let mut waits: Vec<f32> = Vec::with_capacity(valid_tr);
    let mut trade_wait: Vec<f32> = vec![f32::NAN; n];
    if act_ids.len() >= 2 {
        for (k, &ti) in order.iter().enumerate() {
            let t = &trades[ti as usize];
            let p = match t.flag {
                66 => t.ask_order,
                83 => t.bid_order,
                _ => 0,
            };
            if p <= 0 {
                continue;
            }
            let pos = act_ids.partition_point(|&a| a < p);
            let place = if pos == 0 {
                0
            } else if pos >= act_ids.len() {
                offs[k]
            } else {
                let (a0, a1) = (act_ids[pos - 1], act_ids[pos]);
                let (t0, t1) = (act_offs[pos - 1], act_offs[pos]);
                if a1 > a0 {
                    t0 + (((p - a0) as f64 / (a1 - a0) as f64) * (t1 - t0) as f64) as i64
                } else {
                    t1
                }
            };
            let w = ((offs[k] - place).max(0) as f64 / US as f64) as f32;
            trade_wait[k] = w;
            waits.push(w);
        }
    }
    let (wait_med, wait_disp) = if waits.len() >= 5 {
        let (p25, p50, p75) = (
            quantile_sel(&mut waits, 0.25),
            quantile_sel(&mut waits, 0.5),
            quantile_sel(&mut waits, 0.75),
        );
        ((1.0 + p50 as f64).ln() as f32, (p75 - p25) / (p50 + 1.0))
    } else {
        (f32::NAN, f32::NAN)
    };

    // ---- 成交时间结构 ----
    let (mut dist, mut last_ts, mut secs, mut last_sec) = (0usize, i64::MIN, 0usize, i64::MIN);
    let (mut gs, mut gs2, mut gc) = (0.0f64, 0.0f64, 0usize);
    for &o in &offs {
        if o != last_ts {
            if last_ts != i64::MIN {
                let g = (o - last_ts) as f64 / US as f64;
                gs += g;
                gs2 += g * g;
                gc += 1;
            }
            last_ts = o;
            dist += 1;
        }
        let s = o / US;
        if s != last_sec {
            last_sec = s;
            secs += 1;
        }
    }
    let _ = dist;
    let interval_cv = if gc >= 5 && gs > 0.0 {
        let m = gs / gc as f64;
        ((gs2 / gc as f64 - m * m).max(0.0).sqrt() / m) as f32
    } else {
        f32::NAN
    };

    // ---- 60 秒时间格 ----
    let mut b_amt = vec![0.0f32; NB60];
    let mut b_vol = vec![0.0f32; NB60];
    let mut b_cnt = vec![0.0f32; NB60];
    let mut b_buy = vec![0.0f32; NB60];
    let mut b_large = vec![0.0f32; NB60];
    let mut b_wsum = vec![0.0f64; NB60];
    let mut b_wc = vec![0u32; NB60];
    let mut b_first = vec![f32::NAN; NB60];
    let mut b_last = vec![f32::NAN; NB60];
    let mut w_amt = [0.0f64; N_W4];
    let mut w_cnt = [0.0f64; N_W4];
    let mut w_buy = [0.0f64; N_W4];
    let mut w_large = [0.0f64; N_W4];
    let mut w_first = [f32::NAN; N_W4];
    let mut w_last = [f32::NAN; N_W4];
    for (k, &ti) in order.iter().enumerate() {
        let t = &trades[ti as usize];
        let b = ((offs[k] / (60 * US)) as usize).min(NB60 - 1);
        b_amt[b] += t.turnover;
        b_vol[b] += t.volume;
        b_cnt[b] += 1.0;
        if b_first[b].is_nan() {
            b_first[b] = t.price;
        }
        b_last[b] = t.price;
        let is_buy = t.flag == 66;
        if is_buy {
            b_buy[b] += t.turnover;
        }
        let is_large = act_pos[k] != u32::MAX && large_thr.is_finite() && a_amt[act_pos[k] as usize] >= large_thr;
        if is_large {
            b_large[b] += t.turnover;
        }
        if trade_wait[k].is_finite() {
            b_wsum[b] += (1.0 + trade_wait[k] as f64).ln();
            b_wc[b] += 1;
        }
        for (w, &(lo, hi)) in W4_BOUNDS.iter().enumerate() {
            if offs[k] >= lo * US && offs[k] < hi * US {
                w_amt[w] += t.turnover as f64;
                w_cnt[w] += 1.0;
                if is_buy {
                    w_buy[w] += t.turnover as f64;
                }
                if is_large {
                    w_large[w] += t.turnover as f64;
                }
                if w_first[w].is_nan() {
                    w_first[w] = t.price;
                }
                w_last[w] = t.price;
            }
        }
    }
    let mut b_ret = vec![f32::NAN; NB60];
    let mut b_ret_in = vec![f32::NAN; NB60];
    let mut b_wait = vec![f32::NAN; NB60];
    let mut b_ordsize = vec![f32::NAN; NB60];
    let mut prev: Option<f32> = None;
    for b in 0..NB60 {
        if b_cnt[b] > 0.0 {
            let refp = prev.unwrap_or(b_first[b]);
            if refp > 0.0 {
                b_ret[b] = (b_last[b] - refp) / refp;
            }
            if b_first[b] > 0.0 {
                b_ret_in[b] = (b_last[b] - b_first[b]) / b_first[b];
            }
            prev = Some(b_last[b]);
            if b_wc[b] > 0 {
                b_wait[b] = (b_wsum[b] / b_wc[b] as f64) as f32;
            }
            if b_ord[b] > 0.0 {
                b_ordsize[b] = safe_ln(b_amt[b] as f64 / b_ord[b] as f64);
            }
        }
    }
    let mut shape = vec![f32::NAN; NB60];
    for b in 0..NB60 {
        shape[b] = (b_vol[b] as f64 / total_vol) as f32;
    }

    // 分钟波动（收盘对收盘）、分钟内波动（v1 算法）、跳跃与下行占比、价格冲击
    let nanstd = |v: &[f32], min: usize| -> f32 {
        if v.iter().filter(|x| x.is_finite()).count() >= min {
            features::col_std(v)
        } else {
            f32::NAN
        }
    };
    let ivol_cc = nanstd(&b_ret, 10);
    let ivol_in = nanstd(&b_ret_in, 10);
    let (mut rv, mut bv, mut down, mut abs_sum, mut amt_used) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut prev_abs: Option<f64> = None;
    let mut nv = 0usize;
    for b in 0..NB60 {
        let r = b_ret[b] as f64;
        if r.is_finite() {
            nv += 1;
            rv += r * r;
            if r < 0.0 {
                down += r * r;
            }
            abs_sum += r.abs();
            amt_used += b_amt[b] as f64;
            if let Some(p) = prev_abs {
                bv += p * r.abs();
            }
            prev_abs = Some(r.abs());
        } else {
            prev_abs = None;
        }
    }
    let bv = bv * std::f64::consts::FRAC_PI_2;
    let jump = if nv >= 10 && rv > 0.0 { ((1.0 - bv / rv).max(0.0)) as f32 } else { f32::NAN };
    let downside = if nv >= 10 && rv > 0.0 { (down / rv) as f32 } else { f32::NAN };
    let impact = if nv >= 10 && abs_sum > 0.0 && amt_used > 0.0 { safe_ln(abs_sum / amt_used) } else { f32::NAN };

    // 窗口度量
    let mut w4 = [[f32::NAN; N_GAP_M]; N_W4];
    for w in 0..N_W4 {
        if w_amt[w] > 0.0 {
            if w_first[w] > 0.0 && w_cnt[w] >= 2.0 {
                w4[w][0] = (w_last[w] - w_first[w]) / w_first[w];
            }
            w4[w][1] = safe_ln(w_amt[w]);
            w4[w][2] = safe_ln(w_cnt[w]);
            if large_thr.is_finite() {
                w4[w][3] = (w_large[w] / w_amt[w]) as f32;
            }
            w4[w][4] = (w_buy[w] / w_amt[w]) as f32;
        }
        let (a, b) = W4_BIN[w];
        w4[w][5] = nanstd(&b_ret[a..b], 5);
    }
    let win_ret = |lo: i64, hi: i64| -> f32 {
        let a = offs.partition_point(|&o| o < lo * US);
        let b = offs.partition_point(|&o| o < hi * US);
        if b < a + 2 {
            return f32::NAN;
        }
        let (p0, p1) = (trades[order[a] as usize].price, trades[order[b - 1] as usize].price);
        if p0 > 0.0 { (p1 - p0) / p0 } else { f32::NAN }
    };
    let early_ret = win_ret(0, 1_800);
    let late_ret = win_ret(12_420, SESSION_SEC + 1);
    let day_ret = win_ret(0, SESSION_SEC + 1);

    let mut prof = [f32::NAN; N_PROF];
    prof[P_LOG_AMT] = safe_ln(total_amt);
    prof[P_RVOL] = ivol_cc;
    prof[P_IVOL] = ivol_in;
    prof[P_LOG_CNT] = (n as f32).ln();
    prof[P_INTERVAL_CV] = interval_cv;
    prof[P_ACTIVE_SEC] = secs as f32 / SESSION_SEC as f32;
    prof[P_OPEN30] = b_amt[..30].iter().map(|&x| x as f64).sum::<f64>() as f32 / total_amt as f32;
    prof[P_LATE30] = b_amt[207..].iter().map(|&x| x as f64).sum::<f64>() as f32 / total_amt as f32;
    prof[P_MEAN_ORD] = ord_mean;
    prof[P_LARGE] = if w_amt[0] > 0.0 && large_thr.is_finite() { (w_large[0] / w_amt[0]) as f32 } else { f32::NAN };
    prof[P_HHI] = hhi;
    prof[P_PAS_SPLIT] = if !p_set.is_empty() { valid_tr as f32 / p_set.len() as f32 } else { f32::NAN };
    prof[P_ACT_SPLIT] = if na > 0 { valid_tr as f32 / na as f32 } else { f32::NAN };
    prof[P_WAIT_MED] = wait_med;
    prof[P_WAIT_DISP] = wait_disp;
    prof[P_IMPACT] = impact;
    prof[P_JUMP] = jump;
    prof[P_DOWN] = downside;
    prof[P_TICK] = (0.01 / (total_amt / total_vol)) as f32;
    prof[P_ABR] = (buy_amt / total_amt) as f32;
    prof[P_DAY_RET] = day_ret;
    prof[P_AMT_AM] = w4[1][1];
    prof[P_AMT_PM] = w4[2][1];
    prof[P_AMT_LATE] = w4[3][1];
    prof[P_CNT_AM] = w4[1][2];
    prof[P_CNT_PM] = w4[2][2];
    prof[P_CNT_LATE] = w4[3][2];

    Some(Basic {
        order,
        offs,
        prof,
        b_ret,
        b_ret_in,
        b_amt,
        b_vol,
        b_cnt,
        b_buy,
        b_large,
        b_wait,
        b_ordsize,
        shape,
        w4,
        early_ret,
        late_ret,
        ivol_cc,
        ivol_in,
    })
}

/// 盘口：相对价差（以 0.01 元计）、前五档深度相对成交额、前五档失衡的波动
fn lob_stats(lob: &[MarketRecord], log_amt: f32) -> (f32, f32, f32) {
    let (mut s, mut d, mut c) = (0.0f64, 0.0f64, 0usize);
    let mut imb: Vec<f32> = Vec::with_capacity(lob.len());
    for m in lob {
        let (a1, b1) = (m.ask_prcs[0], m.bid_prcs[0]);
        if !(a1 > 0.0 && b1 > 0.0 && a1 >= b1) {
            continue;
        }
        s += ((a1 - b1) / 0.01) as f64;
        let (mut dep, mut ask5, mut bid5) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..5 {
            dep += m.ask_prcs[k] as f64 * m.ask_vols[k] as f64 + m.bid_prcs[k] as f64 * m.bid_vols[k] as f64;
            ask5 += m.ask_vols[k] as f64;
            bid5 += m.bid_vols[k] as f64;
        }
        d += dep;
        c += 1;
        if ask5 + bid5 > 0.0 {
            imb.push(((bid5 - ask5) / (bid5 + ask5)) as f32);
        }
    }
    if c < 20 {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    let spread = (s / c as f64) as f32;
    let depth = if log_amt.is_finite() { safe_ln(d / c as f64) - log_amt } else { f32::NAN };
    let imb_std = if imb.len() >= 20 { features::col_std(&imb) } else { f32::NAN };
    (spread, depth, imb_std)
}

// ============================================================================
// 单股中间特征（第 ② 步）
// ============================================================================

pub struct StockMidV3 {
    bin_series: [Vec<f32>; N_SCALE],
    b: Basic,
    win_ch: [[f32; N_CH]; N_WIN],
    summary: [f32; N_TARGET],
}

pub fn extract_stock_mid_v3(trades: &[TradeRecord]) -> Option<StockMidV3> {
    let b = basic_stats(trades)?;
    let mut buf = FastAnnealBuf::new();
    let mut agg_b = OrdAgg::new();
    let mut agg_a = OrdAgg::new();
    let end_us = (SESSION_SEC + 1) * US;
    let range = |lo_us: i64, hi_us: i64| -> (usize, usize) {
        (
            b.offs.partition_point(|&o| o < lo_us),
            b.offs.partition_point(|&o| o < hi_us),
        )
    };
    let mut bin_series: [Vec<f32>; N_SCALE] = Default::default();
    for s in 0..N_SCALE {
        let nb = (SESSION_SEC / SCALE_SEC[s]) as usize;
        let w_us = SCALE_SEC[s] * US;
        let mut ser = vec![f32::NAN; nb * N_CH];
        for bi in 0..nb {
            let lo_us = bi as i64 * w_us;
            let hi_us = if bi + 1 == nb { end_us } else { lo_us + w_us };
            let (lo, hi) = range(lo_us, hi_us);
            if hi < lo + 3 {
                continue;
            }
            let idx = &b.order[lo..hi];
            agg_b.run(trades, idx, true);
            agg_a.run(trades, idx, false);
            let bc = anneal_core16(&agg_b.vols, adaptive_steps(agg_b.vols.len(), 4_000), &mut buf);
            let ac = anneal_core16(&agg_a.vols, adaptive_steps(agg_a.vols.len(), 4_000), &mut buf);
            write_ch(&mut ser[bi * N_CH..(bi + 1) * N_CH], &bc, &ac);
        }
        bin_series[s] = ser;
    }
    let mut win_ch = [[f32::NAN; N_CH]; N_WIN];
    for w in 0..N_WIN {
        let (lo, hi) = range(WIN_START_SEC[w] * US, end_us);
        if hi < lo + 3 {
            continue;
        }
        let idx = &b.order[lo..hi];
        agg_b.run(trades, idx, true);
        agg_a.run(trades, idx, false);
        let bc = anneal_core16(&agg_b.vols, adaptive_steps(agg_b.vols.len(), 15_000), &mut buf);
        let ac = anneal_core16(&agg_a.vols, adaptive_steps(agg_a.vols.len(), 15_000), &mut buf);
        write_ch(&mut win_ch[w], &bc, &ac);
    }
    let p = &b.prof;
    let mut summary = [f32::NAN; N_TARGET];
    summary[T_DAY_RET] = p[P_DAY_RET];
    summary[T_LATE30] = b.late_ret;
    summary[T_EARLY30] = b.early_ret;
    summary[T_LOG_AMT] = p[P_LOG_AMT];
    summary[T_IVOL_CC] = b.ivol_cc;
    summary[T_IVOL_IN] = b.ivol_in;
    summary[T_ABR] = p[P_ABR];
    summary[T_MEAN_ORD] = p[P_MEAN_ORD];
    summary[T_LOG_CNT] = p[P_LOG_CNT];
    summary[T_LARGE] = p[P_LARGE];
    summary[T_WAIT] = p[P_WAIT_MED];
    summary[T_HHI] = p[P_HHI];
    Some(StockMidV3 { bin_series, b, win_ch, summary })
}

// ============================================================================
// 每股每日画像：一次性预计算，v3 只读严格早于当天的日期
// ============================================================================

const PROFILE_MAGIC: &[u8; 4] = b"ACP3";

/// 中间数据（第一轮）的列名：34 个画像字段 + 237 分钟成交量占比
pub const PROFILE_FIELD_NAMES: [&str; N_PROF] = [
    "log_amt", "rvol", "ivol", "log_cnt", "interval_cv", "active_sec", "open30", "late30",
    "mean_ord", "large", "hhi", "pas_split", "act_split", "wait_med", "wait_disp", "impact",
    "jump", "down", "tick", "abr", "day_ret", "final_r", "spread", "depth", "imb_std", "mv",
    "turn", "ind", "amt_am", "amt_pm", "amt_late", "cnt_am", "cnt_pm", "cnt_late",
];

pub fn anneal_cross_peer_v3_profile_names() -> Vec<String> {
    let mut v: Vec<String> = PROFILE_FIELD_NAMES.iter().map(|n| format!("acp3p_{n}")).collect();
    for b in 0..NB60 {
        v.push(format!("acp3p_shape_{b:03}"));
    }
    assert_eq!(v.len(), N_PROFILE_COLS);
    v
}

/// 从日频库读当天一个字段；当天还没更新（文件缺失、日历无此日或整行为空）直接报错，不静默留空。
fn read_daily_field(ds: &str, field: &str, date: i64, codes: &[String]) -> std::io::Result<Vec<f32>> {
    let err = |m: String| Error::new(ErrorKind::InvalidData, format!("日频字段 {ds}/{field} [{date}]: {m}"));
    let dir = crate::data_paths::vars_path(ds);
    let mut cal = csv::Reader::from_path(dir.join("calendar_map.csv")).map_err(|e| err(e.to_string()))?;
    let mut row = None;
    for (r, rec) in cal.records().enumerate() {
        let rec = rec.map_err(|e| err(e.to_string()))?;
        if rec.get(0).and_then(|s| s.trim().parse::<i64>().ok()) == Some(date) {
            row = Some(r);
            break;
        }
    }
    let row = row.ok_or_else(|| err("日历中没有这一天".to_string()))?;
    let mut sym = csv::Reader::from_path(dir.join("symbol_map.csv")).map_err(|e| err(e.to_string()))?;
    let hdr = sym.headers().map_err(|e| err(e.to_string()))?.clone();
    let col = |name: &str| hdr.iter().position(|h| h == name).ok_or_else(|| err(format!("symbol_map 缺少 {name} 列")));
    let (cs, cp) = (col("symbol")?, col("pos")?);
    let mut pos: FxHashMap<String, usize> = FxHashMap::default();
    for rec in sym.records() {
        let rec = rec.map_err(|e| err(e.to_string()))?;
        if let (Some(s), Some(p)) = (rec.get(cs), rec.get(cp).and_then(|x| x.trim().parse::<usize>().ok())) {
            pos.insert(format!("{:0>6}", s.trim()), p);
        }
    }
    let file = hdf5_metno::File::open(dir.join(format!("{field}.h5"))).map_err(|e| err(e.to_string()))?;
    let ds_h = file.dataset("data").map_err(|e| err(e.to_string()))?;
    let values: ndarray::Array1<f64> = ds_h
        .read_slice_1d(ndarray::s![row, ..])
        .map_err(|e| err(e.to_string()))?;
    if !values.iter().any(|v| v.is_finite()) {
        return Err(err("这一天整行为空，日频库可能尚未更新".to_string()));
    }
    Ok(codes
        .iter()
        .map(|c| match pos.get(c) {
            Some(&p) if p < values.len() => values[p] as f32,
            _ => f32::NAN,
        })
        .collect())
}

/// 第一轮：当天全市场每只股票的中间数据（列序见 anneal_cross_peer_v3_profile_names）。
pub fn compute_anneal_cross_peer_v3_profile_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction")?;
    let mv = read_daily_field("DailyDerived", "S_DQ_MV", date, &codes)?;
    let turn = read_daily_field("DailyDerived", "S_DQ_TURN", date, &codes)?;
    let ind = read_daily_field("SzBa", "industry", date, &codes)?;
    let rows: Vec<Option<Vec<f32>>> = codes
        .par_iter()
        .enumerate()
        .map(|(j, code)| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            let b = basic_stats(&trades)?;
            let mut p = b.prof;
            // 全天可还原性：买卖两侧 final_r 的均值
            let (mut agg_b, mut agg_a, mut buf) = (OrdAgg::new(), OrdAgg::new(), FastAnnealBuf::new());
            agg_b.run(&trades, &b.order, true);
            agg_a.run(&trades, &b.order, false);
            let bc = anneal_core16(&agg_b.vols, adaptive_steps(agg_b.vols.len(), 15_000), &mut buf);
            let ac = anneal_core16(&agg_a.vols, adaptive_steps(agg_a.vols.len(), 15_000), &mut buf);
            p[P_FINAL_R] = nanmean(&[bc[1], ac[1]]);
            if let Ok(lob) = read_market_fast_inner(code, date, false, true, usize::MAX) {
                let (s, d, i) = lob_stats(&lob, p[P_LOG_AMT]);
                p[P_SPREAD] = s;
                p[P_DEPTH] = d;
                p[P_IMB_STD] = i;
            }
            p[P_MV] = if mv[j] > 0.0 { mv[j].ln() } else { f32::NAN };
            p[P_TURN] = if turn[j] > 0.0 { turn[j].ln() } else { f32::NAN };
            p[P_IND] = ind[j];
            let mut row = Vec::with_capacity(N_PROFILE_COLS);
            row.extend_from_slice(&p);
            row.extend_from_slice(&b.shape);
            Some(row)
        })
        .collect();
    let mut kept = Vec::with_capacity(codes.len());
    let mut vals = Vec::with_capacity(codes.len() * N_PROFILE_COLS);
    for (c, r) in codes.into_iter().zip(rows) {
        if let Some(v) = r {
            kept.push(c);
            vals.extend_from_slice(&v);
        }
    }
    Ok((kept, vals))
}

/// 过去 N_HIST_DAYS 个交易日（严格早于 date）的中间数据，按 valid_codes 对齐。
struct History {
    days: usize,
    /// [日][股][字段]
    prof: Vec<f32>,
    /// [日][股][237]
    shape: Vec<f32>,
}

fn load_history(dir: &Path, date: i64, codes: &[String]) -> std::io::Result<History> {
    // calendar.txt 由 prepare_anneal_cross_peer_v3_history_* 写入（完整交易日历，每行一个日期）
    let cal_path = dir.join("calendar.txt");
    let cal = std::fs::read_to_string(&cal_path)
        .map_err(|e| Error::new(e.kind(), format!("读取 {} 失败: {e}", cal_path.display())))?;
    let mut all: Vec<i64> = cal
        .split_whitespace()
        .filter_map(|s| s.parse::<i64>().ok())
        .filter(|&d| d < date)
        .collect();
    all.sort_unstable();
    if all.len() < N_HIST_DAYS {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{date} 之前的交易日只有 {} 天，需要 {N_HIST_DAYS} 天", all.len()),
        ));
    }
    let picked = &all[all.len() - N_HIST_DAYS..];
    let missing: Vec<i64> = picked
        .iter()
        .copied()
        .filter(|d| !dir.join(format!("{d}.bin")).is_file())
        .collect();
    if !missing.is_empty() {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("{date} 需要的中间数据缺 {} 天: {:?}", missing.len(), &missing[..missing.len().min(10)]),
        ));
    }
    let n = codes.len();
    let pos: FxHashMap<&[u8], usize> = codes.iter().enumerate().map(|(i, c)| (c.as_bytes(), i)).collect();
    let days = picked.len();
    let mut prof = vec![f32::NAN; days * n * N_PROF];
    let mut shape = vec![f32::NAN; days * n * NB60];
    prof.par_chunks_mut(n * N_PROF)
        .zip(shape.par_chunks_mut(n * NB60))
        .zip(picked.par_iter())
        .try_for_each(|((pd, sd), d)| -> std::io::Result<()> {
            let path = dir.join(format!("{d}.bin"));
            let bytes = std::fs::read(&path)?;
            let bad = || Error::new(ErrorKind::InvalidData, format!("画像文件损坏: {}", path.display()));
            if bytes.len() < 16 || &bytes[..4] != PROFILE_MAGIC {
                return Err(bad());
            }
            let rd = |k: usize| u32::from_le_bytes(bytes[k..k + 4].try_into().unwrap()) as usize;
            let (m, p, s) = (rd(4), rd(8), rd(12));
            if p != N_PROF || s != NB60 || bytes.len() != 16 + m * 6 + m * p * 4 + m * s * 4 {
                return Err(bad());
            }
            let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            let (p0, s0) = (16 + m * 6, 16 + m * 6 + m * p * 4);
            for j in 0..m {
                let c = &bytes[16 + j * 6..16 + (j + 1) * 6];
                if let Some(&i) = pos.get(c) {
                    for k in 0..N_PROF {
                        pd[i * N_PROF + k] = f32_at(p0 + (j * N_PROF + k) * 4);
                    }
                    for k in 0..NB60 {
                        sd[i * NB60 + k] = f32_at(s0 + (j * NB60 + k) * 4);
                    }
                }
            }
            Ok(())
        })?;
    Ok(History { days, prof, shape })
}

// ============================================================================
// 第 ③ 步：横截面
// ============================================================================

struct ScaleCross {
    nb: usize,
    z: Vec<f32>,
    rk: Vec<f32>,
    mkt_mean: Vec<f32>,
    lead_sum: Vec<f64>,
    lead_cnt: Vec<u32>,
    lag_sum: Vec<f64>,
    lag_cnt: Vec<u32>,
}

fn build_scale_cross(
    s: usize,
    stocks: &[StockMidV3],
    is_leader: &[bool],
    is_laggard: &[bool],
) -> ScaleCross {
    let n = stocks.len();
    let nb = (SESSION_SEC / SCALE_SEC[s]) as usize;
    let mut z = vec![f32::NAN; nb * n * N_CH];
    let mut rk = vec![f32::NAN; nb * n * N_CH];
    #[derive(Clone)]
    struct BinAgg {
        mean: [f32; N_CH],
        ls: [f64; N_CH],
        lc: [u32; N_CH],
        gs: [f64; N_CH],
        gc: [u32; N_CH],
    }
    let aggs: Vec<BinAgg> = z
        .par_chunks_mut(n * N_CH)
        .zip(rk.par_chunks_mut(n * N_CH))
        .enumerate()
        .map(|(b, (zc, rc))| {
            let mut col = vec![0.0f32; n];
            let mut rcol = vec![0.0f32; n];
            let mut obuf: Vec<(f32, u32)> = Vec::with_capacity(n);
            let mut a = BinAgg {
                mean: [f32::NAN; N_CH],
                ls: [0.0; N_CH],
                lc: [0; N_CH],
                gs: [0.0; N_CH],
                gc: [0; N_CH],
            };
            for ch in 0..N_CH {
                let (mut sum, mut cnt) = (0.0f64, 0usize);
                for i in 0..n {
                    let v = stocks[i].bin_series[s][b * N_CH + ch];
                    col[i] = v;
                    if v.is_finite() {
                        sum += v as f64;
                        cnt += 1;
                        if is_leader[i] {
                            a.ls[ch] += v as f64;
                            a.lc[ch] += 1;
                        }
                        if is_laggard[i] {
                            a.gs[ch] += v as f64;
                            a.gc[ch] += 1;
                        }
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let mean = sum / cnt as f64;
                a.mean[ch] = mean as f32;
                let mut ss = 0.0f64;
                for &v in &col {
                    if v.is_finite() {
                        ss += (v as f64 - mean).powi(2);
                    }
                }
                let sd = if cnt >= 2 { (ss / (cnt - 1) as f64).sqrt() } else { f64::NAN };
                if sd.is_finite() && sd > 1e-6 {
                    for i in 0..n {
                        if col[i].is_finite() {
                            zc[i * N_CH + ch] = ((col[i] as f64 - mean) / sd) as f32;
                        }
                    }
                }
                rank_pct_into(&col, &mut rcol, &mut obuf);
                for i in 0..n {
                    rc[i * N_CH + ch] = rcol[i];
                }
            }
            a
        })
        .collect();
    let mut sc = ScaleCross {
        nb,
        z,
        rk,
        mkt_mean: vec![f32::NAN; nb * N_CH],
        lead_sum: vec![0.0; nb * N_CH],
        lead_cnt: vec![0; nb * N_CH],
        lag_sum: vec![0.0; nb * N_CH],
        lag_cnt: vec![0; nb * N_CH],
    };
    for (b, a) in aggs.iter().enumerate() {
        let o = b * N_CH;
        sc.mkt_mean[o..o + N_CH].copy_from_slice(&a.mean);
        sc.lead_sum[o..o + N_CH].copy_from_slice(&a.ls);
        sc.lead_cnt[o..o + N_CH].copy_from_slice(&a.lc);
        sc.lag_sum[o..o + N_CH].copy_from_slice(&a.gs);
        sc.lag_cnt[o..o + N_CH].copy_from_slice(&a.gc);
    }
    sc
}

/// 60 秒时间格上的全市场状态序列
struct MktState {
    ret: Vec<f32>,
    disp: Vec<f32>,
    amt: Vec<f32>,
    abr: Vec<f32>,
    cnt: Vec<f32>,
    large: Vec<f32>,
    wait: Vec<f32>,
}

fn build_mkt_state(stocks: &[StockMidV3]) -> MktState {
    let mut m = MktState {
        ret: vec![f32::NAN; NB60],
        disp: vec![f32::NAN; NB60],
        amt: vec![f32::NAN; NB60],
        abr: vec![f32::NAN; NB60],
        cnt: vec![f32::NAN; NB60],
        large: vec![f32::NAN; NB60],
        wait: vec![f32::NAN; NB60],
    };
    for b in 0..NB60 {
        let rets: Vec<f32> = stocks.iter().map(|s| s.b.b_ret[b]).collect();
        let valid = rets.iter().filter(|v| v.is_finite()).count();
        if valid > 0 {
            m.ret[b] = nanmean(&rets);
        }
        if valid >= 2 {
            m.disp[b] = features::col_std(&rets);
        }
        let (mut a, mut by, mut c, mut lg) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for st in stocks {
            a += st.b.b_amt[b] as f64;
            by += st.b.b_buy[b] as f64;
            c += st.b.b_cnt[b] as f64;
            lg += st.b.b_large[b] as f64;
        }
        m.amt[b] = a as f32;
        m.cnt[b] = c as f32;
        if a > 0.0 {
            m.abr[b] = (by / a) as f32;
            m.large[b] = (lg / a) as f32;
        }
        let waits: Vec<f32> = stocks.iter().map(|s| s.b.b_wait[b]).collect();
        m.wait[b] = nanmean(&waits);
    }
    m
}

fn seg_masks(sc: &ScaleCross, ms: &MktState, b0: usize) -> [Vec<bool>; 16] {
    let ease: Vec<f32> = (b0..sc.nb).map(|b| sc.mkt_mean[b * N_CH + 1]).collect();
    let (a0, a1) = extreme_masks(&ms.ret[b0..]);
    let (b0m, b1m) = extreme_masks(&ms.disp[b0..]);
    let (c0, c1) = extreme_masks(&ms.amt[b0..]);
    let (d0, d1) = extreme_masks(&ease);
    let (e0, e1) = extreme_masks(&ms.abr[b0..]);
    let (f0, f1) = extreme_masks(&ms.cnt[b0..]);
    let (g0, g1) = extreme_masks(&ms.large[b0..]);
    let (h0, h1) = extreme_masks(&ms.wait[b0..]);
    [a0, a1, b0m, b1m, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1]
}

fn red_col_names() -> Vec<String> {
    let mut v = Vec::with_capacity(N_RED_COLS);
    for pre in ["z", "rk"] {
        for d in DIR3_NAMES {
            for m in CORE_NAMES {
                v.push(format!("{pre}_{d}_{m}"));
            }
        }
    }
    v
}

/// 模块 2 输出中要保留的下标：[60 秒（单列 + 8 指标两两相关）, 30 秒（只单列）]
fn red_keep() -> &'static [Vec<usize>; 2] {
    static KEEP: OnceLock<[Vec<usize>; 2]> = OnceLock::new();
    KEEP.get_or_init(|| {
        let cols = red_col_names();
        let dummy = Array2::<f32>::ones((2, N_RED_COLS));
        let (_, names) = features::get_features_factors_rust_full(&dummy.view(), &cols, false);
        let mut pairs: FxHashSet<String> = FxHashSet::default();
        for i in 0..N_RED_COLS {
            for j in 0..N_RED_COLS {
                if i != j {
                    pairs.insert(format!("{}_corr_{}", cols[i], cols[j]));
                }
            }
        }
        // 允许的两两相关：bid/ask × z/rk × 8 指标 = 32 列
        let mut sub: Vec<&String> = Vec::with_capacity(32);
        for pre in 0..2usize {
            for d in 0..2usize {
                for &m in PAIR_CORE.iter() {
                    sub.push(&cols[pre * N_CH + d * N_CORE + m]);
                }
            }
        }
        let mut allowed: FxHashSet<String> = FxHashSet::default();
        for a in &sub {
            for b in &sub {
                if a != b {
                    allowed.insert(format!("{a}_corr_{b}"));
                }
            }
        }
        let (mut k60, mut k30) = (Vec::new(), Vec::new());
        for (i, n) in names.iter().enumerate() {
            let is_pair = pairs.contains(n);
            if !is_pair {
                k60.push(i);
                k30.push(i);
            } else if allowed.contains(n) {
                k60.push(i);
            }
        }
        assert_eq!(k60.len(), N_P2_60);
        assert_eq!(k30.len(), N_BLOCK30);
        [k60, k30]
    })
}

struct BlockBufs {
    z_loc: Vec<f32>,
    rk_loc: Vec<f32>,
    x: Vec<f32>,
    m: Vec<f32>,
    l: Vec<f32>,
    g: Vec<f32>,
    zs: Vec<f32>,
    rs: Vec<f32>,
}

impl BlockBufs {
    fn new() -> Self {
        Self {
            z_loc: Vec::with_capacity(NB30 * N_CH),
            rk_loc: Vec::with_capacity(NB30 * N_CH),
            x: Vec::with_capacity(NB30),
            m: Vec::with_capacity(NB30),
            l: Vec::with_capacity(NB30),
            g: Vec::with_capacity(NB30),
            zs: Vec::with_capacity(NB30),
            rs: Vec::with_capacity(NB30),
        }
    }
}

/// 某只股票某尺度的三个时段块。60 秒：模块 1 + 模块 2（筛后）+ 模块 6；30 秒：只模块 2 单列统计量。
#[allow(clippy::too_many_arguments)]
fn fill_scale_blocks(
    out: &mut [f32],
    i: usize,
    s: usize,
    sc: &ScaleCross,
    masks: Option<&[[Vec<bool>; 16]; N_SEG]>,
    st: &StockMidV3,
    is_leader: bool,
    is_laggard: bool,
    n: usize,
    bb: &mut BlockBufs,
) {
    let nb = sc.nb;
    let w = SCALE_SEC[s];
    bb.z_loc.clear();
    bb.rk_loc.clear();
    for b in 0..nb {
        let o = (b * n + i) * N_CH;
        bb.z_loc.extend_from_slice(&sc.z[o..o + N_CH]);
        bb.rk_loc.extend_from_slice(&sc.rk[o..o + N_CH]);
    }
    let ser = &st.bin_series[s];
    let keep = &red_keep()[s];
    let blk_len = if s == 0 { N_BLOCK60 } else { N_BLOCK30 };
    for seg in 0..N_SEG {
        let b0 = (SEG_START_SEC[seg] / w) as usize;
        let len = nb - b0;
        let edge = ((SEG_EDGE_SEC[seg] / w) as usize).min(len / 2);
        let blk = &mut out[seg * blk_len..(seg + 1) * blk_len];
        let mut p = 0usize;

        if s == 0 {
            // 模块 1：大盘节奏、领先滞后、龙头 / 冷门群、排名霸榜
            for ch in 0..N_CH {
                bb.x.clear();
                bb.m.clear();
                bb.l.clear();
                bb.g.clear();
                bb.zs.clear();
                bb.rs.clear();
                for b in b0..nb {
                    let o = b * N_CH + ch;
                    let x = ser[o];
                    bb.x.push(x);
                    bb.m.push(sc.mkt_mean[o]);
                    bb.l.push(group_mean_excl(sc.lead_sum[o], sc.lead_cnt[o], is_leader, x));
                    bb.g.push(group_mean_excl(sc.lag_sum[o], sc.lag_cnt[o], is_laggard, x));
                    bb.zs.push(bb.z_loc[o]);
                    bb.rs.push(bb.rk_loc[o]);
                }
                let (corr_m, beta_m, alpha_m) = pair_reg(&bb.x, &bb.m);
                let r2 = corr_m * corr_m;
                let res_e = nanmean(&bb.zs[..edge]);
                let res_m = nanmean(&bb.zs[edge..len - edge]);
                let res_l = nanmean(&bb.zs[len - edge..]);
                let lead1 = lag_corr(&bb.x, &bb.m, 1);
                let lag1 = lag_corr(&bb.m, &bb.x, 1);
                let lead3 = lag_corr(&bb.x, &bb.m, 3);
                let lag3 = lag_corr(&bb.m, &bb.x, 3);
                let lead5 = lag_corr(&bb.x, &bb.m, 5);
                let lag5 = lag_corr(&bb.m, &bb.x, 5);
                let c_lead = pair_corr(&bb.x, &bb.l);
                let c_lag = pair_corr(&bb.x, &bb.g);
                let (mut valid, mut top5, mut top10, mut bot10, mut streak, mut max_streak) =
                    (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
                for &r in &bb.rs {
                    if r.is_finite() {
                        valid += 1;
                        if r >= 0.95 {
                            top5 += 1;
                        }
                        if r >= 0.90 {
                            top10 += 1;
                            streak += 1;
                            max_streak = max_streak.max(streak);
                        } else {
                            streak = 0;
                        }
                        if r <= 0.10 {
                            bot10 += 1;
                        }
                    }
                }
                let share = |c: usize| if valid > 0 { c as f32 / valid as f32 } else { f32::NAN };
                let rk_e = nanmean(&bb.rs[..edge]);
                let rk_l = nanmean(&bb.rs[len - edge..]);
                let vals = [
                    corr_m, beta_m, alpha_m, r2, res_e, res_m, res_l, res_l - res_e, res_l - res_m,
                    lead1, lag1, lead1 - lag1, lead3, lag3, lead3 - lag3, lead5 - lag5, c_lead, c_lag,
                    c_lead - c_lag, share(top5), share(top10), share(bot10), share(max_streak), rk_l - rk_e,
                ];
                blk[p..p + 24].copy_from_slice(&vals);
                p += 24;
            }
        }

        // 模块 2：标准分(48) + 排名(48) 矩阵降维（筛选后的列）
        let mut mat = Array2::<f32>::from_elem((len, N_RED_COLS), f32::NAN);
        for (r, b) in (b0..nb).enumerate() {
            let o = b * N_CH;
            for ch in 0..N_CH {
                mat[[r, ch]] = bb.z_loc[o + ch];
                mat[[r, N_CH + ch]] = bb.rk_loc[o + ch];
            }
        }
        let red = features::get_features_factors_rust_values_only(&mat.view(), false);
        for &k in keep.iter() {
            blk[p] = red[k];
            p += 1;
        }

        // 模块 6：大盘状态下的超额响应（只 60 秒）
        if let Some(ms) = masks {
            for mask in ms[seg].iter() {
                for ch in 0..N_CH {
                    let (mut sz, mut cz, mut sr, mut cr) = (0.0f64, 0usize, 0.0f64, 0usize);
                    for (r, &on) in mask.iter().enumerate() {
                        if !on {
                            continue;
                        }
                        let o = (b0 + r) * N_CH + ch;
                        let (zv, rv) = (bb.z_loc[o], bb.rk_loc[o]);
                        if zv.is_finite() {
                            sz += zv as f64;
                            cz += 1;
                        }
                        if rv.is_finite() {
                            sr += rv as f64;
                            cr += 1;
                        }
                    }
                    blk[p] = if cz > 0 { (sz / cz as f64) as f32 } else { f32::NAN };
                    blk[p + 1] = if cr > 0 { (sr / cr as f64) as f32 } else { f32::NAN };
                    p += 2;
                }
            }
        }
        debug_assert_eq!(p, blk_len);
    }
}

type NbrList = Vec<(u32, f32)>;
type Net = Vec<Option<(NbrList, NbrList)>>;

/// 归一化向量（n × dim 连续存放）上求每只股票的前 30 与后 15 相似股
fn knn_top_bot(vecs: &[f32], dim: usize, ok: &[bool]) -> Net {
    let n = ok.len();
    (0..n)
        .into_par_iter()
        .map_init(
            || Vec::<(f32, u32)>::with_capacity(n),
            |sims, i| {
                if !ok[i] {
                    return None;
                }
                let vi = &vecs[i * dim..(i + 1) * dim];
                sims.clear();
                for j in 0..n {
                    if j != i && ok[j] {
                        sims.push((dot(vi, &vecs[j * dim..(j + 1) * dim]), j as u32));
                    }
                }
                let m = sims.len();
                if m == 0 {
                    return None;
                }
                let desc = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
                let asc = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
                let kt = K_TOP.min(m);
                let kb = K_NN.min(m);
                sims.select_nth_unstable_by(kt - 1, desc);
                let mut top: Vec<(f32, u32)> = sims[..kt].to_vec();
                top.sort_by(desc);
                sims.select_nth_unstable_by(kb - 1, asc);
                let mut bot: Vec<(f32, u32)> = sims[..kb].to_vec();
                bot.sort_by(asc);
                Some((
                    top.into_iter().map(|(s, j)| (j, s)).collect(),
                    bot.into_iter().map(|(s, j)| (j, s)).collect(),
                ))
            },
        )
        .collect()
}

/// 历史视角向量：每个字段先按日做横截面去均值（去掉全市场共同波动），再按股票做 z 分数后首尾拼接
fn build_hist_view(h: &History, comps: &[(usize, bool)], n: usize) -> (Vec<f32>, Vec<bool>, usize) {
    let days = h.days;
    let dim = comps.len() * days;
    let mut mats: Vec<Vec<f32>> = Vec::with_capacity(comps.len());
    for &(f, log) in comps {
        let mut m = vec![f32::NAN; n * days];
        for d in 0..days {
            let (mut s, mut c) = (0.0f64, 0usize);
            for i in 0..n {
                let mut v = h.prof[(d * n + i) * N_PROF + f];
                if log {
                    v = if v > 0.0 { v.ln() } else { f32::NAN };
                }
                if v.is_finite() {
                    s += v as f64;
                    c += 1;
                    m[i * days + d] = v;
                }
            }
            if c > 0 {
                let mean = (s / c as f64) as f32;
                for i in 0..n {
                    m[i * days + d] -= mean;
                }
            }
        }
        mats.push(m);
    }
    let mut vecs = vec![0.0f32; n * dim];
    let ok: Vec<bool> = vecs
        .par_chunks_mut(dim)
        .enumerate()
        .map(|(i, dst)| {
            let mut raw = Vec::with_capacity(dim);
            for m in &mats {
                let mut seg: Vec<f32> = m[i * days..(i + 1) * days].to_vec();
                if !zscore_in_place(&mut seg) {
                    return false;
                }
                raw.extend_from_slice(&seg);
            }
            normalize_into(&raw, dst)
        })
        .collect();
    (vecs, ok, dim)
}

/// 日内量形态视角：过去各日每分钟成交量占比的平均形状，减去全市场平均形状
fn build_shape_view(h: &History, n: usize) -> (Vec<f32>, Vec<bool>) {
    let mut mean_shape = vec![f32::NAN; n * NB60];
    let days = h.days;
    mean_shape.par_chunks_mut(NB60).enumerate().for_each(|(i, dst)| {
        let mut c = 0usize;
        let mut sum = [0.0f64; NB60];
        for d in 0..days {
            let o = (d * n + i) * NB60;
            let row = &h.shape[o..o + NB60];
            if row.iter().all(|v| v.is_finite()) {
                c += 1;
                for k in 0..NB60 {
                    sum[k] += row[k] as f64;
                }
            }
        }
        if c >= MIN_SHAPE_DAYS {
            for k in 0..NB60 {
                dst[k] = (sum[k] / c as f64) as f32;
            }
        }
    });
    let mut mkt = [0.0f64; NB60];
    let mut c = 0usize;
    for i in 0..n {
        let row = &mean_shape[i * NB60..(i + 1) * NB60];
        if row[0].is_finite() {
            c += 1;
            for k in 0..NB60 {
                mkt[k] += row[k] as f64;
            }
        }
    }
    let mut vecs = vec![0.0f32; n * NB60];
    let ok: Vec<bool> = vecs
        .par_chunks_mut(NB60)
        .enumerate()
        .map(|(i, dst)| {
            let row = &mean_shape[i * NB60..(i + 1) * NB60];
            if !row[0].is_finite() || c == 0 {
                return false;
            }
            let raw: Vec<f32> = (0..NB60).map(|k| row[k] - (mkt[k] / c as f64) as f32).collect();
            normalize_into(&raw, dst)
        })
        .collect();
    (vecs, ok)
}

/// 当天视角向量：cross_demean 时按时间格去掉全市场均值
fn build_today_view(
    stocks: &[StockMidV3],
    get: &(dyn Fn(&StockMidV3, usize) -> f32 + Sync),
    cross_demean: bool,
) -> (Vec<f32>, Vec<bool>) {
    let n = stocks.len();
    let mut bm = vec![0.0f32; NB60];
    if cross_demean {
        for b in 0..NB60 {
            let col: Vec<f32> = stocks.iter().map(|s| get(s, b)).collect();
            let m = nanmean(&col);
            bm[b] = if m.is_finite() { m } else { 0.0 };
        }
    }
    let mut vecs = vec![0.0f32; n * NB60];
    let ok: Vec<bool> = vecs
        .par_chunks_mut(NB60)
        .enumerate()
        .map(|(i, dst)| {
            let raw: Vec<f32> = (0..NB60).map(|b| get(&stocks[i], b) - bm[b]).collect();
            normalize_into(&raw, dst)
        })
        .collect();
    (vecs, ok)
}

/// 共识近邻：在 8 个历史视角的前 30 / 后 15 中至少出现 3 次
fn consensus_net(views: &[Net], n: usize) -> Net {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let collect = |top: bool| -> NbrList {
                let mut m: FxHashMap<u32, (u32, f32)> = FxHashMap::default();
                for v in views {
                    if let Some((t, b)) = &v[i] {
                        for &(j, s) in if top { t.iter() } else { b.iter() } {
                            let e = m.entry(j).or_insert((0, 0.0));
                            e.0 += 1;
                            e.1 += s;
                        }
                    }
                }
                let mut out: Vec<(u32, u32, f32)> = m
                    .into_iter()
                    .filter(|(_, (c, _))| *c >= 3)
                    .map(|(j, (c, s))| (j, c, s))
                    .collect();
                out.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)).then(a.0.cmp(&b.0)));
                out.truncate(K_NN);
                out.into_iter().map(|(j, c, _)| (j, c as f32 / 8.0)).collect()
            };
            let top = collect(true);
            if top.is_empty() {
                None
            } else {
                Some((top, collect(false)))
            }
        })
        .collect()
}

/// 只在有效值上分 10 档；by_class 给出时先在各类内部排名（用于受交易所影响的指标）
fn deciles(metric: &[f32], by_class: Option<&[u8]>) -> Vec<Option<u16>> {
    let n = metric.len();
    let mut out = vec![None; n];
    let classes: Vec<u8> = match by_class {
        Some(c) => {
            let mut v = c.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        }
        None => vec![0],
    };
    for cl in classes {
        let mut idx: Vec<(f32, usize)> = metric
            .iter()
            .enumerate()
            .filter(|(i, v)| v.is_finite() && by_class.map_or(true, |c| c[*i] == cl))
            .map(|(i, &v)| (v, i))
            .collect();
        idx.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let m = idx.len();
        for (r, &(_, i)) in idx.iter().enumerate() {
            out[i] = Some(((r * 10 / m).min(9)) as u16);
        }
    }
    out
}

fn exchange_of(code: &str) -> u8 {
    match code.as_bytes().first() {
        Some(b'6') => 0,
        Some(b'0') | Some(b'3') => 1,
        _ => 2,
    }
}

const SHL_WORDS: usize = 8;
type Bits = [u64; SHL_WORDS];

#[inline]
fn shl1(a: &Bits) -> Bits {
    let mut out = [0u64; SHL_WORDS];
    let mut carry = 0u64;
    for k in 0..SHL_WORDS {
        out[k] = (a[k] << 1) | carry;
        carry = a[k] >> 63;
    }
    out
}

#[inline]
fn and_cnt(a: &Bits, b: &Bits) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

#[inline]
fn and_bits(a: &Bits, b: &Bits) -> Bits {
    let mut o = [0u64; SHL_WORDS];
    for k in 0..SHL_WORDS {
        o[k] = a[k] & b[k];
    }
    o
}

#[inline]
fn popcnt(a: &Bits) -> u32 {
    a.iter().map(|x| x.count_ones()).sum()
}

fn tail_bits(b0: usize, nb: usize) -> Bits {
    let mut o = [0u64; SHL_WORDS];
    for b in b0..nb {
        o[b / 64] |= 1u64 << (b % 64);
    }
    o
}

// ============================================================================
// 横截面主合成（无 IO 之外的历史读取）
// ============================================================================

pub fn compute_cross_peer_v3_from_mids(
    date: i64,
    codes: &[String],
    mids: Vec<Option<StockMidV3>>,
    prof_dir: &Path,
) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let mut valid_codes = Vec::new();
    let mut stocks: Vec<StockMidV3> = Vec::new();
    for (c, m) in codes.iter().zip(mids) {
        if let Some(st) = m {
            valid_codes.push(c.clone());
            stocks.push(st);
        }
    }
    let n = stocks.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let hist = load_history(prof_dir, date, &valid_codes)?;
    let mut out = vec![f32::NAN; n * N_FACTORS];

    // ---- 过去 20 日均值（分组与异动用）----
    let d0 = hist.days.saturating_sub(N_HIST_MEAN);
    let mut h_mean = vec![f32::NAN; n * N_PROF];
    let mut h_lnrv = vec![f32::NAN; n];
    let mut h_ind = vec![f32::NAN; n];
    for i in 0..n {
        for k in 0..N_PROF {
            let v: Vec<f32> = (d0..hist.days).map(|d| hist.prof[(d * n + i) * N_PROF + k]).collect();
            if v.iter().filter(|x| x.is_finite()).count() >= 10 {
                h_mean[i * N_PROF + k] = nanmean(&v);
            }
        }
        let v: Vec<f32> = (d0..hist.days)
            .map(|d| {
                let x = hist.prof[(d * n + i) * N_PROF + P_RVOL];
                if x > 0.0 { x.ln() } else { f32::NAN }
            })
            .collect();
        if v.iter().filter(|x| x.is_finite()).count() >= 10 {
            h_lnrv[i] = nanmean(&v);
        }
        for d in (0..hist.days).rev() {
            let x = hist.prof[(d * n + i) * N_PROF + P_IND];
            if x.is_finite() && x >= 1.0 {
                h_ind[i] = x;
                break;
            }
        }
    }

    // ---- 把当天的异动填回单股摘要与窗口度量 ----
    const AMT_FIELD: [usize; N_W4] = [P_LOG_AMT, P_AMT_AM, P_AMT_PM, P_AMT_LATE];
    const CNT_FIELD: [usize; N_W4] = [P_LOG_CNT, P_CNT_AM, P_CNT_PM, P_CNT_LATE];
    for (i, st) in stocks.iter_mut().enumerate() {
        st.summary[T_AMT_SUR] = st.summary[T_LOG_AMT] - h_mean[i * N_PROF + P_LOG_AMT];
        let rv = st.summary[T_IVOL_CC];
        st.summary[T_VOL_SUR] = (if rv > 0.0 { rv.ln() } else { f32::NAN }) - h_lnrv[i];
        for w in 0..N_W4 {
            st.b.w4[w][1] -= h_mean[i * N_PROF + AMT_FIELD[w]];
            st.b.w4[w][2] -= h_mean[i * N_PROF + CNT_FIELD[w]];
        }
    }

    // ---- 龙头 / 冷门群 ----
    let mut amt_order: Vec<(f32, usize)> = stocks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.summary[T_LOG_AMT], i))
        .collect();
    amt_order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let k_lead = (n / 10).clamp(1, 50.min(n));
    let k_lag = (n / 5).max(1);
    let mut is_leader = vec![false; n];
    let mut is_laggard = vec![false; n];
    for &(_, i) in &amt_order[n - k_lead..] {
        is_leader[i] = true;
    }
    for &(_, i) in &amt_order[..k_lag] {
        is_laggard[i] = true;
    }

    let ms = build_mkt_state(&stocks);

    // ---- 分钟矩阵模块（两个尺度），算完释放；30 秒尺度顺手提取突变位图 ----
    let mut shock_bits: Vec<[Bits; 6]> = vec![[[0u64; SHL_WORDS]; 6]; n];
    let mut valid_bits: Vec<[Bits; 6]> = vec![[[0u64; SHL_WORDS]; 6]; n];
    for s in 0..N_SCALE {
        let sc = build_scale_cross(s, &stocks, &is_leader, &is_laggard);
        let masks: Option<[[Vec<bool>; 16]; N_SEG]> = if s == 0 {
            Some(std::array::from_fn(|seg| seg_masks(&sc, &ms, (SEG_START_SEC[seg] / SCALE_SEC[s]) as usize)))
        } else {
            None
        };
        let off = if s == 0 { 0 } else { N_SEG * N_BLOCK60 };
        let blk = if s == 0 { N_SEG * N_BLOCK60 } else { N_SEG * N_BLOCK30 };
        out.par_chunks_mut(N_FACTORS).enumerate().for_each_init(BlockBufs::new, |bb, (i, row)| {
            fill_scale_blocks(&mut row[off..off + blk], i, s, &sc, masks.as_ref(), &stocks[i], is_leader[i], is_laggard[i], n, bb);
        });
        if s == 1 {
            shock_bits
                .par_iter_mut()
                .zip(valid_bits.par_iter_mut())
                .enumerate()
                .for_each(|(i, (sb, vb))| {
                    for (c, &ch) in SYNC_CH.iter().enumerate() {
                        for b in 0..NB30 {
                            let z = sc.z[(b * n + i) * N_CH + ch];
                            if z.is_finite() {
                                vb[c][b / 64] |= 1u64 << (b % 64);
                                if z.abs() >= 2.0 {
                                    sb[c][b / 64] |= 1u64 << (b % 64);
                                }
                            }
                        }
                    }
                });
        }
    }

    // ---- 26 个同伴分组 ----
    let exch: Vec<u8> = valid_codes.iter().map(|c| exchange_of(c)).collect();
    let field = |k: usize| -> Vec<f32> { (0..n).map(|i| h_mean[i * N_PROF + k]).collect() };
    let mut labels: Vec<Vec<Option<u16>>> = Vec::with_capacity(N_GROUP);
    const G_FIELDS: [(usize, bool); N_GROUP] = [
        (P_IND, false), (P_MV, false), (P_TURN, false), (P_IND, false), (P_MEAN_ORD, true),
        (P_LARGE, true), (P_HHI, true), (P_PAS_SPLIT, true), (P_INTERVAL_CV, false),
        (P_ACTIVE_SEC, false), (P_OPEN30, false), (P_LATE30, false), (P_WAIT_MED, true),
        (P_WAIT_DISP, true), (P_ACT_SPLIT, true), (P_SPREAD, false), (P_DEPTH, false),
        (P_IMB_STD, false), (P_IMPACT, false), (P_RVOL, false), (P_IVOL, false), (P_JUMP, false),
        (P_DOWN, false), (P_TICK, false), (P_ABR, false), (P_FINAL_R, false),
    ];
    for (g, &(f, by_ex)) in G_FIELDS.iter().enumerate() {
        if g == 0 {
            labels.push(h_ind.iter().map(|&x| if x.is_finite() { Some(x as u16 - 1) } else { None }).collect());
        } else if g == 3 {
            // 行业内规模：同一行业内按流通市值中位数二分（2 × 31 组）
            let mv = field(P_MV);
            let mut grp: FxHashMap<u16, Vec<(f32, usize)>> = FxHashMap::default();
            for i in 0..n {
                if h_ind[i].is_finite() && mv[i].is_finite() {
                    grp.entry(h_ind[i] as u16 - 1).or_default().push((mv[i], i));
                }
            }
            let mut l = vec![None; n];
            for (ind, mut v) in grp {
                v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let m = v.len();
                for (r, &(_, i)) in v.iter().enumerate() {
                    l[i] = Some(ind * 2 + if r * 2 >= m { 1 } else { 0 });
                }
            }
            labels.push(l);
        } else {
            let metric = field(f);
            labels.push(deciles(&metric, if by_ex { Some(&exch) } else { None }));
        }
    }
    let members: Vec<Vec<Vec<u32>>> = labels
        .iter()
        .map(|l| {
            let maxl = l.iter().flatten().copied().max().map_or(0, |x| x as usize + 1);
            let mut m = vec![Vec::new(); maxl];
            for (i, v) in l.iter().enumerate() {
                if let Some(x) = v {
                    m[*x as usize].push(i as u32);
                }
            }
            m
        })
        .collect();
    let mut glob_rank = vec![f32::NAN; n * N_WIN * N_CH];
    glob_rank
        .par_chunks_mut(n)
        .enumerate()
        .for_each_init(
            || (vec![0.0f32; n], Vec::<(f32, u32)>::with_capacity(n)),
            |(col, obuf), (t, chunk)| {
                let (w, ch) = (t / N_CH, t % N_CH);
                for i in 0..n {
                    col[i] = stocks[i].win_ch[w][ch];
                }
                rank_pct_into(col, chunk, obuf);
            },
        );
    let mut grp_out = vec![f32::NAN; N_GROUP * N_WIN * N_CH * n * 3];
    grp_out
        .par_chunks_mut(n * 3)
        .enumerate()
        .for_each_init(
            || (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<(f32, u32)>::new()),
            |(sub, srk, obuf), (t, chunk)| {
                let g = t / (N_WIN * N_CH);
                let rest = t % (N_WIN * N_CH);
                let (w, ch) = (rest / N_CH, rest % N_CH);
                for mem in members[g].iter() {
                    sub.clear();
                    sub.extend(mem.iter().map(|&i| stocks[i as usize].win_ch[w][ch]));
                    srk.clear();
                    srk.resize(sub.len(), f32::NAN);
                    rank_pct_into(sub, srk, obuf);
                    // 组内全局排名的均值：组别本身的水平，纯净版要扣掉它
                    let (mut gs, mut gc) = (0.0f64, 0usize);
                    for &i in mem.iter() {
                        let gl = glob_rank[(w * N_CH + ch) * n + i as usize];
                        if gl.is_finite() {
                            gs += gl as f64;
                            gc += 1;
                        }
                    }
                    let gm = if gc > 0 { (gs / gc as f64) as f32 } else { f32::NAN };
                    for (k, &i) in mem.iter().enumerate() {
                        let i = i as usize;
                        let gl = glob_rank[(w * N_CH + ch) * n + i];
                        chunk[i * 3] = srk[k];
                        chunk[i * 3 + 1] = srk[k] - gl;
                        chunk[i * 3 + 2] = srk[k] - (gl - gm);
                    }
                }
            },
        );

    // ---- 近邻网络 ----
    let mut nets: Vec<Net> = Vec::with_capacity(N_NETS);
    let hist_comps: [&[(usize, bool)]; 8] = [
        &[(P_LOG_AMT, false)],
        &[(P_RVOL, true)],
        &[],
        &[(P_LOG_CNT, false), (P_INTERVAL_CV, false)],
        &[(P_MEAN_ORD, false), (P_LARGE, false)],
        &[(P_WAIT_MED, false)],
        &[(P_FINAL_R, false)],
        &[(P_DAY_RET, false)],
    ];
    for comps in hist_comps.iter() {
        if comps.is_empty() {
            let (v, ok) = build_shape_view(&hist, n);
            nets.push(knn_top_bot(&v, NB60, &ok));
        } else {
            let (v, ok, dim) = build_hist_view(&hist, comps, n);
            nets.push(knn_top_bot(&v, dim, &ok));
        }
    }
    nets.push(consensus_net(&nets[..8], n));
    let today: [(&(dyn Fn(&StockMidV3, usize) -> f32 + Sync), bool); N_TODAY] = [
        (&|s, b| s.b.b_ret[b], false),
        (&|s, b| s.b.shape[b], true),
        (&|s, b| (1.0 + s.b.b_cnt[b]).ln(), true),
        (&|s, b| s.b.b_ordsize[b], true),
    ];
    for (g, cd) in today.iter() {
        let (v, ok) = build_today_view(&stocks, *g, *cd);
        nets.push(knn_top_bot(&v, NB60, &ok));
    }
    let comp_nets: Vec<Net> = COMP_NET_CHS
        .iter()
        .map(|&ch| {
            let (v, ok) = build_today_view(&stocks, &|s, b| s.bin_series[0][b * N_CH + ch], false);
            knn_top_bot(&v, NB60, &ok)
        })
        .collect();

    let trunc = |v: &'_ NbrList| -> usize { v.len().min(K_NN) };
    let xch: Vec<usize> = (0..3)
        .flat_map(|d| XCH_CORE.iter().map(move |&c| d * N_CORE + c))
        .collect();

    // ---- 逐股票写出后半段 ----
    let shock = &shock_bits;
    let valid = &valid_bits;
    out.par_chunks_mut(N_FACTORS).enumerate().for_each(|(i, row)| {
        let st = &stocks[i];
        let tail = &mut row[N_SCALE_ALL..];
        let mut p = 0usize;

        // 同伴群
        for g in 0..N_GROUP {
            for w in 0..N_WIN {
                for ch in 0..N_CH {
                    let o = ((g * N_WIN + w) * N_CH + ch) * n * 3 + i * 3;
                    tail[p] = grp_out[o];
                    tail[p + 1] = grp_out[o + 1];
                    tail[p + 2] = grp_out[o + 2];
                    p += 3;
                }
            }
        }

        // 近邻对比（13 个网 × 3 窗口 × 18 通道 × 10 算子）
        for k in 0..N_NETS {
            for w in 0..N_WIN {
                for &ch in &xch {
                    let x = st.win_ch[w][ch];
                    let vals: [f32; 10] = match &nets[k][i] {
                        Some((top, bot)) if x.is_finite() => {
                            let t = nbr_stat(&top[..trunc(top)], |j| stocks[j].win_ch[w][ch], x);
                            let b = nbr_stat(bot, |j| stocks[j].win_ch[w][ch], x);
                            if t.cnt == 0 {
                                [f32::NAN; 10]
                            } else {
                                [
                                    t.mean, x - t.mean, (x - t.mean) / (t.std + 1e-4), x / (t.mean.abs() + EPS),
                                    x - t.wmean, t.beat as f32 / t.cnt as f32, t.std, x - t.max, x - t.min,
                                    t.mean - b.mean,
                                ]
                            }
                        }
                        _ => [f32::NAN; 10],
                    };
                    tail[p..p + 10].copy_from_slice(&vals);
                    p += 10;
                }
            }
        }

        // 溢出
        let spill = |net: &Net, k: usize| -> [f32; 8] {
            let x = st.summary[k];
            match &net[i] {
                Some((top, bot)) if x.is_finite() => {
                    let t = nbr_stat(&top[..trunc(top)], |j| stocks[j].summary[k], x);
                    let b = nbr_stat(bot, |j| stocks[j].summary[k], x);
                    if t.cnt == 0 {
                        [f32::NAN; 8]
                    } else {
                        [t.mean, t.wmean, x - t.mean, (x - t.mean) / (t.std + 1e-4), x - t.wmean, t.std, b.mean, t.mean - b.mean]
                    }
                }
                _ => [f32::NAN; 8],
            }
        };
        for k in 0..N_NETS {
            for t in 0..N_TARGET {
                tail[p..p + 8].copy_from_slice(&spill(&nets[k], t));
                p += 8;
            }
        }
        for net in comp_nets.iter() {
            for &t in COMP_TARGETS.iter() {
                tail[p..p + 8].copy_from_slice(&spill(net, t));
                p += 8;
            }
        }

        // 缺口：邻居上午 / 全天表现 vs 本股（9 个历史网 × 6 度量 × 4 统计量）
        for k in 0..N_STABLE {
            for m in 0..N_GAP_M {
                let vals: [f32; 4] = match &nets[k][i] {
                    Some((top, _)) => {
                        let tp = &top[..trunc(top)];
                        let pm = |w: usize| -> f32 {
                            let v: Vec<f32> = tp.iter().map(|&(j, _)| stocks[j as usize].b.w4[w][m]).collect();
                            nanmean(&v)
                        };
                        let s = |w: usize| st.b.w4[w][m];
                        let (pa, pf, pp, pl) = (pm(1), pm(0), pm(2), pm(3));
                        let _ = pl;
                        [pa - s(1), pf - s(0), (s(2) - s(1)) - (pp - pa), pp - s(3)]
                    }
                    None => [f32::NAN; 4],
                };
                tail[p..p + 4].copy_from_slice(&vals);
                p += 4;
            }
        }

        // 30 秒突变同步（6 个网 × 6 通道 × 3 时段 × 6 统计量）
        for &k in SYNC_NETS.iter() {
            for c in 0..SYNC_CH.len() {
                for seg in 0..N_SEG {
                    let b0 = (SEG_START_SEC[seg] / 30) as usize;
                    let seg_mask = tail_bits(b0, NB30);
                    let a = and_bits(&shock[i][c], &seg_mask);
                    let va = popcnt(&and_bits(&valid[i][c], &seg_mask));
                    let na = popcnt(&a);
                    let mut vals = [f32::NAN; 6];
                    if va > 0 {
                        vals[0] = na as f32 / va as f32;
                        if let Some((top, _)) = &nets[k][i] {
                            let (mut sr, mut sj, mut sp, mut ss) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                            let (mut cr, mut cj, mut cp, mut cs) = (0usize, 0usize, 0usize, 0usize);
                            let sa = shl1(&a);
                            for &(j, _) in &top[..trunc(top)] {
                                let j = j as usize;
                                let vb = popcnt(&and_bits(&valid[j][c], &seg_mask));
                                if vb == 0 {
                                    continue;
                                }
                                let b = and_bits(&shock[j][c], &seg_mask);
                                let nb = popcnt(&b);
                                sr += nb as f64 / vb as f64;
                                cr += 1;
                                let inter = and_cnt(&a, &b);
                                let uni = na + nb - inter;
                                if uni > 0 {
                                    sj += inter as f64 / uni as f64;
                                    cj += 1;
                                }
                                if na > 0 {
                                    sp += and_cnt(&shl1(&b), &a) as f64 / na as f64;
                                    cp += 1;
                                }
                                if nb > 0 {
                                    ss += and_cnt(&sa, &b) as f64 / nb as f64;
                                    cs += 1;
                                }
                            }
                            if cr > 0 {
                                vals[1] = (sr / cr as f64) as f32;
                            }
                            if cj > 0 {
                                vals[2] = (sj / cj as f64) as f32;
                            }
                            if cp > 0 {
                                vals[3] = (sp / cp as f64) as f32;
                            }
                            if cs > 0 {
                                vals[4] = (ss / cs as f64) as f32;
                            }
                            vals[5] = vals[3] - vals[4];
                        }
                    }
                    tail[p..p + 6].copy_from_slice(&vals);
                    p += 6;
                }
            }
        }
        debug_assert_eq!(p, N_GRP + N_NET_CMP + N_SPILL + N_GAP + N_SYNC);
    });

    Ok((valid_codes, out))
}

// ============================================================================
// 因子名
// ============================================================================

pub fn anneal_cross_peer_v3_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    let mut ch48 = Vec::with_capacity(N_CH);
    for d in DIR3_NAMES {
        for m in CORE_NAMES {
            ch48.push(format!("{d}_{m}"));
        }
    }
    let p1 = [
        "corr_mkt", "beta_mkt", "alpha_mkt", "r2_mkt", "res_early", "res_mid", "res_late",
        "res_late_minus_early", "res_late_minus_mid", "lead1_mkt", "lag1_mkt", "lead_minus_lag1",
        "lead3_mkt", "lag3_mkt", "lead_minus_lag3", "lead_minus_lag5", "corr_leaders",
        "corr_laggards", "corr_lead_minus_lag_grp", "rk_top5_share", "rk_top10_share",
        "rk_bot10_share", "rk_top10_max_streak", "rk_late_minus_early",
    ];
    let regimes = [
        "mkt_surge", "mkt_crash", "mkt_high_disp", "mkt_low_disp", "mkt_high_amt", "mkt_low_amt",
        "mkt_easy_ord", "mkt_hard_ord", "mkt_high_abr", "mkt_low_abr", "mkt_high_cnt", "mkt_low_cnt",
        "mkt_high_large", "mkt_low_large", "mkt_high_wait", "mkt_low_wait",
    ];
    let red_cols = red_col_names();
    let dummy = Array2::<f32>::ones((2, N_RED_COLS));
    let (_, red_names) = features::get_features_factors_rust_full(&dummy.view(), &red_cols, false);
    for s in 0..N_SCALE {
        for seg in 0..N_SEG {
            let tag = format!("acp3_{}_{}", SCALE_TAGS[s], SEG_TAGS[seg]);
            if s == 0 {
                for c in &ch48 {
                    for st in p1 {
                        names.push(format!("{tag}_mkt_{c}_{st}"));
                    }
                }
            }
            for &k in red_keep()[s].iter() {
                let n = red_names[k].replace("_corr_", &format!("_corr_{tag}_"));
                names.push(format!("{tag}_{n}"));
            }
            if s == 0 {
                for r in regimes {
                    for c in &ch48 {
                        names.push(format!("{tag}_mreg_{r}_{c}_mean_z"));
                        names.push(format!("{tag}_mreg_{r}_{c}_mean_rank"));
                    }
                }
            }
        }
    }
    for g in GROUP_NAMES {
        for w in WIN_TAGS {
            for c in &ch48 {
                names.push(format!("acp3_peer_{g}_{w}_{c}_grp_rk"));
                names.push(format!("acp3_peer_{g}_{w}_{c}_grp_minus_gl_rk"));
                names.push(format!("acp3_peer_{g}_{w}_{c}_grp_minus_gl_rk_pure"));
            }
        }
    }
    let ops = [
        "twin_mean", "self_minus_twin", "twin_zscore", "self_over_twin", "self_minus_wtwin",
        "twin_beat_ratio", "twin_std", "self_minus_twin_max", "self_minus_twin_min", "top_minus_bot_twin",
    ];
    for net in NET_NAMES {
        for w in WIN_TAGS {
            for d in 0..3usize {
                for &c in XCH_CORE.iter() {
                    let ch = &ch48[d * N_CORE + c];
                    for op in ops {
                        names.push(format!("acp3_{net}_{w}_{ch}_{op}"));
                    }
                }
            }
        }
    }
    let sops = [
        "peer_mean", "peer_wmean", "self_minus_peer", "peer_zscore", "self_minus_wpeer", "peer_std",
        "antipeer_mean", "peer_minus_antipeer",
    ];
    for net in NET_NAMES {
        for t in TARGET_NAMES {
            for op in sops {
                names.push(format!("acp3_{net}_spill_{t}_{op}"));
            }
        }
    }
    for net in COMP_NET_NAMES {
        for &t in COMP_TARGETS.iter() {
            for op in sops {
                names.push(format!("acp3_{net}_spill_{}_{op}", TARGET_NAMES[t]));
            }
        }
    }
    let gm = ["ret", "amt_surprise", "cnt_surprise", "large", "abr", "rv"];
    let gs = ["am_gap", "full_gap", "catchup", "pm_vs_late_gap"];
    for k in 0..N_STABLE {
        for m in gm {
            for s in gs {
                names.push(format!("acp3_{}_gap_{m}_{s}", NET_NAMES[k]));
            }
        }
    }
    let sstats = ["self_rate", "peer_rate", "jaccard", "peer_lead", "self_lead", "lead_diff"];
    for &k in SYNC_NETS.iter() {
        for &ch in SYNC_CH.iter() {
            for seg in SEG_TAGS {
                for st in sstats {
                    names.push(format!("acp3_sync_{}_{}_{}_{}", NET_NAMES[k], ch48[ch], seg, st));
                }
            }
        }
    }
    assert_eq!(names.len(), N_FACTORS);
    names
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

pub fn compute_anneal_cross_peer_v3_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction")?;
    let mids: Vec<Option<StockMidV3>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            extract_stock_mid_v3(&trades)
        })
        .collect();
    compute_cross_peer_v3_from_mids(date, &codes, mids, &profile_dir()?)
}

/// 柜员游戏扩展系列3 v3：读当天全市场逐笔与历史画像，返回 (股票代码, 展平因子值)，每股 42,268 维
#[pyfunction]
pub fn py_anneal_cross_peer_v3(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_anneal_cross_peer_v3_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_anneal_cross_peer_v3_names() -> Vec<String> {
    anneal_cross_peer_v3_names()
}

/// 第一轮：当天全市场每股 271 列中间数据，返回 (股票代码, 展平数值)
#[pyfunction]
pub fn py_anneal_cross_peer_v3_profile(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_anneal_cross_peer_v3_profile_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_anneal_cross_peer_v3_profile_names() -> Vec<String> {
    anneal_cross_peer_v3_profile_names()
}
