//! 一呼百应（yhyb）：tick 级事件响应网络横截面因子（纯 Rust）。
//!
//! # 思想
//! 把日频收益 lead-lag 网络下放到 tick 级：在逐笔成交 + 盘口快照上定义 23 种事件，
//! 对每只股票 A 度量"全市场其他股票 B 对 A 事件的响应距离"（fwd = B 响应 A，A 领先；
//! bwd = A 响应 B，B 领先），池化成响应速度/协同度因子，并用均匀零模型剔除共同驱动。
//!
//! # 范式（cross-section-pipeline 规范）
//! - 任务粒度 = 一天全市场；rayon 并行读全市场逐笔+盘口（read_trade_fast_inner / read_market_fast_inner）
//! - per-stock 事件检测 → 跨股票池化匹配 → per-stock 因子输出（N_FACTORS = 1380）
//! - v1（读盘）+ v2（Python 传数据）双入口，共享同一份纯计算核心
//!
//! # 因子布局（23 事件 × 4 时段 × 15）
//! 对每个 (事件 e, 时段 p)：`yhyb_e{02}_p{p}_rate` + fwd/bwd 两方向 × 7 个度量
//! [med, mean, hit, fast5, wmean, rmed, rhit]：
//! - med/mean：池化响应距离（秒）中位数/均值（1 秒桶直方图，精度 ±0.5s）
//! - hit：距离 ≤ hit_t_s 秒的占比（协同度，含无响应）
//! - fast5：最快 5% 距离均值（秒）
//! - wmean：按事件强度加权的距离均值（秒，标量累加，无权重直方图）
//! - rmed/rhit：med/hit 除以均匀零模型期望（净响应，剔除共同驱动）
//!
//! # 数据约定
//! - 只使用连续竞价：读取时 with_afternoon_adjust=true（过滤集合竞价+收盘后、下午前移90分钟）
//! - 成交方向直接用逐笔自带 flag（66=主买/83=主卖），不用 Lee-Ready
//! - 跨股匹配一律用 time_us（i64 微秒）；time_sec 是 f32，2025 年 epoch 附近精度仅 ±128s
//! - 涨跌停/停牌无逐笔数据自然无事件；T+1 不影响（只研究当日事件联动）
//!
//! # 零模型（可复现性）
//! 解析均匀零模型（无随机数）：B 的 m 个事件在时段 [0,S] 均匀分布时，
//! A 事件 t 到下一个 B 事件的分布 P(d>y) = ((S-t-y)/(S-t))^m。
//! null_med 用均值池化（系数 2^(-1/m) 全市场预计算成表，匹配循环内零 powf）；
//! null_hit 用均值近似（x̄ = A 事件到时段末距离的均值）。
//! 全确定性，逐日逐股可复现。
//!
//! # 性能（50 核约束）
//! 单日全市场 5914 股 × 1380 因子实测 ~55s（50 核热缓存）。优化：1 秒桶直方图统计、
//! fwd/bwd 一趟归并、零模型查表、wmean 标量累加、(A,事件) 并行粒度、空时段短路。

use crate::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;

// ============================================================================
// 常量
// ============================================================================

pub const N_EVENTS: usize = 23;
pub const N_PERIODS: usize = 4;
pub const N_METRICS: usize = 7;
/// 每 (事件, 时段)：rate + fwd/bwd × 7 度量
pub const N_FACTORS: usize = N_EVENTS * N_PERIODS * (1 + 2 * N_METRICS); // 1380

pub const EVENT_NAMES: [&str; N_EVENTS] = [
    "big", "big_buy", "big_sell", "sweep_buy", "sweep_sell", "ice", "jump", "jump_up",
    "jump_dn", "run_up", "run_dn", "vwap_up", "vwap_dn", "vwap_dev_up", "vwap_dev_dn",
    "imb_buy", "imb_sell", "depth", "wall", "spread", "retreat", "imp_buy", "imp_sell",
];

pub const METRIC_NAMES: [&str; N_METRICS] = ["med", "mean", "hit", "fast5", "wmean", "rmed", "rhit"];

/// 时段边界（adjust_afternoon 后连续时钟，相对当日 UTC 零点的秒；本地 09:30/10:30/12:00/13:27）
/// p0=全天, p1=早盘[9:30,10:30), p2=午盘[10:30,12:00), p3=尾盘[12:00,13:27)（=真实[13:30,14:57)）
const PERIOD_LO_S: [i64; N_PERIODS] = [0, 5400, 9000, 14400];
const PERIOD_HI_S: [i64; N_PERIODS] = [19620, 9000, 14400, 19620];

/// 事件时刻 t 所在日的"本地零点"（epoch 微秒）。
///
/// 注意：time_us 是"含 8h 偏移的 epoch"（本地时间作为伪 epoch，见 fast_csv_reader），
/// 因此本地零点 = UTC 零点 + 8h；若按 UTC 零点计算时段边界会整体错位 8 小时，
/// 导致零模型的时段末 S 早于事件时刻（x=0 → null_med=0 → rmed=inf）。
fn day_base(t: i64) -> i64 {
    (t / 86_400_000_000) * 86_400_000_000 + 28_800_000_000
}

// ============================================================================
// 参数（样例数据调参结果；from_vec 顺序即调参脚本传参顺序，17 个）
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct YhybParams {
    pub big_amt_wan: f64,  // 大单: 单笔成交额 ≥ big_amt_wan 万元（机构单，跨股可比）
    pub sweep_w_s: f64,    // 扫单窗口（秒）
    pub sweep_m: usize,    // 扫单: 窗口内同向大单数 ≥ m
    pub ice_w_s: f64,      // 冰山窗口（秒）
    pub ice_m: usize,      // 冰山: 窗口内同价大单数 ≥ m
    pub jump_q: f64,       // 跳变: |Δp| > max(当日 |Δp| 的 jump_q 分位, 2×tick)
    pub jump_win: usize,   // 跳变滚动σ窗口（笔数，保留备用）
    pub run_r: usize,      // 连涨/连跌: 连续同向 ≥ r 笔
    pub vwap_dev: f64,     // 偏离VWAP: |p-vwap|/vwap > θ（进入语义）
    pub imb_thr: f64,      // 失衡极值: |IOB| > θ（进入语义），IOB 用全档总量
    pub dep_thr: f64,      // 深度突变: |Δdepth|/depth > θ（10档可见量）
    pub wall_k: f64,       // 厚墙: max(买一,卖一) > wall_k × 当日中位一档量（进入语义）
    pub spread_ticks: f64, // 价差扩张: spread > θ×0.01（进入语义）
    pub ret_thr: f64,      // 撤单潮: 相邻快照消失量/前快照可见量 > θ
    pub imp_thr: f64,      // 冲击单: 单笔量 > imp_thr × 当时前5档总量
    pub hit_t_s: f64,      // 命中窗口（秒）
    pub fast_q: f64,       // 最快 5% 分位
}

impl Default for YhybParams {
    fn default() -> Self {
        YhybParams {
            big_amt_wan: 10.0,
            sweep_w_s: 1200.0,
            sweep_m: 2,
            ice_w_s: 180.0,
            ice_m: 2,
            jump_q: 0.99,
            jump_win: 100,
            run_r: 2,
            vwap_dev: 0.002,
            imb_thr: 0.5,
            dep_thr: 0.3,
            wall_k: 2.0,
            spread_ticks: 1.0,
            ret_thr: 0.5,
            imp_thr: 0.05,
            hit_t_s: 60.0,
            fast_q: 0.05,
        }
    }
}

impl YhybParams {
    /// 从固定顺序的 17 个参数构造（调参脚本传参用）。
    pub fn from_vec(v: &[f64]) -> YhybParams {
        assert_eq!(v.len(), 17, "yhyb 参数必须为 17 个，顺序见 YhybParams 字段");
        YhybParams {
            big_amt_wan: v[0],
            sweep_w_s: v[1],
            sweep_m: v[2] as usize,
            ice_w_s: v[3],
            ice_m: v[4] as usize,
            jump_q: v[5],
            jump_win: v[6] as usize,
            run_r: v[7] as usize,
            vwap_dev: v[8],
            imb_thr: v[9],
            dep_thr: v[10],
            wall_k: v[11],
            spread_ticks: v[12],
            ret_thr: v[13],
            imp_thr: v[14],
            hit_t_s: v[15],
            fast_q: v[16],
        }
    }
}

// ============================================================================
// 事件流
// ============================================================================

#[derive(Clone, Default)]
pub struct EvStream {
    pub t: Vec<i64>, // 事件时间（epoch 微秒，升序）
    pub w: Vec<f64>, // 事件强度权重
}

fn push(ev: &mut EvStream, t: i64, w: f64) {
    ev.t.push(t);
    ev.w.push(w);
}

/// 显式符号函数：+0.0/-0.0 一律返回 0（注意：f64::signum(+0.0)=+1.0，会污染 run 检测）。
fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ============================================================================
// 事件检测（per-stock，纯 Rust）
// ============================================================================

/// 逐笔事件（前 15 个）：big, big_buy, big_sell, sweep_buy, sweep_sell, ice,
/// jump, jump_up, jump_dn, run_up, run_dn, vwap_up, vwap_dn, vwap_dev_up, vwap_dev_dn
fn detect_trade_cols(t: &[i64], p: &[f64], v: &[f64], amt: &[f64], f: &[i32], prm: &YhybParams) -> [EvStream; 15] {
    let mut out: [EvStream; 15] = Default::default();
    let n = t.len();
    if n < 2 {
        return out;
    }
    // 当日中位单笔量（保留给 wall 类参考；大单本身用金额阈值）
    let mut v_sorted = v.to_vec();
    v_sorted.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    let _med_v = v_sorted[n / 2];
    // 大单：单笔成交额 ≥ big_amt_wan 万元（机构单，跨股票可比）
    let big_thr = prm.big_amt_wan * 1e4;
    let big: Vec<bool> = amt.iter().map(|&x| x >= big_thr).collect();
    for i in 0..n {
        if big[i] {
            push(&mut out[0], t[i], amt[i]); // big
            if f[i] == 66 {
                push(&mut out[1], t[i], amt[i]); // big_buy
            } else if f[i] == 83 {
                push(&mut out[2], t[i], amt[i]); // big_sell
            }
        }
    }
    // 扫单：窗口内同向大单数 ≥ sweep_m
    let sw_us = (prm.sweep_w_s * 1e6) as i64;
    for (eidx, flag) in [(3usize, 66i32), (4, 83)] {
        let mut tb: Vec<i64> = Vec::new();
        let mut wb: Vec<f64> = Vec::new();
        for i in 0..n {
            if big[i] && f[i] == flag {
                tb.push(t[i]);
                wb.push(amt[i]);
            }
        }
        if tb.len() >= prm.sweep_m {
            let mut j = 0usize;
            for i in 0..tb.len() {
                while tb[j] <= tb[i] - sw_us {
                    j += 1;
                }
                if i - j + 1 >= prm.sweep_m {
                    push(&mut out[eidx], tb[i], wb[i]);
                }
            }
        }
    }
    // 冰山：同价位大单在窗口内 ≥ ice_m
    {
        let mut idx: Vec<usize> = (0..n).filter(|&i| big[i]).collect();
        idx.sort_unstable_by(|&a, &b| p[a].total_cmp(&p[b]).then(t[a].cmp(&t[b])));
        let ice_us = (prm.ice_w_s * 1e6) as i64;
        let mut s = 0usize;
        while s < idx.len() {
            let mut e = s + 1;
            while e < idx.len() && p[idx[e]] == p[idx[s]] {
                e += 1;
            }
            if e - s >= prm.ice_m {
                let mut j = s;
                for i in s..e {
                    while t[idx[j]] <= t[idx[i]] - ice_us {
                        j += 1;
                    }
                    if i - j + 1 >= prm.ice_m {
                        push(&mut out[5], t[idx[i]], amt[idx[i]]);
                    }
                }
            }
            s = e;
        }
    }
    // 跳变：|Δp| > 当日 |Δp| 的 jump_q 分位
    // 分位阈值保证任意波动率的股票（含低价高波动股）都有事件 → 覆盖率可达 90%+；
    // 分位本身保证稀有性（top 1% 冲击），无需 2-tick 下限（下限会挡掉低价股）。
    if n > 2 {
        let mut dp_abs: Vec<f64> = (0..n - 1).map(|i| (p[i + 1] - p[i]).abs()).collect();
        let k = ((n as f64) * prm.jump_q).round().max(1.0) as usize;
        let k = k.min(dp_abs.len() - 1);
        dp_abs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
        let thr = dp_abs[k];
        for i in 0..n - 1 {
            let dp = p[i + 1] - p[i];
            if dp.abs() > thr {
                let wgt = dp.abs();
                push(&mut out[6], t[i + 1], wgt); // jump
                if dp > 0.0 {
                    push(&mut out[7], t[i + 1], wgt); // jump_up
                } else {
                    push(&mut out[8], t[i + 1], wgt); // jump_dn
                }
            }
        }
    }
    // 连涨/连跌：≥ run_r 笔同向（run 覆盖连续同向转移 [s, e]，事件时刻 = run 末笔）
    {
        let mut s = 0usize;
        let mut prev_sign = sgn(p[1] - p[0]);
        for i in 1..n - 1 {
            let sg = sgn(p[i + 1] - p[i]);
            if sg != prev_sign {
                if (i - s) >= prm.run_r && prev_sign != 0.0 {
                    let wgt = (p[i] - p[s]).abs();
                    if prev_sign > 0.0 {
                        push(&mut out[9], t[i], wgt); // run_up
                    } else {
                        push(&mut out[10], t[i], wgt); // run_dn
                    }
                }
                s = i;
                prev_sign = sg;
            }
        }
        // 收尾 run [s, n-2]
        if (n - 1 - s) >= prm.run_r && prev_sign != 0.0 {
            let wgt = (p[n - 1] - p[s]).abs();
            if prev_sign > 0.0 {
                push(&mut out[9], t[n - 1], wgt);
            } else {
                push(&mut out[10], t[n - 1], wgt);
            }
        }
    }
    // VWAP 与偏离
    let mut cum_v = 0.0f64;
    let mut cum_amt = 0.0f64;
    let mut vwap = vec![0.0f64; n];
    for i in 0..n {
        cum_v += v[i];
        cum_amt += amt[i];
        vwap[i] = cum_amt / cum_v.max(1e-9);
    }
    let mut prev_dev = p[0] - vwap[0];
    let mut in_zone = prev_dev.abs() > prm.vwap_dev * vwap[0];
    for i in 1..n {
        let dev = p[i] - vwap[i];
        if prev_dev <= 0.0 && dev > 0.0 {
            push(&mut out[11], t[i], dev.abs()); // vwap_up
        } else if prev_dev >= 0.0 && dev < 0.0 {
            push(&mut out[12], t[i], dev.abs()); // vwap_dn
        }
        let z = dev.abs() > prm.vwap_dev * vwap[i];
        if z && !in_zone {
            let wgt = dev.abs() / vwap[i];
            if dev > 0.0 {
                push(&mut out[13], t[i], wgt); // vwap_dev_up
            } else {
                push(&mut out[14], t[i], wgt); // vwap_dev_dn
            }
        }
        in_zone = z;
        prev_dev = dev;
    }
    out
}

/// 盘口事件（6 个）：imb_buy, imb_sell, depth, wall, spread, retreat
/// ask10/bid10: 10 档可见量（[f64;10]）。
/// IOB 用一档量（bid1/ask1）：一档变化快、买卖交替，覆盖率可达 90%+；
/// 全档 total IOB 在 A 股普遍偏卖（挂单结构），买方向覆盖率低（<40%），故不用。
fn detect_market_cols(
    mt: &[i64],
    ask1p: &[f64],
    bid1p: &[f64],
    ask10: &[[f64; 10]],
    bid10: &[[f64; 10]],
    prm: &YhybParams,
) -> [EvStream; 6] {
    let mut out: [EvStream; 6] = Default::default();
    let m = mt.len();
    if m < 2 {
        return out;
    }
    let depth10: Vec<f64> = (0..m).map(|i| ask10[i].iter().sum::<f64>() + bid10[i].iter().sum::<f64>()).collect();
    // IOB（全档）：进入语义
    let mut in_buy = false;
    let mut in_sell = false;
    // 厚墙：max(买一,卖一) > wall_k × 当日中位一档量
    let mut l1: Vec<f64> = (0..m).map(|i| bid10[i][0].max(ask10[i][0])).collect();
    let mut l1_sorted = l1.clone();
    l1_sorted.select_nth_unstable_by(m / 2, |a, b| a.total_cmp(b));
    let med_l1 = l1_sorted[m / 2];
    let mut in_wall = false;
    // 价差扩张：进入语义
    let mut in_spread = false;
    for i in 0..m {
        // IOB（一档）：进入语义
        let bsum = bid10[i][0];
        let asum = ask10[i][0];
        let denom = (bsum + asum).max(1e-9);
        let imb = (bsum - asum) / denom;
        if imb > prm.imb_thr && !in_buy {
            push(&mut out[0], mt[i], imb.abs());
            in_buy = true;
        } else if imb <= prm.imb_thr {
            in_buy = false;
        }
        if imb < -prm.imb_thr && !in_sell {
            push(&mut out[1], mt[i], imb.abs());
            in_sell = true;
        } else if imb >= -prm.imb_thr {
            in_sell = false;
        }
        // 厚墙
        let wall = l1[i] > prm.wall_k * med_l1;
        if wall && !in_wall {
            push(&mut out[3], mt[i], l1[i]);
            in_wall = true;
        } else if !wall {
            in_wall = false;
        }
        // 价差扩张（tick = 0.01）
        let spread = (ask1p[i] - bid1p[i]) / 0.01;
        let sp = spread > prm.spread_ticks;
        if sp && !in_spread {
            push(&mut out[4], mt[i], spread);
            in_spread = true;
        } else if !sp {
            in_spread = false;
        }
    }
    // 深度突变 + 撤单潮（相邻快照）
    for i in 1..m {
        let prev = depth10[i - 1];
        let cur = depth10[i];
        let dchg = (cur - prev).abs() / prev.max(1e-9);
        if dchg > prm.dep_thr {
            push(&mut out[2], mt[i], dchg);
        }
        let mut disc = 0.0f64;
        for k in 0..10 {
            disc += (bid10[i - 1][k] - bid10[i][k]).max(0.0);
            disc += (ask10[i - 1][k] - ask10[i][k]).max(0.0);
        }
        let rate = disc / prev.max(1e-9);
        if rate > prm.ret_thr {
            push(&mut out[5], mt[i], rate);
        }
    }
    out
}

/// 冲击单（2 个）：imp_buy, imp_sell —— 单笔量 > imp_thr × 当时前5档总量（双档）
fn detect_impact(
    t: &[i64],
    v: &[f64],
    amt: &[f64],
    f: &[i32],
    mt: &[i64],
    ask10: &[[f64; 10]],
    bid10: &[[f64; 10]],
    prm: &YhybParams,
) -> [EvStream; 2] {
    let mut out: [EvStream; 2] = Default::default();
    if t.is_empty() || mt.is_empty() {
        return out;
    }
    let mut j = 0usize;
    for i in 0..t.len() {
        while j + 1 < mt.len() && mt[j + 1] <= t[i] {
            j += 1;
        }
        if mt[j] > t[i] {
            continue; // 该笔早于第一份快照
        }
        let mut d5 = 0.0f64;
        for k in 0..5 {
            d5 += ask10[j][k] + bid10[j][k];
        }
        if v[i] > prm.imp_thr * d5 {
            let eidx = if f[i] == 66 { 0usize } else if f[i] == 83 { 1 } else { continue };
            push(&mut out[eidx], t[i], amt[i]);
        }
    }
    out
}

/// 全事件检测（23 个），输入原始 TradeRecord/MarketRecord（v1 读盘路径共用）。
fn detect_all(trades: &[TradeRecord], market: &[MarketRecord], prm: &YhybParams) -> [EvStream; N_EVENTS] {
    let n = trades.len();
    let mut t = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut amt = Vec::with_capacity(n);
    let mut f = Vec::with_capacity(n);
    for tr in trades {
        t.push(tr.time_us);
        p.push(tr.price as f64);
        v.push(tr.volume as f64);
        amt.push(tr.turnover as f64);
        f.push(tr.flag);
    }
    let m = market.len();
    let mut mt = Vec::with_capacity(m);
    let mut ask1p = Vec::with_capacity(m);
    let mut bid1p = Vec::with_capacity(m);
    let mut ask10: Vec<[f64; 10]> = Vec::with_capacity(m);
    let mut bid10: Vec<[f64; 10]> = Vec::with_capacity(m);
    for mr in market {
        mt.push(mr.time_us);
        ask1p.push(mr.ask_prcs[0] as f64);
        bid1p.push(mr.bid_prcs[0] as f64);
        ask10.push(mr.ask_vols.map(|x| x as f64));
        bid10.push(mr.bid_vols.map(|x| x as f64));
    }
    let tev = detect_trade_cols(&t, &p, &v, &amt, &f, prm);
    let mev = detect_market_cols(&mt, &ask1p, &bid1p, &ask10, &bid10, prm);
    let iev = detect_impact(&t, &v, &amt, &f, &mt, &ask10, &bid10, prm);
    let mut out: [EvStream; N_EVENTS] = Default::default();
    for (i, s) in tev.into_iter().enumerate() {
        out[i] = s;
    }
    for (i, s) in mev.into_iter().enumerate() {
        out[15 + i] = s;
    }
    for (i, s) in iev.into_iter().enumerate() {
        out[21 + i] = s;
    }
    out
}

// ============================================================================
// 横截面聚合（per-stock 因子）
// ============================================================================

/// 直方图桶数：调整后最长交易时段 14220 秒（09:30 ~ 13:27），1 秒一桶。
/// 距离统计用直方图代替精确排序：med/mean/fast5 精度 ±0.5s（桶中点）；
/// 强度加权度量用 wmean（加权均值，标量累加，无权重直方图随机写）。
const N_BUCKETS: usize = 14220;

#[derive(Default)]
struct Hists {
    fwd_cnt: Vec<u32>,
    bwd_cnt: Vec<u32>,
    fwd_wsum: f64,
    fwd_wmean: f64, // Σ(w·d)，除以 wsum 得加权均值（微秒）
    bwd_wsum: f64,
    bwd_wmean: f64,
}

impl Hists {
    fn new() -> Self {
        Hists {
            fwd_cnt: vec![0; N_BUCKETS],
            bwd_cnt: vec![0; N_BUCKETS],
            ..Default::default()
        }
    }
    fn clear(&mut self) {
        self.fwd_cnt.fill(0);
        self.bwd_cnt.fill(0);
        self.fwd_wsum = 0.0;
        self.fwd_wmean = 0.0;
        self.bwd_wsum = 0.0;
        self.bwd_wmean = 0.0;
    }
}

/// 直方图统计：[med, mean, hit, fast5, wmean]（秒）。两遍桶扫描。
fn hist_stats(cnt: &[u32], wsum: f64, wmean: f64, t_bucket: usize, fast_q: f64) -> [f64; 5] {
    let n: u64 = cnt.iter().map(|&c| c as u64).sum();
    if n == 0 {
        return [f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN];
    }
    let k = (((n as f64) * fast_q).round().max(1.0)) as u64;
    let half_n = n / 2;
    let mut acc = 0u64;
    let mut f_acc = 0u64;
    let mut f_sum = 0.0f64;
    let mut mean = 0.0f64;
    let mut hit = 0.0f64;
    let mut med = f64::NAN;
    let mut fast5 = f64::NAN;
    for b in 0..N_BUCKETS {
        let c = cnt[b] as u64;
        if c == 0 {
            continue;
        }
        let mid = b as f64 + 0.5;
        mean += c as f64 * mid;
        if b < t_bucket {
            hit += c as f64;
        }
        if med.is_nan() {
            acc += c;
            if acc > half_n {
                med = mid;
            }
        }
        if f_acc < k {
            let take = c.min(k - f_acc);
            f_sum += take as f64 * mid;
            f_acc += take;
            if f_acc >= k {
                fast5 = f_sum / k as f64;
            }
        }
    }
    let wmean = if wsum > 0.0 { wmean / wsum / 1e6 } else { f64::NAN };
    [med, mean / n as f64, hit / n as f64, fast5, wmean]
}

/// 均匀零模型（预计算版）：
/// - null_med：E[med] = mean over (B,i) of x_i·(1 - 2^(-1/m_B))
///   = mean over B of c_B·mean_i(x_i)，其中 c_B = 1 - 2^(-1/m_B) 只依赖 B 的事件数，
///   可全市场预计算 → 匹配循环内零 powf。
/// - null_hit：E[hit] ≈ mean over B of [1 - (1 - T/x̄_B)^m_B]（x̄_B = A 事件到时段末的均值，
///   用均值近似 Jensen 误差，README 已注明）。每 B 仅 1 次 powf。
#[derive(Clone)]
struct NullTable {
    /// c[bi][e][p] = 1 - 2^(-1/m)：B 股在时段 p 有 m 个事件 e 时的期望中位系数
    c: Vec<Vec<[f64; N_PERIODS]>>,
    /// m[bi][e][p] = 事件数（0 表示无事件）
    m: Vec<Vec<[f64; N_PERIODS]>>,
}

/// 预计算全市场零模型表（O(全市场事件数)，一次）。
fn build_null_table(streams: &[Option<[EvStream; N_EVENTS]>]) -> NullTable {
    let n = streams.len();
    let mut c = vec![vec![[0.0f64; N_PERIODS]; N_EVENTS]; n];
    let mut m = vec![vec![[0.0f64; N_PERIODS]; N_EVENTS]; n];
    for (bi, sb) in streams.iter().enumerate() {
        let Some(sb) = sb else { continue };
        for e in 0..N_EVENTS {
            let t = &sb[e].t;
            if t.is_empty() {
                continue;
            }
            let base = day_base(t[0]);
            for p in 0..N_PERIODS {
                let (lo, hi) = period_slice(t, base, p);
                let cnt = (hi - lo) as f64;
                if cnt > 0.0 {
                    m[bi][e][p] = cnt;
                    c[bi][e][p] = 1.0 - 2f64.powf(-1.0 / cnt);
                }
            }
        }
    }
    NullTable { c, m }
}

/// 单 (A, 事件 e) 的因子：4 时段 × 15 = 60 个值。
/// - 空时段短路：A 无事件时直接输出 rate=0 + 14 个 NaN（跳过清零与统计）
/// - 匹配：一趟归并同时算 fwd/bwd（A 事件序列 vs B 事件序列，双指针）
/// - 收集：1 秒桶直方图（Hists 复用于 4 个时段，无 Vec 分配）
/// - 统计：hist_stats 一次桶扫描
/// - 零模型：查预计算表（null_med 零 powf；null_hit 每 B 1 次 powf）
fn agg_one(
    streams: &[Option<[EvStream; N_EVENTS]>],
    null_t: &NullTable,
    ai: usize,
    e: usize,
    prm: &YhybParams,
) -> Option<Vec<f64>> {
    let sa = streams[ai].as_ref()?;
    let (ta, wa) = (&sa[e].t, &sa[e].w);
    let t_us = prm.hit_t_s * 1e6;
    let t_bucket = (t_us / 1e6) as usize;
    let mut h = Hists::new();
    let mut out = Vec::with_capacity(N_PERIODS * 15);
    for p in 0..N_PERIODS {
        let (alo, ahi, base) = if ta.is_empty() {
            (0usize, 0usize, 0i64)
        } else {
            let base = day_base(ta[0]);
            let (lo, hi) = period_slice(ta, base, p);
            (lo, hi, base)
        };
        let rate = (ahi - alo) as f64;
        if ahi == alo {
            // 空时段短路：只输出 rate，其余 14 个 NaN
            out.push(rate);
            for _ in 0..14 {
                out.push(f64::NAN);
            }
            continue;
        }
        h.clear();
        let s_us = base + PERIOD_HI_S[p] * 1_000_000;
        // A 事件到时段末的距离均值（与 B 无关，提前算一次）
        let mut sum_x = 0.0f64;
        for &a in &ta[alo..ahi] {
            sum_x += ((s_us - a).max(0)) as f64;
        }
        let x_bar = sum_x / (ahi - alo) as f64;
        let mut null_med_acc = 0.0f64;
        let mut null_hit_acc = 0.0f64;
        let mut null_n = 0usize;
        for (bi, sb) in streams.iter().enumerate() {
            if bi == ai {
                continue;
            }
            let Some(sb) = sb else { continue };
            let tb = &sb[e].t;
            let (blo, bhi) = period_slice(tb, base, p);
            if blo == bhi {
                continue;
            }
            // 一趟归并：fwd（A 事件后 B 的下一次）+ bwd（A 事件前 B 最近一次）
            let mut j = blo;
            for i in alo..ahi {
                let a = ta[i];
                let wi = wa[i];
                while j < bhi && tb[j] <= a {
                    j += 1;
                }
                if j > blo {
                    let d = a - tb[j - 1];
                    let bucket = (d / 1_000_000) as usize;
                    if bucket < N_BUCKETS {
                        h.bwd_cnt[bucket] += 1;
                        h.bwd_wsum += wi;
                        h.bwd_wmean += wi * d as f64;
                    }
                }
                if j < bhi {
                    let d = tb[j] - a;
                    let bucket = (d / 1_000_000) as usize;
                    if bucket < N_BUCKETS {
                        h.fwd_cnt[bucket] += 1;
                        h.fwd_wsum += wi;
                        h.fwd_wmean += wi * d as f64;
                    }
                }
            }
            // 零模型（查表 + 均值近似）
            let m_b = null_t.m[bi][e][p];
            let c_b = null_t.c[bi][e][p];
            null_med_acc += c_b * x_bar;
            null_hit_acc += 1.0 - (1.0 - t_us / x_bar).clamp(0.0, 1.0).powf(m_b);
            null_n += 1;
        }
        let fs = hist_stats(&h.fwd_cnt, h.fwd_wsum, h.fwd_wmean, t_bucket, prm.fast_q);
        let bs = hist_stats(&h.bwd_cnt, h.bwd_wsum, h.bwd_wmean, t_bucket, prm.fast_q);
        let null_med = if null_n > 0 { null_med_acc / null_n as f64 / 1e6 } else { f64::NAN };
        let null_hit = if null_n > 0 { null_hit_acc / null_n as f64 } else { f64::NAN };
        out.push(rate);
        for dd in [&fs, &bs] {
            out.push(dd[0]);
            out.push(dd[1]);
            out.push(dd[2]);
            out.push(dd[3]);
            out.push(dd[4]);
            out.push(dd[0] / null_med); // rmed
            out.push(dd[2] / null_hit); // rhit
        }
    }
    Some(out)
}

/// 时段切片（t 有序）：返回 [lo, hi) 落入时段 p 的下标区间。
fn period_slice(t: &[i64], base: i64, p: usize) -> (usize, usize) {
    if p == 0 {
        return (0, t.len());
    }
    let lo = base + PERIOD_LO_S[p] * 1_000_000;
    let hi = base + PERIOD_HI_S[p] * 1_000_000;
    let s = t.partition_point(|&x| x < lo);
    let e = t.partition_point(|&x| x < hi);
    (s, e)
}

/// 单 (A, 事件 e) 的因子：4 时段 × 15 = 60 个值。
/// - 匹配：一趟归并同时算 fwd/bwd（A 事件序列 vs B 事件序列，双指针）
/// - 收集：1 秒桶直方图（无 Vec 分配）
/// 列出某天某子目录下所有股票代码（文件名 `{code}_{date}_{type}.csv`）。
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

/// 从预加载的全市场事件流聚合因子（v1/v2 共同核心）。
/// 并行粒度 (A, 事件)：5900×23 = 13.6 万个小任务（任务内 4 时段复用直方图）；
/// 零模型系数表全市场预计算一次（build_null_table），匹配循环内零 powf。
fn compute_from_streams(
    codes: &[String],
    streams: &[Option<[EvStream; N_EVENTS]>],
    prm: &YhybParams,
) -> (Vec<String>, Vec<f32>) {
    let n_stocks = codes.len();
    let null_t = build_null_table(streams);
    let results: Vec<Option<Vec<f64>>> = (0..n_stocks * N_EVENTS)
        .into_par_iter()
        .map(|idx| agg_one(streams, &null_t, idx / N_EVENTS, idx % N_EVENTS, prm))
        .collect();
    let mut out_codes = Vec::new();
    let mut vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for ai in 0..n_stocks {
        let mut row = Vec::with_capacity(N_FACTORS);
        let mut ok = true;
        for e in 0..N_EVENTS {
            match &results[ai * N_EVENTS + e] {
                Some(v) => row.extend_from_slice(v),
                None => ok = false,
            }
        }
        if ok && row.len() == N_FACTORS {
            out_codes.push(codes[ai].clone());
            vals.extend(row.iter().map(|&x| x as f32));
        }
    }
    (out_codes, vals)
}

/// v1 入口（读盘）：读全市场 → 事件检测 → 横截面池化 → (codes, vals)。
pub fn compute_yhyb_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    compute_yhyb_full_with_params(date, &YhybParams::default())
}

pub fn compute_yhyb_full_with_params(date: i64, prm: &YhybParams) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction");
    let streams: Vec<Option<[EvStream; N_EVENTS]>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, 8 * 1024 * 1024).ok()?;
            if trades.is_empty() {
                return None;
            }
            let market =
                read_market_fast_inner(code, date, false, true, 8 * 1024 * 1024).unwrap_or_default();
            Some(detect_all(&trades, &market, prm))
        })
        .collect();
    Ok(compute_from_streams(&codes, &streams, prm))
}

/// 因子名（与 N_FACTORS 严格对齐，单一源）。
pub fn yhyb_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    for e in 0..N_EVENTS {
        for p in 0..N_PERIODS {
            names.push(format!("yhyb_e{e:02}_p{p}_rate"));
            for dir in ["fwd", "bwd"] {
                for mt in METRIC_NAMES {
                    names.push(format!("yhyb_e{e:02}_p{p}_{mt}_{dir}"));
                }
            }
        }
    }
    names
}

// ============================================================================
// Python 入口（薄包装）
// ============================================================================

/// Python 单日调试（v1，默认参数）：返回 (codes, vals)。
#[pyfunction]
pub fn py_yhyb(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_yhyb_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 单日调试（v1，自定义参数，17 个按 YhybParams 顺序）。
#[pyfunction]
pub fn py_yhyb_params(py: Python<'_>, date: i64, params: Vec<f64>) -> PyResult<(Vec<String>, Vec<f32>)> {
    let prm = YhybParams::from_vec(&params);
    compute_yhyb_full_with_params(date, &prm)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_yhyb_names() -> Vec<String> {
    yhyb_names()
}

/// v2 入口（Python 传数据，样例验证/调参）：
/// - trade_arrays：每股 (n_i, 8) [time_us, time_sec, price, volume, turnover, flag, bid_order, ask_order]
/// - market_arrays：每股 (m_i, 25) [time_us, total_ask_vol, total_bid_vol, ask_prc1, bid_prc1,
///   ask_vol1..10, bid_vol1..10]
/// - params：可选 17 个参数（默认 YhybParams::default()）
#[pyfunction]
pub fn py_yhyb_from_data(
    _py: Python<'_>,
    codes: Vec<String>,
    trade_arrays: Vec<PyReadonlyArray2<f64>>,
    market_arrays: Vec<PyReadonlyArray2<f64>>,
    params: Option<Vec<f64>>,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    if codes.len() != trade_arrays.len() || codes.len() != market_arrays.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "codes.len()={} trade={} market={} 不一致",
            codes.len(),
            trade_arrays.len(),
            market_arrays.len()
        )));
    }
    let prm = params.map(|v| YhybParams::from_vec(&v)).unwrap_or_default();
    let streams: Vec<Option<[EvStream; N_EVENTS]>> = trade_arrays
        .iter()
        .zip(market_arrays.iter())
        .map(|(ta, ma)| {
            let a = ta.as_array();
            let b = ma.as_array();
            if a.nrows() == 0 {
                return None; // 无逐笔 → 无事件；盘口可缺（市场事件自然为空）
            }
            let n = a.nrows();
            let mut t = Vec::with_capacity(n);
            let mut p = Vec::with_capacity(n);
            let mut v = Vec::with_capacity(n);
            let mut amt = Vec::with_capacity(n);
            let mut f = Vec::with_capacity(n);
            for i in 0..n {
                // (n, 8): [time_us, time_sec, price, volume, turnover, flag, bid_order, ask_order]
                t.push(a[[i, 0]] as i64);
                p.push(a[[i, 2]]);
                v.push(a[[i, 3]]);
                amt.push(a[[i, 4]]);
                f.push(a[[i, 5]] as i32);
            }
            let m = b.nrows();
            let mut mt = Vec::with_capacity(m);
            let mut ask1p = Vec::with_capacity(m);
            let mut bid1p = Vec::with_capacity(m);
            let mut ask10: Vec<[f64; 10]> = Vec::with_capacity(m);
            let mut bid10: Vec<[f64; 10]> = Vec::with_capacity(m);
            for i in 0..m {
                mt.push(b[[i, 0]] as i64);
                ask1p.push(b[[i, 3]]);
                bid1p.push(b[[i, 4]]);
                let mut aa = [0.0f64; 10];
                let mut bb = [0.0f64; 10];
                for k in 0..10 {
                    aa[k] = b[[i, 5 + k]];
                    bb[k] = b[[i, 15 + k]];
                }
                ask10.push(aa);
                bid10.push(bb);
            }
            let tev = detect_trade_cols(&t, &p, &v, &amt, &f, &prm);
            let mev = detect_market_cols(&mt, &ask1p, &bid1p, &ask10, &bid10, &prm);
            let iev = detect_impact(&t, &v, &amt, &f, &mt, &ask10, &bid10, &prm);
            let mut out: [EvStream; N_EVENTS] = Default::default();
            for (i, s) in tev.into_iter().enumerate() {
                out[i] = s;
            }
            for (i, s) in mev.into_iter().enumerate() {
                out[15 + i] = s;
            }
            for (i, s) in iev.into_iter().enumerate() {
                out[21 + i] = s;
            }
            Some(out)
        })
        .collect();
    Ok(compute_from_streams(&codes, &streams, &prm))
}

/// Python 调试：单股单日事件时间线（事件名 -> (时间us, 权重)），供单例/验证。
#[pyfunction]
pub fn py_yhyb_events(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<(String, (Vec<i64>, Vec<f64>))>> {
    let trades = read_trade_fast_inner(code, date, false, true, usize::MAX)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))?;
    let market = read_market_fast_inner(code, date, false, true, usize::MAX).unwrap_or_default();
    let ev = detect_all(&trades, &market, &YhybParams::default());
    Ok(EVENT_NAMES
        .iter()
        .zip(ev.into_iter())
        .map(|(name, s)| (name.to_string(), (s.t, s.w)))
        .collect())
}

// ============================================================================
// 测试
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_count() {
        assert_eq!(yhyb_names().len(), N_FACTORS);
        assert_eq!(N_FACTORS, 1380);
    }

    #[test]
    fn test_params_vec_roundtrip() {
        let p = YhybParams::default();
        let v = [
            p.big_amt_wan, p.sweep_w_s, p.sweep_m as f64, p.ice_w_s, p.ice_m as f64, p.jump_k,
            p.jump_win as f64, p.run_r as f64, p.vwap_dev, p.imb_thr, p.dep_thr, p.wall_k,
            p.spread_ticks, p.ret_thr, p.imp_thr, p.hit_t_s, p.fast_q,
        ];
        let p2 = YhybParams::from_vec(&v);
        assert_eq!(p2.big_amt_wan, p.big_amt_wan);
        assert_eq!(p2.sweep_m, p.sweep_m);
    }

    #[test]
    fn test_period_slice() {
        let base = day_base(1_735_637_400_000_000); // 2024-12-31 某时刻
        let t: Vec<i64> = (0..10).map(|i| base + (5400 + i * 60) * 1_000_000).collect();
        // 全部在 p1 [5400,9000)
        let (lo, hi) = period_slice(&t, base, 1);
        assert_eq!((lo, hi), (0, 10));
        // p2 [9000,14400)
        let (lo, hi) = period_slice(&t, base, 2);
        assert_eq!((lo, hi), (0, 0));
        let (lo, hi) = period_slice(&t, base, 0);
        assert_eq!((lo, hi), (0, 10));
    }

    #[test]
    fn test_compute_stats_basic() {
        let mut h = Hists::new();
        // 距离 1,2,3,4,5 秒各一条，权重均匀
        for b in [1usize, 2, 3, 4, 5] {
            h.fwd_cnt[b] += 1;
        }
        h.fwd_wsum = 5.0;
        h.fwd_wmean = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) * 1e6; // Σw·d
        let s = hist_stats(&h.fwd_cnt, h.fwd_wsum, h.fwd_wmean, 2, 0.05);
        assert!((s[0] - 3.5).abs() < 1e-9); // med（桶中点）
        assert!((s[1] - 3.5).abs() < 1e-9); // mean
        assert!((s[2] - 0.4).abs() < 1e-9); // hit（桶 0..2 = 距离 ≤2s）
        assert!((s[3] - 1.5).abs() < 1e-9); // fast5
        assert!((s[4] - 3.0).abs() < 1e-9); // wmean（均匀权重 = 均值）
    }

    #[test]
    fn test_null_table_poisson_consistency() {
        // 高事件率下（m 大），均匀零模型期望 ≈ S/(m+1)（与泊松一致）
        let base = day_base(0);
        let s_us = base + 19_620_000_000;
        let mut evs: [EvStream; N_EVENTS] = Default::default();
        evs[0].t.push(base + 5_400_000_000);
        let streams = vec![Some(evs)];
        let nt = build_null_table(&streams);
        assert!((nt.m[0][0][0] - 1.0).abs() < 1e-9);
        // c = 1 - 2^(-1/1) = 0.5
        assert!((nt.c[0][0][0] - 0.5).abs() < 1e-9);
        let _ = s_us;
    }
}
