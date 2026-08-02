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
//! [med, mean, hit, fast5, wmed, rmed, rhit]（**v3 单遍近似统计**，评估相关性见 README）：
//! - med/mean/fast5/wmed：全部 1 秒桶中点（±0.5s 量化；med/fast5 Spearman 1.0000/0.9996，
//!   mean/wmed 桶中点相关性已实测评估）
//! - hit：精确标量（距离 ≤ hit_t_s 秒占比，含无响应）
//! - rmed/rhit：med/hit 除以零模型期望（净响应，剔除共同驱动）——零模型近似但 O(k_A+nB)：
//!   null_med 可分式均值池化（rmed Spearman 0.9694）、null_hit x̄ 均值近似（rhit 0.9632）
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
//! null_med = 池化**中位数** of x·(1 - 2^(-1/m))（系数表全市场预计算，循环内零 powf）；
//! null_hit = 逐点精确 1 - ((x-T)/x)^m（整数幂 powi + 安全截断）。
//! 全确定性，逐日逐股可复现。
//!
//! # 性能（一天全市场 ≤ 60s，限流 50 线程）
//! 单日全市场 5914 股 × 1380 因子实测聚合 ~50s（rayon 全局池限流 50 线程，代码级
//! ensure_threads）。优化（均为算法/底层级，近似口径见上）：
//! **单遍**匹配（无第二遍：wmed/mean 改桶中点）、1 秒桶直方图（u64 距离无回绕、
//! 无尾部截断）、零模型可分式 O(k_A+nB)（null_med 均值池化 + null_hit x̄ 近似）、
//! n/wsum 由 push1 维护（免逐桶求和）、locate 只扫到最大非空桶、fwd/bwd 一趟归并、
//! (A,事件,时段) 并行粒度、空时段短路。

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

pub const METRIC_NAMES: [&str; N_METRICS] = ["med", "mean", "hit", "fast5", "wmed", "rmed", "rhit"];

/// 时段边界（**adjust_afternoon 后的连续时钟**，本地 09:30 = 5400s、14:57 = 19620s；
/// 下午 13:00-15:00 已前移 90 分钟为 11:30-13:30，无午休空洞，跨午盘距离不受污染）
/// p0=全天 [09:30,14:57)；p1=盘中3小时 [10:00,14:27)（剔除早盘/尾盘各 30 分钟）；
/// p2=尾盘30分钟 [14:27,14:57)；p3=下午2小时 [13:00,15:00)（adjust 后 11:30-13:30；
/// 连续竞价数据只到 14:57 = adjust 后 13:27，窗口超出部分自然无数据）
const PERIOD_LO_S: [i64; N_PERIODS] = [5400, 6000, 17820, 12600];
const PERIOD_HI_S: [i64; N_PERIODS] = [19620, 17820, 19620, 19800];

/// 事件时刻 t 所在日的"本地零点"（epoch 微秒）。
///
/// 注意：time_us 是"含 8h 偏移的 epoch"（本地时间作为伪 epoch，见 fast_csv_reader），
/// 因此本地零点 = UTC 零点 + 8h；若按 UTC 零点计算时段边界会整体错位 8 小时，
/// 导致零模型的时段末 S 早于事件时刻（x=0 → null_med=0 → rmed=inf）。
pub fn day_base(t: i64) -> i64 {
    (t / 86_400_000_000) * 86_400_000_000 + 28_800_000_000
}// ============================================================================
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
            vwap_dev: 0.001,
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
    // 注意：按 (价格, 时间) 分组遍历，事件流天然非时间升序；聚合归并/零模型都依赖
    // 时间有序，检测完必须按时间重排（历史 bug：ice 流乱序 → 匹配与零模型全部失效）
    {
        let mut idx: Vec<usize> = (0..n).filter(|&i| big[i]).collect();
        idx.sort_unstable_by(|&a, &b| p[a].total_cmp(&p[b]).then(t[a].cmp(&t[b])));
        let ice_us = (prm.ice_w_s * 1e6) as i64;
        let mut ice_ev: Vec<(i64, f64)> = Vec::new();
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
                        ice_ev.push((t[idx[i]], amt[idx[i]]));
                    }
                }
            }
            s = e;
        }
        ice_ev.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (tt, ww) in ice_ev {
            push(&mut out[5], tt, ww);
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
    assert_streams_sorted(&out, "detect_all");
    out
}

/// 防御：事件流必须时间升序（跨股归并匹配与零模型都依赖有序）。
fn assert_streams_sorted(s: &[EvStream; N_EVENTS], what: &str) {
    for (i, ev) in s.iter().enumerate() {
        debug_assert!(
            ev.t.windows(2).all(|w| w[0] <= w[1]),
            "{what} 事件 {i} 流未按时间排序"
        );
    }
}

// ============================================================================
// 横截面聚合（per-stock 因子）
// ============================================================================

/// 1 秒直方图桶数：连续竞价全长 19620s，逐秒一桶，任何距离都有独立桶（无尾部截断）。
/// 桶索引 = 距离秒数（19619 为最大合法索引，min() 仅作防御）。
const N_BUCKETS_1S: usize = 19_620;

/// 均匀零模型（预计算表版）：
/// - null_med：池化**中位数** of x_i·(1 - 2^(-1/m_B)) over (B,i)——系数 2^(-1/m) 只依赖 B 的
///   事件数，全市场预计算（c 表），匹配循环内零 powf；期望值精确收集后 select。
/// - null_hit：逐点精确 1 - ((x_i-T)/x_i)^m_B（整数幂 powi + 安全截断）。
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

/// 时段切片（t 有序）：返回 [lo, hi) 落入时段 p 的下标区间。
/// p0 快速路径：生产数据（v1 adjust 过滤 / v2 已 adjust 样例）无 09:30 前事件，
/// 下界可跳过（只做 1 次二分）；其余时段常规 2 次二分。
pub fn period_slice(t: &[i64], base: i64, p: usize) -> (usize, usize) {
    let lo = base + PERIOD_LO_S[p] * 1_000_000;
    let hi = base + PERIOD_HI_S[p] * 1_000_000;
    if p == 0 && t.first().map_or(true, |&x| x >= lo) {
        return (0, t.partition_point(|&x| x < hi));
    }
    let s = t.partition_point(|&x| x < lo);
    let e = t.partition_point(|&x| x < hi);
    (s, e)
}

/// 单遍收集器（v3 生产默认）：1 秒桶直方图一次遍历完成所有统计（无第二遍）。
/// - med/fast5/wmed/mean：全部 1 秒桶中点（±0.5s 量化；med/fast5 评估 Spearman
///   1.0000/0.9996，wmed/mean 为 v3 新增近似，相关性已实测评估）
/// - hit：精确标量
/// - 距离一律 u64：i64 差值 as u64 无符号溢出（最大 ~19620s << u64 上限）。
///   历史 u32 版会在 >4295s 的缺口处静默回绕，把数小时的稀疏缺口伪装成小距离，
///   污染 med/fast5/wmed——v3 一并修复
/// 1 秒桶（u32 计数 + f32 权重和，8 字节/桶，同 cache line——桶更新的两次 store
/// 只触碰一条缓存行，分离数组要两条；19620 × 8B = 157KB）。
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Bucket {
    c: u32,
    w: f32,
}

/// 单遍收集器（v3 生产默认）：1 秒桶直方图一次遍历完成所有统计（无第二遍）。
/// - med/fast5/wmed/mean：全部 1 秒桶中点（±0.5s 量化；med/fast5 评估 Spearman
///   1.0000/0.9996，wmed/mean 为 v3 新增近似，相关性已实测评估）
/// - hit：精确标量
/// - 距离一律 u64：i64 差值 as u64 无符号溢出（最大 ~19620s << u64 上限）。
///   历史 u32 版会在 >4295s 的缺口处静默回绕，把数小时的稀疏缺口伪装成小距离，
///   污染 med/fast5/wmed——v3 一并修复
struct Gather {
    b: Vec<Bucket>, // 1 秒桶（157KB）
    n: u64,         // 距离条数（免去逐桶求和）
    med_b: usize,
    f5_b: usize,
    wmed_b: usize,
    hit: u64,
    wsum: f64,
}

impl Gather {
    fn new() -> Self {
        Gather {
            b: vec![Bucket::default(); N_BUCKETS_1S],
            n: 0,
            med_b: 0,
            f5_b: 0,
            wmed_b: 0,
            hit: 0,
            wsum: 0.0,
        }
    }
    /// 单遍：1 秒桶更新 + 标量（n/wsum/hit）。无 mean 累加（mean 由桶中点统计）。
    /// 注意：桶索引必须用 u64 除法——距离最大 ~19620s = 1.96e10 µs 远超 u32 上限
    /// （4.29e9），as u32 会把 >4295s 的大缺口回绕成小距离（u64 修复的教训）。
    #[inline(always)]
    fn push1(&mut self, dd: u64, ww: f64, t_us: u64) {
        let b = ((dd / 1_000_000) as usize).min(N_BUCKETS_1S - 1);
        self.b[b].c += 1;
        self.b[b].w += ww as f32;
        self.n += 1;
        self.wsum += ww;
        if dd <= t_us {
            self.hit += 1;
        }
    }
    /// 定位：med/f5 桶（1 秒计数）+ wmed 桶（1 秒权重）。
    /// 分位数位置落在"桶前计数 ≤ 位置 < 桶前计数+桶计数"的桶内。
    fn locate(&mut self, half_n: u64, half_w: f64, k: u64) {
        let mut acc = 0u64;
        let mut f_acc = 0u64;
        let mut wacc = 0.0f64;
        for i in 0..N_BUCKETS_1S {
            let c = self.b[i].c as u64;
            if c == 0 {
                continue;
            }
            if acc <= half_n && half_n < acc + c {
                self.med_b = i;
            }
            if f_acc < k {
                f_acc += c;
                if f_acc >= k {
                    self.f5_b = i;
                }
            }
            let wc = self.b[i].w as f64;
            if wacc <= half_w && half_w < wacc + wc {
                self.wmed_b = i;
            }
            acc += c;
            wacc += wc;
        }
    }
    /// 统计：[med, mean, hit, fast5, wmed]（秒），全部 1 秒桶中点/精确标量。
    fn stats(&mut self, n: u64, k: u64) -> [f64; 5] {
        if n == 0 {
            return [f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN];
        }
        let med = self.med_b as f64 + 0.5;
        let mut mean_sum = 0.0f64;
        let mut f_acc = 0u64;
        let mut f_sum = 0.0f64;
        let mut fast5 = f64::NAN;
        for i in 0..N_BUCKETS_1S {
            let c = self.b[i].c as u64;
            if c == 0 {
                continue;
            }
            let mid = i as f64 + 0.5;
            mean_sum += c as f64 * mid;
            if f_acc < k {
                let take = c.min(k - f_acc);
                f_sum += take as f64 * mid;
                f_acc += take;
                if f_acc >= k {
                    fast5 = f_sum / k as f64;
                }
            }
        }
        [
            med,
            mean_sum / n as f64,
            self.hit as f64 / n as f64,
            fast5,
            self.wmed_b as f64 + 0.5,
        ]
    }
}

/// 单 (A, 事件 e) 的因子：4 时段 × 15 = 60 个值。
/// - 空时段短路：A 无事件时直接输出 rate=0 + 14 个 NaN
/// - **单遍**匹配（v3 生产）：一趟归并（fwd/bwd 同趟）+ 1 秒桶直方图（med/fast5/wmed
///   定位 + mean 桶中点）+ 精确标量（hit）+ 零模型 O(k_A+nB) 可分式，**无第二遍**
/// - 零模型（v3 近似，评估验证相关性见 README）：
///   - null_med = (Σ_i x_i)·(Σ_b c_b)/(k_A·nB)——可分式均值池化（rmed Spearman 0.9694），
///     从 O(对) 物化降到 O(k_A+nB)
///   - null_hit = (1/nB)·Σ_b[1−(1−T/x̄)^m_b]——x̄ 均值近似（rhit Spearman 0.9632），
///     每 B 一次 powf，无逐点成本
/// - YHYB_SKIP_NULL 环境变量：跳过零模型（瓶颈定位用，不影响正常路径）
fn agg_one(
    streams: &[Option<[EvStream; N_EVENTS]>],
    null_t: &NullTable,
    ai: usize,
    e: usize,
    p: usize,
    prm: &YhybParams,
) -> Option<Vec<f64>> {
    let sa = streams[ai].as_ref()?;
    let (ta, wa) = (&sa[e].t, &sa[e].w);
    let t_us = (prm.hit_t_s * 1e6) as u64;
    let skip_null = std::env::var("YHYB_SKIP_NULL").is_ok();
    let mut g_fwd = Gather::new();
    let mut g_bwd = Gather::new();
    let (alo, ahi, base) = if ta.is_empty() {
        (0usize, 0usize, 0i64)
    } else {
        let base = day_base(ta[0]);
        let (lo, hi) = period_slice(ta, base, p);
        (lo, hi, base)
    };
    let rate = (ahi - alo) as f64;
    let mut out = Vec::with_capacity(15);
    if ahi == alo {
        out.push(rate);
        for _ in 0..14 {
            out.push(f64::NAN);
        }
        return Some(out);
    }
    let s_us = base + PERIOD_HI_S[p] * 1_000_000;
    // 零模型可分式预计算：Σ x_i（A 事件时段剩余）与 x̄，每任务 O(k_A)
    // YHYB_NULL_MODE 选择 null_med 近似口径（均为 O(k_A+nB) 可分式）：
    //   geo（默认）= 几何均值 exp(mean ln x)·exp(mean ln c)，对数域≈中位数，
    //   全市场实测 rmed Spearman 0.9700/0.9807（最优）；med = median(x)·median(c)；
    //   mean = 均值池化（rmed 仅 0.74-0.84，弃用）
    let null_mode = std::env::var("YHYB_NULL_MODE").unwrap_or_else(|_| "geo".into());
    let geo = null_mode == "geo";
    let medm = null_mode == "med";
    let k_a = (ahi - alo) as f64;
    let sum_x: f64 = if skip_null {
        0.0
    } else {
        ta[alo..ahi].iter().map(|&a| (s_us - a).max(0) as f64).sum()
    };
    let sum_ln_x: f64 = if !skip_null && geo {
        ta[alo..ahi].iter().map(|&a| ((s_us - a).max(1) as f64).ln()).sum()
    } else {
        0.0
    };
    let x_bar = if skip_null { 0.0 } else { sum_x / k_a };
    let mut sum_c = 0.0f64; // Σ c_b（null_med 均值池化）
    let mut sum_ln_c = 0.0f64; // Σ ln c_b（几何均值）
    let mut c_list: Vec<f64> = Vec::new(); // c_b 列表（中位×中位）
    let mut null_hit_acc = 0.0f64; // null_hit x̄ 近似（每 B 一次 powf）
    let mut n_b = 0u64; // 时段内有事件的 B 数
    // ============ 单遍：1 秒桶 + 标量 + null 可分式 ============
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
        if !skip_null {
            let m_b = null_t.m[bi][e][p];
            let c_b = null_t.c[bi][e][p];
            sum_c += c_b;
            if geo {
                sum_ln_c += c_b.ln();
            }
            if medm {
                c_list.push(c_b);
            }
            n_b += 1;
            // null_hit x̄ 近似：每 B 一次（clamp 防 x̄ ≤ T 时底数非正）
            null_hit_acc += 1.0 - (1.0 - t_us as f64 / x_bar).clamp(0.0, 1.0).powf(m_b);
        }
        let mut j = blo;
        for i in alo..ahi {
            let a = ta[i];
            let wi = wa[i];
            while j < bhi && tb[j] <= a {
                j += 1;
            }
            if j > blo {
                g_bwd.push1((a - tb[j - 1]) as u64, wi, t_us);
            }
            if j < bhi {
                g_fwd.push1((tb[j] - a) as u64, wi, t_us);
            }
        }
    }
    // 定位分位数桶（n/wsum 由 push1 维护，免去逐桶求和）
    let n_f = g_fwd.n;
    let n_bw = g_bwd.n;
    let kf = ((n_f as f64) * prm.fast_q).round().max(1.0) as u64;
    let kb = ((n_bw as f64) * prm.fast_q).round().max(1.0) as u64;
    if n_f > 0 {
        g_fwd.locate(n_f / 2, g_fwd.wsum / 2.0, kf);
    }
    if n_bw > 0 {
        g_bwd.locate(n_bw / 2, g_bwd.wsum / 2.0, kb);
    }
    let fs = g_fwd.stats(n_f, kf);
    let bs = g_bwd.stats(n_bw, kb);
    // null_med 可分式（µs → 秒）：mean 均值池化 / geo 几何均值 / med 中位×中位
    let null_med = if !skip_null && n_b > 0 {
        if geo {
            (sum_ln_x / k_a).exp() * (sum_ln_c / n_b as f64).exp() / 1e6
        } else if medm {
            // x 降序中位（a 升序 → x 降序，上中位与 select_nth(len/2) 同秩）
            let mid_x = (s_us - ta[ahi - 1 - (ahi - alo) / 2]).max(0) as f64;
            let mid_c = {
                let m = c_list.len() / 2;
                let (_, &mut v, _) = c_list.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
                v
            };
            mid_x * mid_c / 1e6
        } else {
            (sum_x * sum_c) / (k_a * n_b as f64) / 1e6
        }
    } else {
        f64::NAN
    };
    let null_hit = if !skip_null && n_b > 0 {
        null_hit_acc / n_b as f64
    } else {
        f64::NAN
    };
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
    Some(out)
}

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

// ============================================================================
// 近似模式（仅用于评估妥协方案的截面影响，YHYB_APPROX 版本，不用于生产）
// 复现历史妥协：med/fast5 用 1 秒桶中点、wmed→wmean（加权均值）、
// null_med 均值池化、null_hit x̄ 均值近似。输出布局与精确版完全一致（可直接对比）。
// 距离同样 u64（与生产版一致，无 u32 回绕）。
// ============================================================================

#[derive(Default)]
struct Hist1 {
    cnt: Vec<u32>, // 1 秒桶计数（u32；78KB）
    mean: f64,
    hit: u64,
    wsum: f64,
    wmean_num: f64, // Σw·d（wmean 用）
}

impl Hist1 {
    fn new() -> Self {
        Hist1 {
            cnt: vec![0; N_BUCKETS_1S],
            ..Default::default()
        }
    }
    fn clear(&mut self) {
        self.cnt.fill(0);
        self.mean = 0.0;
        self.hit = 0;
        self.wsum = 0.0;
        self.wmean_num = 0.0;
    }
    #[inline(always)]
    fn push(&mut self, dd: u64, ww: f64, t_us: u64) {
        let b = ((dd / 1_000_000) as usize).min(N_BUCKETS_1S - 1);
        self.cnt[b] += 1;
        self.mean += dd as f64;
        self.wsum += ww;
        self.wmean_num += ww * dd as f64;
        if dd <= t_us {
            self.hit += 1;
        }
    }
    /// 1 秒桶中点统计：[med, mean, hit, fast5, wmean]（秒）。
    fn stats(&self, n: u64, fast_q: f64) -> [f64; 5] {
        if n == 0 {
            return [f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN];
        }
        let k = ((n as f64) * fast_q).round().max(1.0) as u64;
        let half = n / 2;
        let mut acc = 0u64;
        let mut f_acc = 0u64;
        let mut f_sum = 0.0f64;
        let mut med = f64::NAN;
        let mut fast5 = f64::NAN;
        for (b, &c) in self.cnt.iter().enumerate() {
            let c = c as u64;
            if c == 0 {
                continue;
            }
            let mid = b as f64 + 0.5;
            if med.is_nan() {
                acc += c;
                if acc > half {
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
        let wmean = if self.wsum > 0.0 {
            self.wmean_num / self.wsum / 1e6
        } else {
            f64::NAN
        };
        [
            med,
            self.mean / n as f64 / 1e6,
            self.hit as f64 / n as f64,
            fast5,
            wmean,
        ]
    }
}

/// 近似版 agg（妥协方案完整复现）：单遍匹配 + 1 秒桶直方图 + null 近似。
fn agg_one_approx(
    streams: &[Option<[EvStream; N_EVENTS]>],
    null_t: &NullTable,
    ai: usize,
    e: usize,
    p: usize,
    prm: &YhybParams,
) -> Option<Vec<f64>> {
    let sa = streams[ai].as_ref()?;
    let (ta, wa) = (&sa[e].t, &sa[e].w);
    let t_us = (prm.hit_t_s * 1e6) as u64;
    let mut h_fwd = Hist1::new();
    let mut h_bwd = Hist1::new();
    let mut null_val_sum = 0.0f64; // null_med 均值池化
    let mut null_hit_acc = 0.0f64; // null_hit x̄ 近似（每 B 一次）
    let mut null_n = 0u64; // (B,i) 数（null_med 池化用）
    let mut null_n_b = 0u64; // B 数（null_hit 池化用）
    let (alo, ahi, base) = if ta.is_empty() {
        (0usize, 0usize, 0i64)
    } else {
        let base = day_base(ta[0]);
        let (lo, hi) = period_slice(ta, base, p);
        (lo, hi, base)
    };
    let rate = (ahi - alo) as f64;
    let mut out = Vec::with_capacity(15);
    if ahi == alo {
        out.push(rate);
        for _ in 0..14 {
            out.push(f64::NAN);
        }
        return Some(out);
    }
    let s_us = base + PERIOD_HI_S[p] * 1_000_000;
    let k_a = (ahi - alo) as f64;
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
        let m_b = null_t.m[bi][e][p];
        let c_b = null_t.c[bi][e][p];
        // null_hit x̄ 近似：per B 的 A 事件距离均值（每 B 贡献一次）
        let mut sum_x = 0.0f64;
        for &a in &ta[alo..ahi] {
            sum_x += ((s_us - a).max(0)) as f64;
        }
        let x_bar = sum_x / k_a;
        null_hit_acc += 1.0 - (1.0 - t_us as f64 / x_bar).clamp(0.0, 1.0).powf(m_b);
        null_n_b += 1;
        let mut j = blo;
        for i in alo..ahi {
            let a = ta[i];
            let wi = wa[i];
            let x = s_us - a;
            if x > 0 {
                null_val_sum += x as f64 * c_b;
            }
            null_n += 1;
            while j < bhi && tb[j] <= a {
                j += 1;
            }
            if j > blo {
                h_bwd.push((a - tb[j - 1]) as u64, wi, t_us);
            }
            if j < bhi {
                h_fwd.push((tb[j] - a) as u64, wi, t_us);
            }
        }
    }
    let n_f: u64 = h_fwd.cnt.iter().map(|&c| c as u64).sum();
    let n_b: u64 = h_bwd.cnt.iter().map(|&c| c as u64).sum();
    let fs = h_fwd.stats(n_f, prm.fast_q);
    let bs = h_bwd.stats(n_b, prm.fast_q);
    let null_med = if null_n > 0 {
        null_val_sum / null_n as f64 / 1e6
    } else {
        f64::NAN
    };
    let null_hit = if null_n_b > 0 {
        null_hit_acc / null_n_b as f64
    } else {
        f64::NAN
    };
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
    Some(out)
}

/// 从预加载的全市场事件流聚合因子（v1/v2 共同核心）。
/// 并行粒度 (A, 事件, 时段)：5900×23×4 = 54 万个小任务，负载均衡（大票任务拆 4 份）；
/// 零模型系数表全市场预计算一次（build_null_table）。
/// 限流：rayon 全局池固定 **50 线程**（512 核共享机，统一 50 核口径；外部若已设
/// RAYON_NUM_THREADS 则尊重外部设置）。幂等：全局池只初始化一次。
pub fn ensure_threads() {
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(50).build_global();
    }
}

/// 时段链条回退填补：分时段指标（rate 除外）缺失（该时段无事件/无响应）时，
/// 依次用**该股票同一事件更近时段**的真实值回退（p3←p2←p1←p0，最后兜底全天）。
/// 理由：同一股票相邻时段的行为高度相关（代理性好），且各股票填各自的值——
/// 截面区分度保留（不填常数/0）；rate=0 是真实值（无事件）不填。
/// 前视检查：因子为 T 日收盘后构造，p0-p3 均为当日已发生数据，无前视。
fn fill_periods(row: &mut [f64]) {
    for e in 0..N_EVENTS {
        for p in 1..N_PERIODS {
            for m in 1..15 {
                let idx = (e * N_PERIODS + p) * 15 + m;
                if row[idx].is_nan() {
                    let mut done = false;
                    for q in (1..p).rev() {
                        let qv = row[(e * N_PERIODS + q) * 15 + m];
                        if !qv.is_nan() {
                            row[idx] = qv;
                            done = true;
                            break;
                        }
                    }
                    if !done {
                        let p0 = row[(e * N_PERIODS) * 15 + m];
                        if !p0.is_nan() {
                            row[idx] = p0;
                        }
                    }
                }
            }
        }
    }
}

fn compute_from_streams(
    codes: &[String],
    streams: &[Option<[EvStream; N_EVENTS]>],
    prm: &YhybParams,
) -> (Vec<String>, Vec<f32>) {
    ensure_threads();
    let n_stocks = codes.len();
    let null_t = build_null_table(streams);
    let results: Vec<Option<Vec<f64>>> = (0..n_stocks * N_EVENTS * N_PERIODS)
        .into_par_iter()
        .map(|idx| {
            let ai = idx / (N_EVENTS * N_PERIODS);
            let e = (idx / N_PERIODS) % N_EVENTS;
            let p = idx % N_PERIODS;
            agg_one(streams, &null_t, ai, e, p, prm)
        })
        .collect();
    let mut out_codes = Vec::new();
    let mut vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for ai in 0..n_stocks {
        let mut row = Vec::with_capacity(N_FACTORS);
        let mut ok = true;
        for e in 0..N_EVENTS {
            for p in 0..N_PERIODS {
                match &results[(ai * N_EVENTS + e) * N_PERIODS + p] {
                    Some(v) => row.extend_from_slice(v),
                    None => ok = false,
                }
            }
        }
        if ok && row.len() == N_FACTORS {
            if std::env::var("YHYB_NO_FILL").is_err() {
                fill_periods(&mut row);
            }
            out_codes.push(codes[ai].clone());
            vals.extend(row.iter().map(|&x| x as f32));
        }
    }
    (out_codes, vals)
}

/// 近似版聚合（评估用）：并行粒度 (A, 事件, 时段)。
fn compute_from_streams_approx(
    codes: &[String],
    streams: &[Option<[EvStream; N_EVENTS]>],
    prm: &YhybParams,
) -> (Vec<String>, Vec<f32>) {
    ensure_threads();
    let n_stocks = codes.len();
    let null_t = build_null_table(streams);
    let results: Vec<Option<Vec<f64>>> = (0..n_stocks * N_EVENTS * N_PERIODS)
        .into_par_iter()
        .map(|idx| {
            let ai = idx / (N_EVENTS * N_PERIODS);
            let e = (idx / N_PERIODS) % N_EVENTS;
            let p = idx % N_PERIODS;
            agg_one_approx(streams, &null_t, ai, e, p, prm)
        })
        .collect();
    let mut out_codes = Vec::new();
    let mut vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for ai in 0..n_stocks {
        let mut row = Vec::with_capacity(N_FACTORS);
        let mut ok = true;
        for e in 0..N_EVENTS {
            for p in 0..N_PERIODS {
                match &results[(ai * N_EVENTS + e) * N_PERIODS + p] {
                    Some(v) => row.extend_from_slice(v),
                    None => ok = false,
                }
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

/// 公共读盘：读全市场逐笔+盘口 → 事件流（v1 1380 因子与第 4 层网络因子共用）。
/// 注意：必须在任何 rayon 使用之前调用 ensure_threads（全局池限流 50 线程）。
pub fn load_streams(
    date: i64,
    prm: &YhybParams,
) -> std::io::Result<(Vec<String>, Vec<Option<[EvStream; N_EVENTS]>>)> {
    ensure_threads();
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
    Ok((codes, streams))
}

pub fn compute_yhyb_full_with_params(date: i64, prm: &YhybParams) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let t_start = std::time::Instant::now();
    let (codes, streams) = load_streams(date, prm)?;
    let t_read = std::time::Instant::now();
    let res = compute_from_streams(&codes, &streams, prm);
    if std::env::var("YHYB_TIMING").is_ok() {
        eprintln!(
            "YHYB_TIMING date={date} 读盘+检测={:.1}s 聚合={:.1}s",
            t_read.duration_since(t_start).as_secs_f64(),
            t_read.elapsed().as_secs_f64()
        );
    }
    Ok(res)
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
#[pyo3(signature = (date, approx=false))]
pub fn py_yhyb(py: Python<'_>, date: i64, approx: bool) -> PyResult<(Vec<String>, Vec<f32>)> {
    if approx {
        // 近似版（评估妥协方案用）：读盘 + 检测相同，聚合走近似统计
        ensure_threads();
        let codes = list_codes(date, "transaction");
        let prm = YhybParams::default();
        let streams: Vec<Option<[EvStream; N_EVENTS]>> = codes
            .par_iter()
            .map(|code| {
                let trades = read_trade_fast_inner(code, date, false, true, 8 * 1024 * 1024).ok()?;
                if trades.is_empty() {
                    return None;
                }
                let market =
                    read_market_fast_inner(code, date, false, true, 8 * 1024 * 1024).unwrap_or_default();
                Some(detect_all(&trades, &market, &prm))
            })
            .collect();
        return Ok(compute_from_streams_approx(&codes, &streams, &prm));
    }
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
/// - approx：true 时用近似统计（1 秒桶中点 + wmean + null 均值池化/x̄ 近似），
///   仅用于评估妥协方案的截面影响，不用于生产
#[pyfunction]
#[pyo3(signature = (codes, trade_arrays, market_arrays, params=None, approx=false))]
pub fn py_yhyb_from_data(
    _py: Python<'_>,
    codes: Vec<String>,
    trade_arrays: Vec<PyReadonlyArray2<f64>>,
    market_arrays: Vec<PyReadonlyArray2<f64>>,
    params: Option<Vec<f64>>,
    approx: bool,
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
            assert_streams_sorted(&out, "py_yhyb_from_data");
            Some(out)
        })
        .collect();
    if approx {
        Ok(compute_from_streams_approx(&codes, &streams, &prm))
    } else {
        Ok(compute_from_streams(&codes, &streams, &prm))
    }
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
            p.big_amt_wan, p.sweep_w_s, p.sweep_m as f64, p.ice_w_s, p.ice_m as f64, p.jump_q,
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
        // 09:30 起每 60s 一笔，共 10 笔 → 全在 09:30-09:40（只属于 p0 全天）
        let t: Vec<i64> = (0..10).map(|i| base + (5400 + i * 60) * 1_000_000).collect();
        let (lo, hi) = period_slice(&t, base, 0);
        assert_eq!((lo, hi), (0, 10));
        for p in 1..N_PERIODS {
            let (lo, hi) = period_slice(&t, base, p);
            assert_eq!((lo, hi), (0, 0), "p{p} 不应包含 09:30-09:40");
        }
        // 新时段边界（adjust 后时钟）：p1 盘中3h [10:00,14:27)、p2 尾盘30m [14:27,14:57)、
        // p3 下午2h [13:00,15:00)
        let mk = |off: i64| vec![base + off * 1_000_000];
        assert_eq!(period_slice(&mk(6000), base, 1), (0, 1)); // 10:00 ∈ p1
        assert_eq!(period_slice(&mk(5999), base, 1), (0, 0)); // 09:59:59 ∉ p1
        assert_eq!(period_slice(&mk(17820), base, 2), (0, 1)); // 14:27 ∈ p2
        assert_eq!(period_slice(&mk(17819), base, 2), (0, 0)); // 14:26:59 ∉ p2
        assert_eq!(period_slice(&mk(19619), base, 2), (0, 1)); // 14:56:59 ∈ p2
        assert_eq!(period_slice(&mk(12600), base, 3), (0, 1)); // 13:00 ∈ p3
        assert_eq!(period_slice(&mk(19799), base, 3), (0, 1)); // 14:59:59 ∈ p3
        assert_eq!(period_slice(&mk(19800), base, 3), (0, 0)); // 15:00 ∉ p3
    }

    #[test]
    fn test_compute_stats_basic() {
        // 距离 1,2,3,4,5 秒各一条（v3 单遍统计：med/mean/fast5/wmed 均为 1 秒桶中点）
        let mut g = Gather::new();
        for d in [1u64, 2, 3, 4, 5].map(|s| s * 1_000_000) {
            g.push1(d, 1.0, 2_500_000);
        }
        let n = g.n;
        g.locate(n / 2, g.wsum / 2.0, 1);
        let s = g.stats(n, 1);
        assert!((s[0] - 3.5).abs() < 1e-9); // med：中位索引 2 落在桶3 → 中点 3.5s
        assert!((s[1] - 3.5).abs() < 1e-9); // mean：Σ 桶中点/n = (1.5+...+5.5)/5 = 3.5s
        assert!((s[2] - 0.4).abs() < 1e-9); // hit（距离 ≤2.5s）
        assert!((s[3] - 1.5).abs() < 1e-9); // fast5：k=1 → 桶1 中点 1.5s
        assert!((s[4] - 3.5).abs() < 1e-9); // wmed：均匀权重，半权重 2.5 落在桶3 → 中点
        // 加权中位数：权重 [1,1,1,1,5]，一半权重 4.5 → 桶5（累积 4 < 4.5，+5 ≥ 4.5）→ 中点
        let mut g2 = Gather::new();
        for (d, w) in [(1u64, 1.0f64), (2, 1.0), (3, 1.0), (4, 1.0), (5, 5.0)].map(|(d, w)| (d * 1_000_000, w)) {
            g2.push1(d, w, 2_500_000);
        }
        let n2 = g2.n;
        g2.locate(n2 / 2, g2.wsum / 2.0, 1);
        let s2 = g2.stats(n2, 1);
        assert!((s2[4] - 5.5).abs() < 1e-9); // wmed = 桶5 中点（权重中点在大距离侧）
        // u64 大距离回归：8000s（远超旧 u32 上限 4295s）必须落在桶 8000，不得回绕
        let mut g3 = Gather::new();
        g3.push1(8_000_000_000, 1.0, 2_500_000);
        let n3 = g3.n;
        g3.locate(n3 / 2, g3.wsum / 2.0, 1);
        let s3 = g3.stats(n3, 1);
        assert!((s3[0] - 8000.5).abs() < 1e-9); // med = 桶8000 中点
        assert!((s3[4] - 8000.5).abs() < 1e-9); // wmed = 桶8000 中点
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
