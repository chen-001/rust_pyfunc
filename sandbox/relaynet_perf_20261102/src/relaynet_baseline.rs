//! 接力网络（relaynet）横截面因子：同一天全市场股票当节点，
//! 用逐笔对齐到 30 秒格子的价格变动构造多种"共动相似度"，
//! 稀疏化成多张图后在图上跑 Dijkstra/中心性/社区，逐股票输出因子。
//!
//! 数据流（单日）：
//!   1. `list_stock_codes` 枚举 symbol_map ∩ 当日 Level2 文件
//!   2. rayon 读逐笔 → 每股 (last_price[474], ofi[474], active_cnt)
//!   3. 六种相似度矩阵：rd 收益相关 / ra 绝对收益相关 / rs 符号一致 /
//!      rt 市场上行段相关 / ro 主动买压相关 / rl 领先-滞后(有向)
//!   4. ~50 张图（阈值 / kNN 并集与互近邻 / 最大生成树 / 行业内 / 有向）
//!   5. 每张图 22 个节点统计量 + MST 附加 5 个 + 行业 4 个 + 有向 12 个
//!   6. 上午/下午分图 10 统计 + delta、矩阵直出、跨图汇总
//!
//! 因子名规则: rlg_{图标签}_{统计量}，全部由 relaynet_names() 统一生成。
//! 图标签 = g{格子}_{相似度}_{稀疏化}，例如 g30_rd_thr20、g30_ro_knnU30。
//!
//! 与 dijkstra_metrics（单股内部图）的关系：那个把一只股票的逐笔分桶当节点；
//! 本模块把"一只股票"当节点，研究跨股票的信息传导结构，是不同粒度。

use crate::fast_csv_reader::read_trade_fast_inner;
use crate::yupei_dist::compute::day_start_us_of;
use ndarray::Array2;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::rc::Rc;

// ============================================================================
// 常量
// ============================================================================

/// 30 秒格子: 9:30→11:30(上午240格) 接着下午平移后 11:30→13:27(237格), 共 474
const CELL_US: i64 = 30_000_000;
const N_CELL: usize = 474;
/// 下午与上午的分界格（>= 此格算下午）
const PM_SPLIT: usize = 240;
/// 尾盘半小时起始格（14:30-15:00 对应最后 60 个 30s 格）
const TAIL_START: usize = 414;
/// 60 秒格子
const N_CELL60: usize = 237;
const CELL60_US: i64 = 60_000_000;
/// 参与建图的最小活跃格子数（约 1/10 交易日）
const MIN_ACTIVE: usize = 40;
/// 相似度进图的最低"同动样本"（rs/rt 用）
const MIN_PAIR: usize = 10;

/// 因子总数（= relaynet_names().len()，由注册表唯一决定）
pub fn n_factors() -> usize {
    RELAY_NAMES.len()
}

// ============================================================================
// 输入数据
// ============================================================================

/// 每股网格化后的特征
struct StockPrep {
    /// 每格最后一笔成交价（前向填充后取对数收益前的原始价）
    last_px: [f64; N_CELL],
    /// 每格主动买额-主动卖额
    ofi: [f64; N_CELL],
    /// 30s 有成交的格子数
    n_active: usize,
    /// 60s 有成交的格子数
    n_active60: usize,
    /// 当日成交额
    amount: f64,
    /// 当日收益（收盘/第一笔 - 1）
    day_ret: f64,
    /// 30s 收益率序列（含 0 填充）；上午/下午切片共用
    ret30: Vec<f64>,
    /// 60s 收益率序列
    ret60: Vec<f64>,
    /// 60s OFI
    ofi60: Vec<f64>,
    /// 每格成交股数
    vol30: Vec<f64>,
    /// 每格成交金额
    tv30: Vec<f64>,
    /// 每格成交笔数
    cnt30: Vec<f64>,
    /// 每格大单成交额占比（单笔 > 全天平均每笔 5 倍）
    big30: Vec<f64>,
    /// 每格是否有成交 0/1
    act30: Vec<f64>,
    /// 每格不同 bid_order 数量
    buyord30: Vec<f64>,
    /// 每格不同 ask_order 数量
    sellord30: Vec<f64>,
    /// 每格最大单一 bid_order 成交额占比
    dom30: Vec<f64>,
    /// 盘口失衡 (bid_vol-ask_vol)/(bid_vol+ask_vol) 格内均值（来自快照）
    dimb30: Vec<f64>,
    /// 相对价差 (ask1-bid1)/mid 格内均值
    spr30: Vec<f64>,
    /// 快照 mid 的格内对数收益
    midr30: Vec<f64>,
    /// 盘口失衡的格间变动（dimb 差分）
    rba30: Vec<f64>,
    /// 买方多样度-卖方多样度 归一化 (B-S)/(B+S)
    ordbal30: Vec<f64>,
    /// act30 自滞后 2 格（活跃格惯性）
    actlag30: Vec<f64>,
    /// |ret| 差分 (|r_t|-|r_{t-1}|)
    rtn30: Vec<f64>,
    /// 最大单一 ask_order 成交额占比
    domsell30: Vec<f64>,
    /// (主动买额-主动卖额)/(总额)
    imb30: Vec<f64>,
    /// 格内单笔成交量的 std
    vstd30: Vec<f64>,
    /// 格内单笔成交量偏度
    vskew30: Vec<f64>,
    /// 格内最大单笔金额占比
    vmaxshare30: Vec<f64>,
    /// 格内单笔金额的基尼系数
    vgini30: Vec<f64>,
    /// 格内单笔金额的变异系数
    amtcv30: Vec<f64>,
    /// 相邻成交间隔变异系数
    gapcv30: Vec<f64>,
    /// 相邻成交间隔 std（秒）
    gapstd30: Vec<f64>,
    /// 前半格成交额占比
    fhalf30: Vec<f64>,
    /// 成交落在整分/30 秒的比例
    rndsec30: Vec<f64>,
    /// 格内首尾成交间隔（秒）
    tspan30: Vec<f64>,
    /// 格内成交时间均匀度（归一化熵）
    tunif30: Vec<f64>,
    /// 卖档 10 档累积深度斜率
    aslope30: Vec<f64>,
    /// 买档 10 档累积深度斜率
    bslope30: Vec<f64>,
    /// 卖档深度偏度
    dskewa30: Vec<f64>,
    /// 买档深度偏度
    dskewb30: Vec<f64>,
    /// 一档深度占比
    l1conc30: Vec<f64>,
    /// 买档一档占比 - 卖档一档占比
    l1gap30: Vec<f64>,
    /// 卖档总深度格内 std
    dsdask30: Vec<f64>,
    /// 买档总深度格内 std
    dsdbid30: Vec<f64>,
    /// (买-卖) 总深度格内 std
    dsdnet30: Vec<f64>,
}

fn prep_stock(recs: &[crate::fast_csv_reader::TradeRecord], day_start_us: i64) -> StockPrep {
    let mut last_px = [f64::NAN; N_CELL];
    let mut last_px60 = [f64::NAN; N_CELL60];
    let mut ofi = [0.0f64; N_CELL];
    let mut ofi60 = vec![0.0f64; N_CELL60];
    let mut amount = 0.0f64;
    let mut first_px = f64::NAN;
    let mut last_px_any = f64::NAN;
    // 格子聚合（订单类用 (cell, order, amount) 事后排序去重，比每格挂 HashSet 省内存）
    let mut volacc = [0.0f64; N_CELL];
    let mut tvacc = [0.0f64; N_CELL];
    let mut cntacc = [0.0f64; N_CELL];
    let mut ntrades = 0usize;
    let mut bo: Vec<(u32, i64, f64)> = Vec::with_capacity(recs.len());
    let mut so: Vec<(u32, i64, f64)> = Vec::with_capacity(recs.len());
    let mut buysell: Vec<(f64, f64)> = vec![(0.0, 0.0); N_CELL];
    let mut ctrades: Vec<Vec<(i64, f64, f64)>> =
        (0..N_CELL).map(|_| Vec::new()).collect();
    for r in recs {
        let off = r.time_us - day_start_us;
        if off < 0 {
            continue;
        }
        let p = r.price as f64;
        let sgn = match r.flag {
            66 => 1.0,
            83 => -1.0,
            _ => 0.0,
        };
        let t = r.turnover as f64;
        amount += t;
        ntrades += 1;
        if !first_px.is_finite() {
            first_px = p;
        }
        last_px_any = p;
        let c30 = (off / CELL_US) as usize;
        if c30 < N_CELL {
            last_px[c30] = p;
            ofi[c30] += sgn * t;
            volacc[c30] += r.volume as f64;
            tvacc[c30] += t;
            cntacc[c30] += 1.0;
            bo.push((c30 as u32, r.bid_order, t));
            so.push((c30 as u32, r.ask_order, t));
            if sgn > 0.0 { buysell[c30].0 += t; } else if sgn < 0.0 { buysell[c30].1 += t; }
            ctrades[c30].push((off, r.volume as f64, t));
        }
        let c60 = (off / CELL60_US) as usize;
        if c60 < N_CELL60 {
            last_px60[c60] = p;
            ofi60[c60] += sgn * t;
        }
    }
    // 前向填充价格
    let mut prev = f64::NAN;
    let mut n_active = 0usize;
    for i in 0..N_CELL {
        if last_px[i].is_finite() {
            prev = last_px[i];
            n_active += 1;
        } else {
            last_px[i] = prev;
        }
    }
    let mut prev60 = f64::NAN;
    let mut n_active60 = 0usize;
    for i in 0..N_CELL60 {
        if last_px60[i].is_finite() {
            prev60 = last_px60[i];
            n_active60 += 1;
        } else {
            last_px60[i] = prev60;
        }
    }
    // 收益率（首格沿用前收不可得 → NaN→0）
    let mut ret30 = vec![0.0f64; N_CELL];
    for i in 1..N_CELL {
        if last_px[i].is_finite() && last_px[i - 1].is_finite() && last_px[i - 1] > 0.0 {
            ret30[i] = last_px[i] / last_px[i - 1] - 1.0;
        }
    }
    let mut ret60 = vec![0.0f64; N_CELL60];
    for i in 1..N_CELL60 {
        if last_px60[i].is_finite() && last_px60[i - 1].is_finite() && last_px60[i - 1] > 0.0 {
            ret60[i] = last_px60[i] / last_px60[i - 1] - 1.0;
        }
    }
    let day_ret = if first_px.is_finite() && first_px > 0.0 && last_px_any.is_finite() {
        last_px_any / first_px - 1.0
    } else {
        f64::NAN
    };
    // 量/笔数/大单/活跃格（大单阈值：单笔 > 全天平均每笔 5 倍）
    let big_thr = if ntrades > 0 { 5.0 * amount / ntrades as f64 } else { f64::INFINITY };
    let mut big30 = vec![0.0f64; N_CELL];
    let mut act30 = vec![0.0f64; N_CELL];
    for c in 0..N_CELL {
        if tvacc[c] > 0.0 {
            let bn: f64 = ctrades[c].iter().filter(|&&(_, _, t)| t > big_thr).map(|&(_, _, t)| t).sum();
            big30[c] = bn / tvacc[c];
            act30[c] = 1.0;
        }
    }
    // 每格不同订单数 / 主导买卖单占比：按 (cell, order) 排序后扫一遍
    let mut buyord30 = vec![0.0f64; N_CELL];
    let mut sellord30 = vec![0.0f64; N_CELL];
    let mut dom30 = vec![0.0f64; N_CELL];
    let mut domsell30 = vec![0.0f64; N_CELL];
    bo.sort_unstable_by_key(|&(c, o, _)| (c, o));
    let mut i = 0;
    while i < bo.len() {
        let (c, o) = (bo[i].0, bo[i].1);
        let mut s = 0.0f64;
        while i < bo.len() && bo[i].0 == c && bo[i].1 == o {
            s += bo[i].2;
            i += 1;
        }
        buyord30[c as usize] += 1.0;
        if s > dom30[c as usize] {
            dom30[c as usize] = s;
        }
    }
    so.sort_unstable_by_key(|&(c, o, _)| (c, o));
    let mut j = 0;
    while j < so.len() {
        let (c, o) = (so[j].0, so[j].1);
        let mut s = 0.0f64;
        while j < so.len() && so[j].0 == c && so[j].1 == o {
            s += so[j].2;
            j += 1;
        }
        sellord30[c as usize] += 1.0;
        if s > domsell30[c as usize] {
            domsell30[c as usize] = s;
        }
    }
    for c in 0..N_CELL {
        if tvacc[c] > 0.0 {
            dom30[c] /= tvacc[c];
            domsell30[c] /= tvacc[c];
        }
    }
    // 买卖结构
    let mut ordbal30 = vec![0.0f64; N_CELL];
    let mut imb30 = vec![0.0f64; N_CELL];
    for c in 0..N_CELL {
        let (b, s) = (buyord30[c], sellord30[c]);
        if b + s > 0.0 {
            ordbal30[c] = (b - s) / (b + s);
        }
        let (ba, sa) = buysell[c];
        if ba + sa > 0.0 {
            imb30[c] = (ba - sa) / (ba + sa);
        }
    }
    // act30 滞后 2 格 / |ret| 差分
    let mut actlag30 = vec![0.0f64; N_CELL];
    for c in 2..N_CELL {
        actlag30[c] = act30[c - 2];
    }
    let mut rtn30 = vec![0.0f64; N_CELL];
    for c in 1..N_CELL {
        rtn30[c] = ret30[c].abs() - ret30[c - 1].abs();
    }
    // 成交形状/节奏/时间均匀度（对每格逐笔明细）
    let (mut vstd30, mut vskew30, mut vmaxshare30, mut vgini30, mut amtcv30,
         mut gapcv30, mut gapstd30, mut fhalf30, mut rndsec30, mut tspan30, mut tunif30) =
        (vec![0.0f64; N_CELL], vec![0.0f64; N_CELL], vec![0.0f64; N_CELL],
         vec![0.0f64; N_CELL], vec![0.0f64; N_CELL], vec![0.0f64; N_CELL],
         vec![0.0f64; N_CELL], vec![0.0f64; N_CELL], vec![0.0f64; N_CELL],
         vec![0.0f64; N_CELL], vec![0.0f64; N_CELL]);
    for c in 0..N_CELL {
        let tr = &ctrades[c];
        let m = tr.len();
        if m == 0 {
            continue;
        }
        let vols: Vec<f64> = tr.iter().map(|&(_, v, _)| v).collect();
        let amts: Vec<f64> = tr.iter().map(|&(_, _, t)| t).collect();
        let (mv, ma) = (vols.iter().sum::<f64>() / m as f64, amts.iter().sum::<f64>() / m as f64);
        vstd30[c] = (vols.iter().map(|&x| (x - mv) * (x - mv)).sum::<f64>() / m as f64).sqrt();
        let sdv = vstd30[c];
        if sdv > 0.0 {
            vskew30[c] = vols.iter().map(|&x| ((x - mv) / sdv).powi(3)).sum::<f64>() / m as f64;
        }
        if tvacc[c] > 0.0 {
            vmaxshare30[c] = amts.iter().cloned().fold(0.0f64, f64::max) / tvacc[c];
            let mut sorted = amts.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let num: f64 = sorted.iter().enumerate()
                .map(|(k, &x)| (2 * (k + 1)) as f64 * x).sum();
            vgini30[c] = num / (m as f64 * sorted.iter().sum::<f64>()) - (m as f64 + 1.0) / m as f64;
        }
        if ma > 0.0 {
            let sda = (amts.iter().map(|&x| (x - ma) * (x - ma)).sum::<f64>() / m as f64).sqrt();
            amtcv30[c] = sda / ma;
        }
        if m >= 2 {
            let mut gaps: Vec<f64> = Vec::with_capacity(m - 1);
            for k in 1..m {
                gaps.push((tr[k].0 - tr[k - 1].0) as f64 / 1e6);
            }
            let mg = gaps.iter().sum::<f64>() / gaps.len() as f64;
            let sdg = (gaps.iter().map(|&x| (x - mg) * (x - mg)).sum::<f64>()
                / gaps.len() as f64).sqrt();
            gapstd30[c] = sdg;
            if mg > 0.0 {
                gapcv30[c] = sdg / mg;
            }
            tspan30[c] = (tr[m - 1].0 - tr[0].0) as f64 / 1e6;
        }
        // 前半格成交额占比 / 整点感 / 时间均匀度
        let mut fh = 0.0;
        let mut rnd = 0usize;
        let mut bins = [0usize; 6];
        for &(off, _, t) in tr {
            let within = off % CELL_US;
            if within < CELL_US / 2 {
                fh += t;
            }
            let sec = (off / 1_000_000) % 60;
            if sec == 0 || sec == 30 {
                rnd += 1;
            }
            let b = ((within * 6) / CELL_US).clamp(0, 5) as usize;
            bins[b] += 1;
        }
        if tvacc[c] > 0.0 {
            fhalf30[c] = fh / tvacc[c];
        }
        rndsec30[c] = rnd as f64 / m as f64;
        // 归一化熵: -Σp·ln p / ln6
        let mut h = 0.0;
        for &bc in &bins {
            if bc > 0 {
                let p = bc as f64 / m as f64;
                h -= p * p.ln();
            }
        }
        tunif30[c] = h / 6f64.ln();
    }
    StockPrep {
        last_px, ofi, n_active, n_active60, amount, day_ret, ret30, ret60, ofi60,
        vol30: volacc.to_vec(), tv30: tvacc.to_vec(), cnt30: cntacc.to_vec(),
        big30, act30, buyord30, sellord30, dom30,
        dimb30: vec![0.0; N_CELL], spr30: vec![0.0; N_CELL],
        midr30: vec![0.0; N_CELL], rba30: vec![0.0; N_CELL],
        ordbal30, actlag30, rtn30, domsell30, imb30,
        vstd30, vskew30, vmaxshare30, vgini30, amtcv30,
        gapcv30, gapstd30, fhalf30, rndsec30, tspan30, tunif30,
        aslope30: vec![0.0; N_CELL], bslope30: vec![0.0; N_CELL],
        dskewa30: vec![0.0; N_CELL], dskewb30: vec![0.0; N_CELL],
        l1conc30: vec![0.0; N_CELL], l1gap30: vec![0.0; N_CELL],
        dsdask30: vec![0.0; N_CELL], dsdbid30: vec![0.0; N_CELL],
        dsdnet30: vec![0.0; N_CELL],
    }
}

/// 10 档累积深度斜率（最小二乘，y=cumsum(vols)）
fn depth_slope(vols: &[f32; 10]) -> f64 {
    let mut cum = [0.0f64; 10];
    let mut s = 0.0;
    for k in 0..10 {
        s += vols[k] as f64;
        cum[k] = s;
    }
    let mx = 4.5f64;
    let my = cum.iter().sum::<f64>() / 10.0;
    let mut num = 0.0;
    for k in 0..10 {
        num += (k as f64 - mx) * (cum[k] - my);
    }
    num / 82.5 // Σ(k-4.5)²
}

/// 10 个数的偏度
fn skew10(vols: &[f32; 10]) -> f64 {
    let a: Vec<f64> = vols.iter().map(|&x| x as f64).collect();
    let m = a.iter().sum::<f64>() / 10.0;
    let sd = (a.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / 10.0).sqrt();
    if sd <= 0.0 {
        return 0.0;
    }
    a.iter().map(|&x| ((x - m) / sd).powi(3)).sum::<f64>() / 10.0
}

/// 盘口快照填 dimb30/spr30/midr30/rba30 + 十档形状；mrecs 为空时保持 0。
fn fill_market(p: &mut StockPrep, mrecs: &[crate::fast_csv_reader::MarketRecord], day_start_us: i64) {
    if mrecs.is_empty() {
        return;
    }
    let mut dnum = [0.0f64; N_CELL];
    let mut snum = [0.0f64; N_CELL];
    let mut dcnt = [0usize; N_CELL];
    let mut scnt = [0usize; N_CELL];
    let mut mid = [f64::NAN; N_CELL];
    // 十档形状的格内和/计数/深度平方和
    let mut asl = [0.0f64; N_CELL];
    let mut bsl = [0.0f64; N_CELL];
    let mut dska = [0.0f64; N_CELL];
    let mut dskb = [0.0f64; N_CELL];
    let mut l1b = [0.0f64; N_CELL];
    let mut l1a = [0.0f64; N_CELL];
    let mut shcnt = [0usize; N_CELL];
    let (mut ta1, mut ta2, mut tb1, mut tb2, mut tn1, mut tn2, mut tncnt) =
        ([0.0f64; N_CELL], [0.0f64; N_CELL], [0.0f64; N_CELL],
         [0.0f64; N_CELL], [0.0f64; N_CELL], [0.0f64; N_CELL], [0usize; N_CELL]);
    for r in mrecs {
        let off = r.time_us - day_start_us;
        if off < 0 {
            continue;
        }
        let c = (off / CELL_US) as usize;
        if c >= N_CELL {
            continue;
        }
        let (bv, av) = (r.total_bid_vol as f64, r.total_ask_vol as f64);
        if bv + av > 0.0 {
            dnum[c] += (bv - av) / (bv + av);
            dcnt[c] += 1;
        }
        let (a1, b1) = (r.ask_prcs[0] as f64, r.bid_prcs[0] as f64);
        let m = (a1 + b1) / 2.0;
        if a1 > 0.0 && b1 > 0.0 && m > 0.0 {
            snum[c] += (a1 - b1) / m;
            scnt[c] += 1;
            mid[c] = m;
        }
        // 十档形状
        let sa: f64 = r.ask_vols.iter().map(|&x| x as f64).sum();
        let sb: f64 = r.bid_vols.iter().map(|&x| x as f64).sum();
        if sa > 0.0 && sb > 0.0 {
            asl[c] += depth_slope(&r.ask_vols);
            bsl[c] += depth_slope(&r.bid_vols);
            dska[c] += skew10(&r.ask_vols);
            dskb[c] += skew10(&r.bid_vols);
            l1a[c] += r.ask_vols[0] as f64 / sa;
            l1b[c] += r.bid_vols[0] as f64 / sb;
            shcnt[c] += 1;
        }
        ta1[c] += sa;
        ta2[c] += sa * sa;
        tb1[c] += sb;
        tb2[c] += sb * sb;
        let d = sb - sa;
        tn1[c] += d;
        tn2[c] += d * d;
        tncnt[c] += 1;
    }
    for c in 0..N_CELL {
        if dcnt[c] > 0 {
            p.dimb30[c] = dnum[c] / dcnt[c] as f64;
        }
        if scnt[c] > 0 {
            p.spr30[c] = snum[c] / scnt[c] as f64;
        }
        if shcnt[c] > 0 {
            let k = shcnt[c] as f64;
            p.aslope30[c] = asl[c] / k;
            p.bslope30[c] = bsl[c] / k;
            p.dskewa30[c] = dska[c] / k;
            p.dskewb30[c] = dskb[c] / k;
            p.l1conc30[c] = (l1b[c] + l1a[c]) / (2.0 * k);
            p.l1gap30[c] = (l1b[c] - l1a[c]) / k;
        }
        if tncnt[c] > 1 {
            let k = tncnt[c] as f64;
            p.dsdask30[c] = ((ta2[c] - ta1[c] * ta1[c] / k) / k).max(0.0).sqrt();
            p.dsdbid30[c] = ((tb2[c] - tb1[c] * tb1[c] / k) / k).max(0.0).sqrt();
            p.dsdnet30[c] = ((tn2[c] - tn1[c] * tn1[c] / k) / k).max(0.0).sqrt();
        }
    }
    // mid 前向填充后取对数收益
    let mut prev = f64::NAN;
    for c in 0..N_CELL {
        if mid[c].is_finite() {
            prev = mid[c];
        } else {
            mid[c] = prev;
        }
    }
    for c in 1..N_CELL {
        if mid[c].is_finite() && mid[c - 1].is_finite() && mid[c - 1] > 0.0 {
            p.midr30[c] = (mid[c] / mid[c - 1]).ln();
        }
    }
    for c in 1..N_CELL {
        p.rba30[c] = p.dimb30[c] - p.dimb30[c - 1];
    }
}

// ============================================================================
// 图结构
// ============================================================================

struct UGraph {
    n: usize,
    adj: Vec<Vec<(u32, f32)>>,
}
impl UGraph {
    fn deg(&self, i: usize) -> usize {
        self.adj[i].len()
    }
}

#[derive(PartialEq)]
struct State {
    d: f64,
    u: usize,
}
impl Eq for State {}
impl Ord for State {
    fn cmp(&self, o: &Self) -> Ordering {
        o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

// ---------------------------------------------------------------------------
// 图构建
// ---------------------------------------------------------------------------

/// 密度自适应阈值图：先每行留 top-k 候选（k=max(200,n/25)），
/// 候选集里再过阈值 t。等价"每只股票的最强 k 个候选里再过门槛"，
/// 防止 rtv 这类高相关族变成全连图。
fn build_thresh_limited(s: &Array2<f64>, t: f64, allow: &[bool]) -> UGraph {
    let n = s.nrows();
    let k = (n / 25).max(200).min(n.saturating_sub(1)).max(1);
    let mut inset = vec![vec![false; n]; n];
    for i in 0..n {
        if !allow[i] {
            continue;
        }
        let mut heap: BinaryHeap<(OrderedF, u32)> = BinaryHeap::new();
        for j in 0..n {
            if j == i || !allow[j] {
                continue;
            }
            let v = s[[i, j]];
            if !v.is_finite() {
                continue;
            }
            let item = (OrderedF(v), j as u32);
            if heap.len() < k {
                heap.push(item);
            } else if heap.peek().unwrap().0 .0 < v {
                heap.pop();
                heap.push(item);
            }
        }
        for (_, j) in heap {
            inset[i][j as usize] = true;
        }
    }
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        if !allow[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !allow[j] {
                continue;
            }
            if !(inset[i][j] || inset[j][i]) {
                continue;
            }
            let v = s[[i, j]];
            if v.is_finite() && v > t {
                adj[i].push((j as u32, v as f32));
                adj[j].push((i as u32, v as f32));
            }
        }
    }
    UGraph { n, adj }
}

fn build_knn(s: &Array2<f64>, k: usize, mutual: bool) -> UGraph {
    let n = s.nrows();
    let mut inset = vec![vec![false; n]; n];
    for i in 0..n {
        // 第 i 行的 top-k（堆排前 k）
        let mut heap: BinaryHeap<(OrderedF, u32)> = BinaryHeap::new();
        for j in 0..n {
            if j == i {
                continue;
            }
            let v = s[[i, j]];
            if !v.is_finite() {
                continue;
            }
            let item = (OrderedF(v), j as u32);
            if heap.len() < k {
                heap.push(item);
            } else if heap.peek().unwrap().0 .0 < v {
                heap.pop();
                heap.push(item);
            }
        }
        for (_, j) in heap {
            inset[i][j as usize] = true;
        }
    }
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let keep = if mutual {
                inset[i][j] && inset[j][i]
            } else {
                inset[i][j] || inset[j][i]
            };
            if keep {
                let v = s[[i, j]];
                if v.is_finite() {
                    adj[i].push((j as u32, v as f32));
                    adj[j].push((i as u32, v as f32));
                }
            }
        }
    }
    UGraph { n, adj }
}

fn build_mst(s: &Array2<f64>) -> UGraph {
    let n = s.nrows();
    let mut intree = vec![false; n];
    let mut best = vec![f64::NEG_INFINITY; n];
    let mut par = vec![usize::MAX; n];
    // 找一个有相似度的起点
    let start = (0..n).find(|&i| (0..n).any(|j| j != i && s[[i, j]].is_finite()));
    if let Some(st) = start {
        intree[st] = true;
        for j in 0..n {
            if j != st {
                let v = s[[st, j]];
                if v.is_finite() {
                    best[j] = v;
                    par[j] = st;
                }
            }
        }
        for _ in 1..n {
            let mut u = usize::MAX;
            let mut bv = f64::NEG_INFINITY;
            for j in 0..n {
                if !intree[j] && best[j] > bv {
                    bv = best[j];
                    u = j;
                }
            }
            if u == usize::MAX {
                break;
            }
            intree[u] = true;
            for j in 0..n {
                if !intree[j] {
                    let v = s[[u, j]];
                    if v.is_finite() && v > best[j] {
                        best[j] = v;
                        par[j] = u;
                    }
                }
            }
        }
    }
    let mut adj = vec![Vec::new(); n];
    for j in 0..n {
        if par[j] != usize::MAX {
            let v = s[[j, par[j]]] as f32;
            adj[j].push((par[j] as u32, v));
            adj[par[j]].push((j as u32, v));
        }
    }
    UGraph { n, adj }
}

fn build_ind_thresh(s: &Array2<f64>, inds: &[i64], t: f64) -> UGraph {
    let n = s.nrows();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        if inds[i] <= 0 {
            continue;
        }
        for j in (i + 1)..n {
            if inds[j] == inds[i] {
                let v = s[[i, j]];
                if v.is_finite() && v > t {
                    adj[i].push((j as u32, v as f32));
                    adj[j].push((i as u32, v as f32));
                }
            }
        }
    }
    UGraph { n, adj }
}

/// 有向图: adj_out[i] = (j, sim), sim>0 才连
struct DGraph {
    n: usize,
    out_adj: Vec<Vec<(u32, f32)>>,
    in_adj: Vec<Vec<(u32, f32)>>,
}
fn build_digraph(s: &Array2<f64>, t: f64) -> DGraph {
    let n = s.nrows();
    let mut out_adj = vec![Vec::new(); n];
    let mut in_adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = s[[i, j]];
            if v.is_finite() && v > t {
                out_adj[i].push((j as u32, v as f32));
                in_adj[j].push((i as u32, v as f32));
            }
        }
    }
    DGraph { n, out_adj, in_adj }
}

// ---------------------------------------------------------------------------
// OrderedF 包装
// ---------------------------------------------------------------------------
#[derive(PartialEq)]
struct OrderedF(f64);
impl Eq for OrderedF {}
impl Ord for OrderedF {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for OrderedF {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

// ---------------------------------------------------------------------------
// 单源 Dijkstra / 连通分量
// ---------------------------------------------------------------------------

fn dijkstra(g: &UGraph, src: usize) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; g.n];
    d[src] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State { d: 0.0, u: src });
    while let Some(State { d: du, u }) = heap.pop() {
        if du > d[u] + 1e-12 {
            continue;
        }
        for &(v, sim) in &g.adj[u] {
            let w = (1.0 - sim as f64).max(1e-9);
            if du + w < d[v as usize] - 1e-12 {
                d[v as usize] = du + w;
                heap.push(State { d: du + w, u: v as usize });
            }
        }
    }
    d
}

fn components(g: &UGraph) -> (Vec<u32>, Vec<u32>) {
    let n = g.n;
    let mut cid = vec![u32::MAX; n];
    let mut sizes = Vec::new();
    let mut stack = Vec::new();
    for s in 0..n {
        if cid[s] != u32::MAX {
            continue;
        }
        let id = sizes.len() as u32;
        let mut sz = 0u32;
        cid[s] = id;
        stack.push(s);
        while let Some(u) = stack.pop() {
            sz += 1;
            for &(v, _) in &g.adj[u] {
                if cid[v as usize] == u32::MAX {
                    cid[v as usize] = id;
                    stack.push(v as usize);
                }
            }
        }
        sizes.push(sz);
    }
    (cid, sizes)
}

// ---------------------------------------------------------------------------
// 图统计
// ---------------------------------------------------------------------------

fn dist_stats(g: &UGraph, nsrc: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (h, f, e, _, _, _, _) = dist_stats_q(g, nsrc);
    (h, f, e)
}

/// 采样源的个数：max(10, n/25)，把全源 Dijkstra 压成 ~1/25
fn n_src_of(n: usize) -> usize {
    (n / 25).max(10).min(n)
}

/// 采样最短路统计：只从 nsrc 个等距源跑 Dijkstra，
/// 采样源出真实值，其余节点填采样源的平均（形状不变）
fn dist_stats_q(g: &UGraph, nsrc: usize)
    -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
{
    let n = g.n;
    // 只在有边节点里采样源（全孤立图上均匀抽样可能一个连通源都抽不到）
    let mut cand: Vec<usize> = (0..n).filter(|&i| !g.adj[i].is_empty()).collect();
    if cand.is_empty() {
        cand = (0..n).collect();
    }
    let nsrc = nsrc.min(cand.len()).max(1);
    let step = (cand.len() / nsrc).max(1);
    let sources: Vec<usize> = cand.iter().step_by(step).take(nsrc).copied().collect();
    let res: Vec<(usize, f64, f64, f64, f64, f64, f64, f64)> = sources
        .par_iter()
        .map(|&s| {
            let d = dijkstra(g, s);
            let (mut h, mut f, mut e) = (0.0, 0.0, 0.0f64);
            let mut ds: Vec<f64> = Vec::with_capacity(n);
            for (j, &x) in d.iter().enumerate() {
                if j != s && x.is_finite() && x > 0.0 {
                    h += 1.0 / x;
                    f += x;
                    if x > e {
                        e = x;
                    }
                    ds.push(x);
                }
            }
            if ds.is_empty() {
                (s, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            } else {
                ds.sort_by(|a, b| a.total_cmp(b));
                let c = ds.len() as f64;
                (s, h / c, f / c, e,
                 quantile_sorted(&ds, 0.25), quantile_sorted(&ds, 0.75),
                 quantile_sorted(&ds, 0.50), quantile_sorted(&ds, 0.90))
            }
        })
        .collect();
    // 采样源均值作为兜底
    let (mut sh, mut sf, mut se, mut s25, mut s75, mut s50, mut s90) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut valid = 0usize;
    for t in &res {
        if t.1.is_finite() {
            sh += t.1; sf += t.2; se += t.3; s25 += t.4; s75 += t.5;
            s50 += t.6; s90 += t.7;
            valid += 1;
        }
    }
    let fill = if valid > 0 {
        let k = valid as f64;
        (sh / k, sf / k, se / k, s25 / k, s75 / k, s50 / k, s90 / k)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };
    let (mut h, mut f, mut e, mut q25, mut q75, mut q50, mut q90) =
        (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
         vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
         vec![f64::NAN; n]);
    // 非采样节点：先取它"采样邻居"的均值（保留分辨率），没有再退到全局均值
    let mut src_idx = vec![usize::MAX; n];
    for (k, t) in res.iter().enumerate() {
        src_idx[t.0] = k;
    }
    for i in 0..n {
        if src_idx[i] != usize::MAX || g.adj[i].is_empty() {
            continue;
        }
        let (mut ah, mut af, mut ae, mut a25, mut a75, mut a50, mut a90) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut c = 0usize;
        for &(v, _) in &g.adj[i] {
            let vi = v as usize;
            if src_idx[vi] != usize::MAX {
                let t = &res[src_idx[vi]];
                if t.1.is_finite() {
                    ah += t.1; af += t.2; ae += t.3; a25 += t.4; a75 += t.5;
                    a50 += t.6; a90 += t.7;
                    c += 1;
                }
            }
        }
        if c > 0 {
            let k = c as f64;
            h[i] = ah / k; f[i] = af / k; e[i] = ae / k;
            q25[i] = a25 / k; q75[i] = a75 / k; q50[i] = a50 / k; q90[i] = a90 / k;
        } else {
            h[i] = fill.0; f[i] = fill.1; e[i] = fill.2;
            q25[i] = fill.3; q75[i] = fill.4; q50[i] = fill.5; q90[i] = fill.6;
        }
    }
    for (s_, a, b, c, d, k, m50, m90) in res {
        h[s_] = a; f[s_] = b; e[s_] = c; q25[s_] = d; q75[s_] = k;
        q50[s_] = m50; q90[s_] = m90;
    }
    (h, f, e, q25, q75, q50, q90)
}

/// 已排序数组的分位数（线性插值）
fn quantile_sorted(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

fn betweenness_bin(g: &UGraph, nsrc: usize) -> Vec<f64> {
    let n = g.n;
    let mut cand: Vec<usize> = (0..n).filter(|&i| !g.adj[i].is_empty()).collect();
    if cand.is_empty() {
        cand = (0..n).collect();
    }
    let nsrc = nsrc.min(cand.len()).max(1);
    let step = (cand.len() / nsrc).max(1);
    let sources: Vec<usize> = cand.iter().step_by(step).take(nsrc).copied().collect();
    let k = n as f64 / sources.len() as f64;
    (0..sources.len())
        .into_par_iter()
        .map(|si| {
            let s = sources[si];
            let mut stack: Vec<usize> = Vec::with_capacity(256);
            let mut pred: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n];
            let mut dist = vec![-1i32; n];
            let mut bet = vec![0.0f64; n];
            sigma[s] = 1.0;
            dist[s] = 0;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(s);
            while let Some(u) = queue.pop_front() {
                stack.push(u);
                for &(v, _) in &g.adj[u] {
                    let v = v as usize;
                    if dist[v] < 0 {
                        dist[v] = dist[u] + 1;
                        queue.push_back(v);
                    }
                    if dist[v] == dist[u] + 1 {
                        sigma[v] += sigma[u];
                        pred[v].push(u as u32);
                    }
                }
            }
            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    delta[v as usize] += (sigma[v as usize] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    bet[w] += delta[w];
                }
            }
            bet
        })
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for i in 0..n {
                    a[i] += b[i];
                }
                a
            },
        )
        .into_iter()
        .map(|x| x * k / 2.0)
        .collect()
}

/// 加权中介（Dijkstra 版 Brandes），nsrc 个等距抽样源近似
fn betweenness_weighted(g: &UGraph, nsrc: usize) -> Vec<f64> {
    let n = g.n;
    let mut cand: Vec<usize> = (0..n).filter(|&i| !g.adj[i].is_empty()).collect();
    if cand.is_empty() {
        cand = (0..n).collect();
    }
    let nsrc = nsrc.min(cand.len()).max(1);
    let step = (cand.len() / nsrc).max(1);
    let sources: Vec<usize> = cand.iter().step_by(step).take(nsrc).copied().collect();
    let k = n as f64 / sources.len() as f64;
    (0..sources.len())
        .into_par_iter()
        .map(|si| {
            let s = sources[si];
            let mut stack: Vec<usize> = Vec::with_capacity(256);
            let mut pred: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n];
            let mut dist = vec![f64::INFINITY; n];
            let mut bet = vec![0.0f64; n];
            sigma[s] = 1.0;
            dist[s] = 0.0;
            let mut heap = BinaryHeap::new();
            heap.push(State { d: 0.0, u: s });
            while let Some(State { d: du, u }) = heap.pop() {
                if du > dist[u] + 1e-12 {
                    continue;
                }
                stack.push(u);
                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    let w = (1.0 - sim as f64).max(1e-9);
                    let nd = du + w;
                    if nd < dist[v] - 1e-9 {
                        dist[v] = nd;
                        heap.push(State { d: nd, u: v });
                        sigma[v] = sigma[u];
                        pred[v].clear();
                        pred[v].push(u as u32);
                    } else if (nd - dist[v]).abs() < 1e-9 && dist[v].is_finite() {
                        sigma[v] += sigma[u];
                        pred[v].push(u as u32);
                    }
                }
            }
            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    let vv = v as usize;
                    if sigma[w] > 0.0 {
                        delta[vv] += (sigma[vv] / sigma[w]) * (1.0 + delta[w]);
                    }
                }
                if w != s {
                    bet[w] += delta[w];
                }
            }
            bet
        })
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for i in 0..n {
                    a[i] += b[i];
                }
                a
            },
        )
        .into_iter()
        .map(|x| x * k / 2.0)
        .collect()
}

fn pagerank(g: &UGraph, d: f64, iters: usize) -> Vec<f64> {
    let n = g.n;
    let mut pr = vec![1.0 / n as f64; n];
    let mut next = vec![0.0; n];
    let strength: Vec<f64> = g
        .adj
        .iter()
        .map(|a| a.iter().map(|&(_, s)| s as f64).sum())
        .collect();
    for _ in 0..iters {
        let mut dangling = 0.0;
        for i in 0..n {
            next[i] = (1.0 - d) / n as f64;
            if strength[i] <= 0.0 {
                dangling += pr[i];
                continue;
            }
            for &(v, s) in &g.adj[i] {
                next[v as usize] += d * pr[i] * s as f64 / strength[i];
            }
        }
        for i in 0..n {
            next[i] += d * dangling / n as f64;
        }
        std::mem::swap(&mut pr, &mut next);
        next.fill(0.0);
    }
    pr
}

fn eigenvec(g: &UGraph, iters: usize) -> Vec<f64> {
    let n = g.n;
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut y = vec![0.0; n];
    for _ in 0..iters {
        for i in 0..n {
            y[i] = g.adj[i].iter().map(|&(v, s)| s as f64 * x[v as usize]).sum();
        }
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 0.0 {
            break;
        }
        for i in 0..n {
            x[i] = y[i] / norm;
            y[i] = 0.0;
        }
    }
    x
}

fn kcore(g: &UGraph) -> Vec<u32> {
    let n = g.n;
    let mut deg: Vec<u32> = g.adj.iter().map(|a| a.len() as u32).collect();
    let mut core = vec![0u32; n];
    let mut alive = vec![true; n];
    let mut k = 1u32;
    loop {
        let mut progress = true;
        while progress {
            progress = false;
            for i in 0..n {
                if alive[i] && deg[i] < k {
                    alive[i] = false;
                    core[i] = k - 1;
                    progress = true;
                    for &(v, _) in &g.adj[i] {
                        if alive[v as usize] {
                            deg[v as usize] -= 1;
                        }
                    }
                }
            }
        }
        if !alive.iter().any(|&a| a) {
            break;
        }
        k += 1;
    }
    for i in 0..n {
        if alive[i] {
            core[i] = k - 1;
        }
    }
    core
}

fn clustering_w(g: &UGraph) -> Vec<f64> {
    let n = g.n;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let di = g.deg(i);
            if di < 2 {
                return 0.0f64;
            }
            let mut acc = 0.0;
            for a in 0..g.adj[i].len() {
                let (j, sij) = g.adj[i][a];
                for &(k, sjk) in &g.adj[j as usize] {
                    if k as usize == i {
                        continue;
                    }
                    if let Some(&(_, ski)) = g.adj[i].iter().find(|&&(x, _)| x == k) {
                        let prod = sij as f64 * sjk as f64 * ski as f64;
                        if prod > 0.0 {
                            acc += prod.powf(1.0 / 3.0);
                        }
                    }
                }
            }
            acc / (di as f64 * (di - 1) as f64)
        })
        .collect()
}

fn communities(g: &UGraph) -> Vec<u32> {
    let n = g.n;
    let mut lab: Vec<u32> = (0..n as u32).collect();
    let mut changed = true;
    let mut it = 0;
    while changed && it < 100 {
        changed = false;
        it += 1;
        for i in 0..n {
            if g.adj[i].is_empty() {
                continue;
            }
            let mut w: BTreeMap<u32, f64> = Default::default();
            for &(v, s) in &g.adj[i] {
                *w.entry(lab[v as usize]).or_insert(0.0) += s as f64;
            }
            let best = w
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal).then(b.0.cmp(a.0)))
                .map(|(&l, _)| l);
            if let Some(l) = best {
                if l != lab[i] {
                    lab[i] = l;
                    changed = true;
                }
            }
        }
    }
    lab
}

// ---------------------------------------------------------------------------
// 每图统计打包
// ---------------------------------------------------------------------------

const STATS_UNDIR: [&str; 51] = [
    "deg", "str", "top1", "top3", "meansim", "entsim", "kcore", "reach2", "compsize",
    "harm", "farness", "ecc", "bet", "betw", "pagerank", "eigv", "clusw", "comm_size",
    "comm_inshare", "dist_mkt", "dist_lead", "neighind",
    "ndeg", "hhi", "jacc", "leafnb", "d25", "d75", "assort",
    // 抗极端值扩展层（22 个，全部用分位数/top-k 均值/比值，不用单点极值）
    "str_p25", "str_p50", "str_p90", "str_t10m", "str_cv",
    "ndeg_p50", "reach3", "pdl50", "dist_p90", "dist_iqr",
    "bet_t10", "pr_log", "tri_p75", "corefrac",
    "comm_top_sim", "comm_edges_out", "comm_deg_in",
    "dist_mkt_top10", "lead_bet_mean", "local_density", "deg_rank",
    "mstreach",
];
const REDUCED_STATS: [&str; 9] = [
    "deg", "str", "top1", "top3", "meansim", "entsim", "compsize", "dist_mkt",
    "neighind",
];
const MST_EXTRA: [&str; 5] = ["submass", "is_art", "treedepth", "hops_mkt", "leaf"];
const DIR_STATS: [&str; 12] = [
    "outstr", "instr", "netlead", "outdeg", "indeg", "leadshare", "netpr", "prlead",
    "outharm", "inharm", "outreach", "inreach",
];

fn stat_undir(
    g: &UGraph,
    s: &Array2<f64>,
    inds: &[i64],
    mkt_corr: &[f64],
    lead_set: &[bool],
    full: bool,
) -> Vec<Vec<f64>> {
    let n = g.n;
    let mut cols: Vec<Vec<f64>> = Vec::new();

    let mut deg = vec![0.0f64; n];
    let mut str_ = vec![0.0f64; n];
    let mut top1 = vec![f64::NAN; n];
    let mut top3 = vec![f64::NAN; n];
    let mut meansim = vec![f64::NAN; n];
    let mut entsim = vec![f64::NAN; n];
    for i in 0..n {
        let a = &g.adj[i];
        deg[i] = a.len() as f64;
        str_[i] = a.iter().map(|&(_, x)| x as f64).sum();
        if !a.is_empty() {
            let mut sims: Vec<f64> = a.iter().map(|&(_, x)| x as f64).collect();
            sims.sort_by(|x, y| y.total_cmp(x));
            top1[i] = sims[0];
            top3[i] = sims.iter().take(3).sum::<f64>() / sims.len().min(3) as f64;
            meansim[i] = str_[i] / a.len() as f64;
            let ss: f64 = sims.iter().sum();
            if ss > 0.0 {
                entsim[i] = -sims
                    .iter()
                    .map(|&p| {
                        let q = p / ss;
                        if q > 0.0 { q * q.ln() } else { 0.0 }
                    })
                    .sum::<f64>();
            }
        }
    }
    cols.push(deg);
    cols.push(str_);
    cols.push(top1);
    cols.push(top3);
    cols.push(meansim);
    cols.push(entsim);

    if !full {
        // 缩减包：只出前 6 个 + compsize + dist_mkt + neighind
        let (cid, sizes) = components(g);
        cols.push(
            cid.iter()
                .map(|&c| if c == u32::MAX { f64::NAN } else { sizes[c as usize] as f64 })
                .collect(),
        );
        cols.push(dist_to_mkt(g, mkt_corr));
        let mut neighind = vec![f64::NAN; n];
        for i in 0..n {
            if !g.adj[i].is_empty() && inds[i] > 0 {
                let same = g.adj[i]
                    .iter()
                    .filter(|&&(v, _)| inds[v as usize] == inds[i])
                    .count();
                neighind[i] = same as f64 / g.adj[i].len() as f64;
            }
        }
        cols.push(neighind);
        let _ = s;
        let _ = lead_set;
        return cols;
    }

    cols.push(kcore(g).iter().map(|&x| x as f64).collect());

    let mut reach2 = vec![0.0; n];
    for i in 0..n {
        let mut mark = vec![false; n];
        mark[i] = true;
        for &(v, _) in &g.adj[i] {
            mark[v as usize] = true;
            for &(w, _) in &g.adj[v as usize] {
                mark[w as usize] = true;
            }
        }
        reach2[i] = mark.iter().filter(|&&b| b).count() as f64 - 1.0;
    }
    cols.push(reach2);

    let (cid, sizes) = components(g);
    let lcc_id = sizes
        .iter()
        .enumerate()
        .max_by_key(|&(_, &z)| z)
        .map(|(i, _)| i as u32);
    cols.push(
        cid.iter()
            .map(|&c| if c == u32::MAX { f64::NAN } else { sizes[c as usize] as f64 })
            .collect(),
    );

    let (harm, farness, ecc, d25, d75, pdl50, dist_p90) = dist_stats_q(g, n_src_of(n));
    cols.push(harm);
    cols.push(farness);
    cols.push(ecc);

    cols.push(betweenness_bin(g, n_src_of(n)));
    cols.push(betweenness_weighted(g, n_src_of(n)));
    cols.push(pagerank(g, 0.85, 50));
    cols.push(eigenvec(g, 50));
    cols.push(clustering_w(g));

    let lab = communities(g);
    let mut lsize: std::collections::HashMap<u32, u32> = Default::default();
    for &l in &lab {
        *lsize.entry(l).or_insert(0) += 1;
    }
    cols.push(lab.iter().map(|&l| lsize[&l] as f64).collect());
    let mut inshare = vec![f64::NAN; n];
    for i in 0..n {
        if g.adj[i].is_empty() {
            continue;
        }
        let tot: f64 = g.adj[i].iter().map(|&(_, s)| s as f64).sum();
        let inside: f64 = g.adj[i]
            .iter()
            .filter(|&&(v, _)| lab[v as usize] == lab[i])
            .map(|&(_, s)| s as f64)
            .sum();
        inshare[i] = inside / tot;
    }
    cols.push(inshare);

    cols.push(dist_to_mkt(g, mkt_corr));
    cols.push(dist_to_set(g, lead_set));

    let mut neighind = vec![f64::NAN; n];
    for i in 0..n {
        if !g.adj[i].is_empty() && inds[i] > 0 {
            let same = g.adj[i]
                .iter()
                .filter(|&&(v, _)| inds[v as usize] == inds[i])
                .count();
            neighind[i] = same as f64 / g.adj[i].len() as f64;
        }
    }
    cols.push(neighind);
    let _ = lcc_id;
    let _ = s;

    // ---- 追加的 7 个统计量（只 full 图有） ----
    // ndeg 邻居平均度 / hhi 强度集中度 / jacc 邻域 Jaccard / leafnb 叶子邻居占比
    let nw = (n + 63) / 64;
    let mut nbit = vec![vec![0u64; nw]; n];
    for i in 0..n {
        for &(v, _) in &g.adj[i] {
            nbit[i][v as usize / 64] |= 1u64 << (v as usize % 64);
        }
    }
    let degu: Vec<f64> = g.adj.iter().map(|a| a.len() as f64).collect();
    // 每个节点的邻居均度（assort 外延与 ndeg 列共用）
    let ndeg_arr: Vec<f64> = g.adj.iter().map(|a| {
        if a.is_empty() { f64::NAN }
        else { a.iter().map(|&(v, _)| degu[v as usize]).sum::<f64>() / a.len() as f64 }
    }).collect();
    let str_v: Vec<f64> = cols[1].clone();
    // ---- 抗极端值扩展层需要的共享中间量 ----
    let kcr = kcore(g);
    let mut kcr_sorted: Vec<u32> = kcr.clone();
    kcr_sorted.sort_unstable();
    let kc_med = if kcr_sorted.is_empty() { 0.0 } else { kcr_sorted[kcr_sorted.len() / 2] as f64 };
    let betw_arr: Vec<f64> = cols[13].clone();   // betw（已有列）
    let pr_arr: Vec<f64> = cols[14].clone();     // pagerank
    let dm_arr: Vec<f64> = cols[19].clone();     // dist_mkt
    let labs = &lab;
    let mut deg_sorted: Vec<f64> = degu.clone();
    deg_sorted.sort_by(|a, b| a.total_cmp(b));
    // newstats[i][k]：28 个统计
    // 0 ndeg 1 hhi 2 jacc 3 leafnb 4 d25 5 d75 6 assort
    // 7 str_p25 8 str_p50 9 str_p90 10 str_t10m 11 str_cv
    // 12 ndeg_p50 13 reach3 14 pdl50 15 dist_p90 16 dist_iqr
    // 17 bet_t10 18 pr_log 19 tri_p75 20 corefrac
    // 21 comm_top_sim 22 comm_edges_out 23 comm_deg_in
    // 24 dist_mkt_top10 25 lead_bet_mean 26 local_density 27 deg_rank
    // （mstreach 在最后单独算）
    let newstats: Vec<[f64; 28]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut o = [f64::NAN; 28];
            let a = &g.adj[i];
            if a.is_empty() {
                return o;
            }
            let di = a.len() as f64;
            // ---- 邻居边权排序后的分位数与稳健均值 ----
            let mut sw: Vec<f64> = a.iter().map(|&(_, x)| x as f64).collect();
            sw.sort_by(|x, y| x.total_cmp(y));
            let m = sw.len();
            let qt = |q: f64| -> f64 {
                let pos = q * (m - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(m - 1);
                sw[lo] + (sw[hi] - sw[lo]) * (pos - lo as f64)
            };
            o[7] = qt(0.25);
            o[8] = qt(0.50);
            o[9] = qt(0.90);
            let kt = ((m as f64 * 0.10).ceil() as usize).max(1);
            o[10] = sw[m - kt..].iter().sum::<f64>() / kt as f64;  // str_t10m
            let mw = str_v[i] / di;
            let sd = (sw.iter().map(|&x| (x - mw) * (x - mw)).sum::<f64>() / di).sqrt();
            o[11] = if mw.abs() > 1e-12 { sd / mw.abs() } else { f64::NAN }; // str_cv
            // ---- 邻居结构 ----
            o[0] = a.iter().map(|&(v, _)| degu[v as usize]).sum::<f64>() / di; // ndeg
            let mut nb_deg: Vec<f64> = a.iter().map(|&(v, _)| degu[v as usize]).collect();
            nb_deg.sort_by(|x, y| x.total_cmp(y));
            let dp = nb_deg.len();
            o[12] = nb_deg[dp / 2]; // ndeg_p50
            // reach3/local_density 用位图并集
            {
                let mut ball = vec![0u64; nw];
                for &(v, _) in a {
                    ball[v as usize / 64] |= 1u64 << (v as usize % 64);
                    for w in 0..nw {
                        ball[w] |= nbit[v as usize][w];
                    }
                }
                // ball = 两步内；再加一步
                let mut b3 = ball.clone();
                let mut cnt2 = 0u32;
                for w in 0..nw {
                    cnt2 += ball[w].count_ones();
                }
                // 两步球内逐节点并其邻居 → 三步球
                for j in 0..n {
                    if (ball[j / 64] >> (j % 64)) & 1 == 1 {
                        for w in 0..nw {
                            b3[w] |= nbit[j][w];
                        }
                    }
                }
                let mut cnt3 = 0u32;
                for w in 0..nw {
                    cnt3 += b3[w].count_ones();
                }
                o[13] = (cnt3 as i64 - 1 - (if a.iter().any(|&(v,_)| v as usize == i){1}else{0})) as f64; // reach3(不含自己)
                // local_density: 两步球内已有边/可能边
                let ball_n = cnt2 as usize;
                if ball_n >= 2 {
                    let mut e_in = 0usize;
                    for j in 0..n {
                        if (ball[j / 64] >> (j % 64)) & 1 == 1 {
                            for &(v, _) in &g.adj[j] {
                                let vi = v as usize;
                                if vi > j && (ball[vi / 64] >> (vi % 64)) & 1 == 1 {
                                    e_in += 1;
                                }
                            }
                        }
                    }
                    let poss = ball_n * (ball_n - 1) / 2;
                    o[26] = e_in as f64 / poss as f64;
                }
            }
            // ---- hhi / jacc / leafnb / assort（原有 4 个） ----
            let swsum = str_v[i];
            if swsum > 0.0 {
                o[1] = a.iter().map(|&(_, x)| { let q = x as f64 / swsum; q * q }).sum::<f64>();
            }
            let mut js = 0.0;
            let mut leaf = 0.0;
            let mut triw: Vec<f64> = Vec::new();
            for &(v, sij) in a {
                let vi = v as usize;
                let mut inter = 0usize;
                for w in 0..nw {
                    inter += (nbit[i][w] & nbit[vi][w]).count_ones() as usize;
                }
                let uni = a.len() + g.adj[vi].len() - inter;
                if uni > 0 { js += inter as f64 / uni as f64; }
                if g.adj[vi].len() == 1 { leaf += 1.0; }
                // 三角形强度（正积才计入，负边三角形跳过）
                for &(k, sjk) in &g.adj[vi] {
                    let ki = k as usize;
                    if ki == i { continue; }
                    if (nbit[i][ki / 64] >> (ki % 64)) & 1 == 1 && ki > vi {
                        if let Some(&(_, ski)) = a.iter().find(|&&(x, _)| x == k) {
                            let prod = sij as f64 * sjk as f64 * ski as f64;
                            if prod > 0.0 { triw.push(prod.powf(1.0 / 3.0)); }
                        }
                    }
                }
            }
            o[2] = js / di;
            o[3] = leaf / di;
            // assort（外延邻居均度版）
            {
                let (mut xa, mut ya) = (Vec::new(), Vec::new());
                for &(v, _) in a {
                    let vi = v as usize;
                    let dv = g.adj[vi].len();
                    if dv > 1 {
                        let yv = (ndeg_arr[vi] * dv as f64 - di) / (dv - 1) as f64;
                        xa.push(dv as f64); ya.push(yv);
                    }
                }
                if xa.len() >= 3 {
                    let mm = xa.len() as f64;
                    let (mx, my) = (xa.iter().sum::<f64>() / mm, ya.iter().sum::<f64>() / mm);
                    let (mut num, mut vx, mut vy) = (0.0, 0.0, 0.0);
                    for k in 0..xa.len() {
                        num += (xa[k] - mx) * (ya[k] - my);
                        vx += (xa[k] - mx) * (xa[k] - mx);
                        vy += (ya[k] - my) * (ya[k] - my);
                    }
                    if vx > 0.0 && vy > 0.0 { o[6] = num / (vx * vy).sqrt(); }
                }
            }
            // tri_p75：节点参与三角形强度（(sij·sjk·ski)^(1/3)）的 75 分位
            if !triw.is_empty() {
                triw.sort_by(|x, y| x.total_cmp(y));
                let pos = 0.75 * (triw.len() - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(triw.len() - 1);
                o[19] = triw[lo] + (triw[hi] - triw[lo]) * (pos - lo as f64);
            }
            // corefrac：邻居中 kcore>=全局中位数的占比
            o[20] = a.iter().filter(|&&(v, _)| kcr[v as usize] as f64 >= kc_med).count() as f64 / di;
            // ---- 中心性稳健版 ----
            o[14] = pdl50[i];
            o[15] = dist_p90[i];
            o[4] = d25[i];
            o[5] = d75[i];
            if d25[i].is_finite() && d75[i].is_finite() { o[16] = d75[i] - d25[i]; }
            // bet_t10：邻居 betw 最强 10% 的均值
            {
                let mut bw: Vec<f64> = a.iter()
                    .map(|&(v, _)| betw_arr[v as usize])
                    .filter(|x| x.is_finite())
                    .collect();
                if !bw.is_empty() {
                    bw.sort_by(|x, y| y.total_cmp(x));
                    let kb = ((bw.len() as f64 * 0.10).ceil() as usize).max(1);
                    o[17] = bw[..kb].iter().sum::<f64>() / kb as f64;
                }
            }
            if pr_arr[i].is_finite() && pr_arr[i] > 0.0 { o[18] = pr_arr[i].ln(); }
            // ---- 社区 ----
            {
                let mut w_in: Vec<f64> = Vec::new();
                let (mut s_in, mut s_all) = (0.0, 0.0);
                let mut c_in = 0usize;
                for &(v, w) in a {
                    s_all += w as f64;
                    if labs[v as usize] == labs[i] {
                        s_in += w as f64;
                        w_in.push(w as f64);
                        c_in += 1;
                    }
                }
                if s_all > 0.0 { o[22] = 1.0 - s_in / s_all; }        // comm_edges_out
                o[23] = c_in as f64 / di;                            // comm_deg_in
                if !w_in.is_empty() {
                    w_in.sort_by(|x, y| y.total_cmp(x));
                    let kb = (w_in.len() / 2).max(1).min(3);
                    o[21] = w_in[..kb].iter().sum::<f64>() / kb as f64; // comm_top_sim
                }
            }
            // ---- 锚点/市场 ----
            {
                // dist_mkt_top10：邻居"经它到市场锚点"距离(dm[v]+w)最小的 10% 均值
                let mut dv: Vec<f64> = a.iter()
                    .map(|&(v, w)| dm_arr[v as usize] + w as f64)
                    .filter(|x| x.is_finite())
                    .collect();
                if !dv.is_empty() {
                    dv.sort_by(|x, y| x.total_cmp(y));
                    let kb = ((dv.len() as f64 * 0.10).ceil() as usize).max(1);
                    o[24] = dv[..kb].iter().sum::<f64>() / kb as f64;
                }
                // lead_bet_mean：邻居里属于 lead_set 的 betw 均值
                let bs: Vec<f64> = a.iter()
                    .filter(|&&(v, _)| lead_set[v as usize])
                    .map(|&(v, _)| betw_arr[v as usize])
                    .filter(|x| x.is_finite())
                    .collect();
                if !bs.is_empty() {
                    o[25] = bs.iter().sum::<f64>() / bs.len() as f64;
                }
            }
            // deg_rank：度在全市场分位
            {
                let di0 = a.len() as f64;
                let pos = deg_sorted.partition_point(|&x| x < di0);
                o[27] = pos as f64 / deg_sorted.len() as f64;
            }
            o
        })
        .collect();
    // 先把原有的 ndeg/hhi/jacc/leafnb/d25/d75/assort 7 个推回 cols
    for k in 0..7 {
        cols.push(newstats.iter().map(|t| match k {
            0 => t[0], 1 => t[1], 2 => t[2], 3 => t[3],
            4 => t[4], 5 => t[5], _ => t[6],
        }).collect());
    }
    // 新 21 个（7..27 共 21 个）
    for k in 7..28 {
        cols.push(newstats.iter().map(|t| t[k]).collect());
    }
    // mstreach：三步球占全市场比例（补充 reach2 绝对数）
    let n_f = n as f64;
    cols.push(newstats.iter().map(|t| if t[13].is_finite() { t[13] / n_f } else { f64::NAN }).collect());
    cols
}

/// 到"市场"（多源虚拟节点，边权 1-corr_i,mkt）的距离
fn dist_to_mkt(g: &UGraph, mkt_corr: &[f64]) -> Vec<f64> {
    let n = g.n;
    let mut d = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    for i in 0..n {
        let c = mkt_corr[i];
        if c.is_finite() {
            let w = (1.0 - c).max(1e-9);
            d[i] = w;
            heap.push(State { d: w, u: i });
        }
    }
    while let Some(State { d: du, u }) = heap.pop() {
        if du > d[u] + 1e-12 {
            continue;
        }
        for &(v, sim) in &g.adj[u] {
            let w = (1.0 - sim as f64).max(1e-9);
            if du + w < d[v as usize] - 1e-12 {
                d[v as usize] = du + w;
                heap.push(State { d: du + w, u: v as usize });
            }
        }
    }
    d.iter().map(|&x| if x.is_finite() { x } else { f64::NAN }).collect()
}

/// 到指定节点集合的最短路（多源 Dijkstra）
fn dist_to_set(g: &UGraph, set: &[bool]) -> Vec<f64> {
    let n = g.n;
    let mut d = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    for i in 0..n {
        if set[i] {
            d[i] = 0.0;
            heap.push(State { d: 0.0, u: i });
        }
    }
    while let Some(State { d: du, u }) = heap.pop() {
        if du > d[u] + 1e-12 {
            continue;
        }
        for &(v, sim) in &g.adj[u] {
            let w = (1.0 - sim as f64).max(1e-9);
            if du + w < d[v as usize] - 1e-12 {
                d[v as usize] = du + w;
                heap.push(State { d: du + w, u: v as usize });
            }
        }
    }
    d.iter().map(|&x| if x.is_finite() { x } else { f64::NAN }).collect()
}

// ---------------------------------------------------------------------------
// MST 附加统计
// ---------------------------------------------------------------------------

fn mst_extra(g: &UGraph, mkt_corr: &[f64]) -> Vec<Vec<f64>> {
    let n = g.n;
    // treedepth: 找一个中心根 = 调和中心最大的点，BFS 深度
    let (harm, _, _) = dist_stats(g, n_src_of(n));
    let root = harm
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i);
    let mut depth = vec![f64::NAN; n];
    if let Some(r) = root {
        let mut dd = vec![-1i32; n];
        dd[r] = 0;
        let mut q = std::collections::VecDeque::from([r]);
        while let Some(u) = q.pop_front() {
            for &(v, _) in &g.adj[u] {
                if dd[v as usize] < 0 {
                    dd[v as usize] = dd[u] + 1;
                    q.push_back(v as usize);
                }
            }
        }
        for i in 0..n {
            depth[i] = if dd[i] >= 0 { dd[i] as f64 } else { f64::NAN };
        }
    }
    // subtree mass: 以 root 为根, 子树大小
    let mut submass = vec![f64::NAN; n];
    let mut parent = vec![usize::MAX; n];
    if let Some(r) = root {
        // BFS 建树
        let mut order = vec![r];
        let mut seen = vec![false; n];
        seen[r] = true;
        let mut qi = 0;
        while qi < order.len() {
            let u = order[qi];
            qi += 1;
            for &(v, _) in &g.adj[u] {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    parent[v as usize] = u;
                    order.push(v as usize);
                }
            }
        }
        let mut sz = vec![1u32; n];
        for &u in order.iter().rev() {
            if parent[u] != usize::MAX {
                sz[parent[u]] += sz[u];
            }
        }
        for i in 0..n {
            submass[i] = if seen[i] { sz[i] as f64 } else { f64::NAN };
        }
    }
    // is_art: 孩子数>0 且非根非叶 → 近似"内部分叉节点"
    let mut is_art = vec![0.0f64; n];
    let mut leaf = vec![f64::NAN; n];
    for i in 0..n {
        let d = g.deg(i);
        leaf[i] = if d == 1 { 1.0 } else { 0.0 };
        if d >= 3 {
            is_art[i] = 1.0;
        }
    }
    // hops_mkt: 到市场锚点（mkt_corr 最大者）在 MST 上的跳数
    let anchor = mkt_corr
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i);
    let mut hops = vec![f64::NAN; n];
    if let Some(a0) = anchor {
        let mut dist = vec![-1i32; n];
        dist[a0] = 0;
        let mut q = std::collections::VecDeque::from([a0]);
        while let Some(u) = q.pop_front() {
            for &(v, _) in &g.adj[u] {
                if dist[v as usize] < 0 {
                    dist[v as usize] = dist[u] + 1;
                    q.push_back(v as usize);
                }
            }
        }
        for i in 0..n {
            hops[i] = if dist[i] >= 0 { dist[i] as f64 } else { f64::NAN };
        }
    }
    vec![submass, is_art, depth, hops, leaf]
}

// ---------------------------------------------------------------------------
// 有向图统计
// ---------------------------------------------------------------------------

fn dir_stats(g: &DGraph) -> Vec<Vec<f64>> {
    let n = g.n;
    let mut outstr = vec![0.0; n];
    let mut instr = vec![0.0; n];
    let mut outdeg = vec![0.0; n];
    let mut indeg = vec![0.0; n];
    for i in 0..n {
        outdeg[i] = g.out_adj[i].len() as f64;
        indeg[i] = g.in_adj[i].len() as f64;
        outstr[i] = g.out_adj[i].iter().map(|&(_, s)| s as f64).sum();
        instr[i] = g.in_adj[i].iter().map(|&(_, s)| s as f64).sum();
    }
    let netlead: Vec<f64> = outstr.iter().zip(&instr).map(|(a, b)| a - b).collect();
    let leadshare: Vec<f64> = outstr
        .iter()
        .zip(&instr)
        .map(|(a, b)| if a + b > 0.0 { a / (a + b) } else { f64::NAN })
        .collect();
    // 有向 PageRank（沿出边走）
    let mut pr = vec![1.0 / n as f64; n];
    let mut next = vec![0.0; n];
    for _ in 0..50 {
        let mut dangling = 0.0;
        for i in 0..n {
            next[i] = 0.15 / n as f64;
            if outstr[i] <= 0.0 {
                dangling += pr[i];
                continue;
            }
            for &(v, s) in &g.out_adj[i] {
                next[v as usize] += 0.85 * pr[i] * s as f64 / outstr[i];
            }
        }
        for i in 0..n {
            next[i] += 0.85 * dangling / n as f64;
        }
        std::mem::swap(&mut pr, &mut next);
        next.fill(0.0);
    }
    // netpr: 用入边方向的 PageRank（"被影响"分）- 出向（"影响"分）
    let mut pr2 = vec![1.0 / n as f64; n];
    let mut next2 = vec![0.0; n];
    for _ in 0..50 {
        let mut dangling = 0.0;
        for i in 0..n {
            next2[i] = 0.15 / n as f64;
            if instr[i] <= 0.0 {
                dangling += pr2[i];
                continue;
            }
            for &(v, s) in &g.in_adj[i] {
                next2[v as usize] += 0.85 * pr2[i] * s as f64 / instr[i];
            }
        }
        for i in 0..n {
            next2[i] += 0.85 * dangling / n as f64;
        }
        std::mem::swap(&mut pr2, &mut next2);
        next2.fill(0.0);
    }
    let netpr: Vec<f64> = pr.iter().zip(&pr2).map(|(a, b)| a - b).collect();
    let prlead: Vec<f64> = pr.iter().zip(&pr2).map(|(a, b)| a / b.max(1e-12)).collect();

    // 出/入调和距离（沿出边 / 沿入边多源... 简单版: 从每点沿出边到所有点 —— 用反向 BFS 近似跳数）
    let outh = dir_harmonic(g, false);
    let inh = dir_harmonic(g, true);
    // outreach/inreach: 出/入 k 跳可达计数（BFS 到 2 跳）
    let mut outreach = vec![0.0; n];
    let mut inreach = vec![0.0; n];
    for i in 0..n {
        let mut mark = vec![false; n];
        mark[i] = true;
        for &(v, _) in &g.out_adj[i] {
            mark[v as usize] = true;
            for &(w, _) in &g.out_adj[v as usize] {
                mark[w as usize] = true;
            }
        }
        outreach[i] = mark.iter().filter(|&&b| b).count() as f64 - 1.0;
        mark = vec![false; n];
        mark[i] = true;
        for &(v, _) in &g.in_adj[i] {
            mark[v as usize] = true;
            for &(w, _) in &g.in_adj[v as usize] {
                mark[w as usize] = true;
            }
        }
        inreach[i] = mark.iter().filter(|&&b| b).count() as f64 - 1.0;
    }
    vec![
        outstr, instr, netlead, outdeg, indeg, leadshare, netpr, prlead, outh, inh, outreach,
        inreach,
    ]
}

/// 沿有向边（out=沿出边多远能辐射, in=沿入边多远被辐射）的调和中心
fn dir_harmonic(g: &DGraph, use_in: bool) -> Vec<f64> {
    let n = g.n;
    let adj = if use_in { &g.in_adj } else { &g.out_adj };
    (0..n)
        .into_par_iter()
        .map(|s| {
            let mut d = vec![f64::INFINITY; n];
            d[s] = 0.0;
            let mut heap = BinaryHeap::new();
            heap.push(State { d: 0.0, u: s });
            while let Some(State { d: du, u }) = heap.pop() {
                if du > d[u] + 1e-12 {
                    continue;
                }
                for &(v, sim) in &adj[u] {
                    let w = (1.0 - sim as f64).max(1e-9);
                    if du + w < d[v as usize] - 1e-12 {
                        d[v as usize] = du + w;
                        heap.push(State { d: du + w, u: v as usize });
                    }
                }
            }
            let mut h = 0.0;
            let mut cnt = 0;
            for (j, &x) in d.iter().enumerate() {
                if j != s && x.is_finite() && x > 0.0 {
                    h += 1.0 / x;
                    cnt += 1;
                }
            }
            if cnt == 0 { f64::NAN } else { h / cnt as f64 }
        })
        .collect()
}

// ============================================================================
// 相似度矩阵
// ============================================================================

/// 列居中矩阵 + 每列 L2 模（sd<=0 的列模为 0 → 相关=NaN）
fn centered_cols(rets: &Array2<f64>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (t, n) = (rets.nrows(), rets.ncols());
    let mean: Vec<f64> = (0..n).map(|j| rets.column(j).sum() / t as f64).collect();
    let cols: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let m = mean[j];
            rets.column(j).iter().map(|&x| x - m).collect()
        })
        .collect();
    let sd: Vec<f64> = cols
        .iter()
        .map(|c| c.iter().map(|&x| x * x).sum::<f64>().sqrt())
        .collect();
    (cols, sd)
}

/// 用预先中心化的列算皮尔逊（session 切片复用，省重复中心化）
fn pearson_from(cols: &[Vec<f64>], sd: &[f64]) -> Array2<f64> {
    let n = cols.len();
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n];
            if sd[i] > 0.0 {
                let ci = &cols[i];
                for j in (i + 1)..n {
                    if sd[j] <= 0.0 {
                        continue;
                    }
                    let mut num = 0.0;
                    let cj = &cols[j];
                    for k in 0..ci.len() {
                        num += ci[k] * cj[k];
                    }
                    row[j] = num / (sd[i] * sd[j]);
                }
            }
            row
        })
        .collect();
    for (i, row) in rows.into_iter().enumerate() {
        for j in (i + 1)..n {
            out[[i, j]] = row[j];
            out[[j, i]] = row[j];
        }
    }
    out
}

/// 列标准化皮尔逊: rets (T,N), nan=0, sd<=0 列全 NaN
fn pearson_mat(rets: &Array2<f64>) -> Array2<f64> {
    let (cols, sd) = centered_cols(rets);
    pearson_from(&cols, &sd)
}

/// 符号一致率（两者都非零才计入）
fn sign_mat(rets: &Array2<f64>, min_pair: usize) -> Array2<f64> {
    let (t, n) = (rets.nrows(), rets.ncols());
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n];
            let ci = rets.column(i);
            for j in (i + 1)..n {
                let cj = rets.column(j);
                let (mut same, mut tot) = (0usize, 0usize);
                for k in 0..t {
                    let (a, b) = (ci[k], cj[k]);
                    if a != 0.0 && b != 0.0 {
                        tot += 1;
                        if (a > 0.0) == (b > 0.0) {
                            same += 1;
                        }
                    }
                }
                if tot >= min_pair {
                    row[j] = same as f64 / tot as f64;
                }
            }
            row
        })
        .collect();
    for (i, row) in rows.into_iter().enumerate() {
        for j in (i + 1)..n {
            out[[i, j]] = row[j];
            out[[j, i]] = row[j];
        }
    }
    out
}

/// 上行尾部相关: 只在 mkt[t]>0 的行上算 pearson
fn tail_mat(rets: &Array2<f64>, mkt: &[f64]) -> Array2<f64> {
    let rows: Vec<usize> = (0..rets.nrows()).filter(|&t| mkt[t] > 0.0).collect();
    if rows.len() < 20 {
        return Array2::<f64>::from_elem((rets.ncols(), rets.ncols()), f64::NAN);
    }
    let sub = rets.select(ndarray::Axis(0), &rows);
    pearson_mat(&sub)
}

/// 领先-滞后（非对称）: ll[i,j] = corr(r_i[t], r_j[t+1]) - corr(r_i[t], r_j[t-1])
/// >0 表示 i 领先 j；建图取 >thr 作 i→j 边。
fn leadlag_mat(rets: &Array2<f64>) -> Array2<f64> {
    let (t, n) = (rets.nrows(), rets.ncols());
    let (cols, sd) = centered_cols(rets);
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n];
            if sd[i] <= 0.0 {
                return row;
            }
            let ci = &cols[i];
            for j in 0..n {
                if j == i || sd[j] <= 0.0 {
                    continue;
                }
                let cj = &cols[j];
                let (mut fwd, mut bwd) = (0.0, 0.0);
                for k in 0..t - 1 {
                    fwd += ci[k] * cj[k + 1];
                }
                for k in 1..t {
                    bwd += ci[k] * cj[k - 1];
                }
                let denom = sd[i] * sd[j];
                row[j] = (fwd - bwd) / denom;
            }
            row
        })
        .collect();
    for (i, row) in rows.into_iter().enumerate() {
        for j in 0..n {
            out[[i, j]] = row[j];
        }
    }
    out
}

/// 0/1 矩阵的 Jaccard 相似度: |A∩B|/|A∪B|，用 u64 位掩码加速
fn jaccard_mat(rets: &Array2<f64>) -> Array2<f64> {
    let (t, n) = (rets.nrows(), rets.ncols());
    let nw = (t + 63) / 64;
    let mut bits = vec![vec![0u64; nw]; n];
    let mut acnt = vec![0usize; n];
    for j in 0..n {
        for i in 0..t {
            if rets[[i, j]] != 0.0 {
                bits[j][i / 64] |= 1u64 << (i % 64);
                acnt[j] += 1;
            }
        }
    }
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n];
            for j in (i + 1)..n {
                let mut inter = 0usize;
                for w in 0..nw {
                    inter += (bits[i][w] & bits[j][w]).count_ones() as usize;
                }
                let uni = acnt[i] + acnt[j] - inter;
                if uni > 0 {
                    row[j] = inter as f64 / uni as f64;
                }
            }
            row
        })
        .collect();
    for (i, row) in rows.into_iter().enumerate() {
        for j in (i + 1)..n {
            out[[i, j]] = row[j];
            out[[j, i]] = row[j];
        }
    }
    out
}

/// 把每股某条 30s 序列堆成 (N_CELL, n) 矩阵（缺失股票列全 0）
fn stack_series(preps: &[Option<StockPrep>], n: usize, f: impl Fn(&StockPrep) -> &Vec<f64>) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((N_CELL, n));
    for (j, p) in preps.iter().enumerate() {
        if let Some(p) = p {
            let s = f(p);
            for i in 0..N_CELL {
                m[[i, j]] = s[i];
            }
        }
    }
    m
}

/// 序列编号 → prep_stock 里哪条 (N_CELL) 序列
/// 0=ret30 1=ofi 2=ret60 3=ofi60（0~3 在 SimBank 里直接拿现成矩阵）
fn series_vec<'a>(p: &'a StockPrep, k: u8) -> &'a Vec<f64> {
    match k {
        4 => &p.vol30, 5 => &p.tv30, 6 => &p.cnt30, 7 => &p.act30,
        8 => &p.dom30, 9 => &p.buyord30, 10 => &p.sellord30, 11 => &p.dimb30,
        12 => &p.spr30, 13 => &p.midr30, 14 => &p.ordbal30,
        16 => &p.actlag30, 17 => &p.rtn30, 18 => &p.domsell30, 19 => &p.imb30,
        20 => &p.vstd30, 21 => &p.vskew30, 22 => &p.vmaxshare30, 23 => &p.vgini30,
        24 => &p.amtcv30, 25 => &p.gapcv30, 26 => &p.gapstd30, 27 => &p.fhalf30,
        28 => &p.rndsec30, 29 => &p.tspan30, 30 => &p.tunif30,
        31 => &p.aslope30, 32 => &p.bslope30, 33 => &p.dskewa30, 34 => &p.dskewb30,
        35 => &p.l1conc30, 36 => &p.l1gap30, 37 => &p.dsdask30, 38 => &p.dsdbid30,
        39 => &p.dsdnet30, 40 => &p.big30, _ => &p.rba30,
    }
}

#[derive(Clone, Copy)]
enum SimKind { P, PJ, S, T, L, J }
struct FamSrc { series: u8, kind: SimKind }

/// 相似度族 → (序列id, 算法)
fn fam_src(fam: &str) -> FamSrc {
    match fam {
        "rd" => FamSrc { series: 0, kind: SimKind::P },
        "ra" => FamSrc { series: 0, kind: SimKind::PJ },
        "rs" => FamSrc { series: 0, kind: SimKind::S },
        "ro" => FamSrc { series: 1, kind: SimKind::P },
        "rl" => FamSrc { series: 0, kind: SimKind::L },
        "rd60" => FamSrc { series: 2, kind: SimKind::P },
        "ra60" => FamSrc { series: 2, kind: SimKind::PJ },
        "ro60" => FamSrc { series: 3, kind: SimKind::P },
        "rv" => FamSrc { series: 4, kind: SimKind::P },
        "rcnt" => FamSrc { series: 6, kind: SimKind::P },
        "ract" => FamSrc { series: 7, kind: SimKind::J },
        "rdpt" => FamSrc { series: 11, kind: SimKind::P },
        "rspr" => FamSrc { series: 12, kind: SimKind::P },
        "rmid" => FamSrc { series: 13, kind: SimKind::P },
        "rbal" => FamSrc { series: 14, kind: SimKind::P },
        "rlag" => FamSrc { series: 16, kind: SimKind::P },
        "rrtn" => FamSrc { series: 17, kind: SimKind::P },
        "rvstd" => FamSrc { series: 20, kind: SimKind::P },
        "rvskew" => FamSrc { series: 21, kind: SimKind::P },
        "rvmax" => FamSrc { series: 22, kind: SimKind::P },
        "ramtcv" => FamSrc { series: 24, kind: SimKind::P },
        "rgapcv" => FamSrc { series: 25, kind: SimKind::P },
        "rgapstd" => FamSrc { series: 26, kind: SimKind::P },
        "rfhalf" => FamSrc { series: 27, kind: SimKind::P },
        "rrnd" => FamSrc { series: 28, kind: SimKind::P },
        "rtunif" => FamSrc { series: 30, kind: SimKind::P },
        "rimb" => FamSrc { series: 19, kind: SimKind::P },
        "rl1c" => FamSrc { series: 35, kind: SimKind::P },
        "rl1g" => FamSrc { series: 36, kind: SimKind::P },
        "rdsda" => FamSrc { series: 37, kind: SimKind::P },
        "rdsdb" => FamSrc { series: 38, kind: SimKind::P },
        "rdsdn" => FamSrc { series: 39, kind: SimKind::P },
        _ => FamSrc { series: 0, kind: SimKind::P },
    }
}

/// 相似度矩阵库：序列 (T,N) + 中心化统计 + 成品种矩阵，三层懒缓存
struct SimBank {
    preps: Vec<Option<StockPrep>>,
    n: usize,
    r30: Array2<f64>, o30: Array2<f64>, r60: Array2<f64>, o60: Array2<f64>,
    mkt: Vec<f64>, mkt60: Vec<f64>,
    series: RefCell<Vec<Option<Rc<Array2<f64>>>>>,
    stats: RefCell<std::collections::HashMap<(u8, u8), Rc<(Vec<Vec<f64>>, Vec<f64>)>>>,
    sims: RefCell<std::collections::HashMap<(u8, u8), Rc<Array2<f64>>>>,
}

fn sess_u8(s: Sess) -> u8 {
    match s { Sess::Day => 0, Sess::Pm => 1, Sess::Tl => 2 }
}

impl SimBank {
    fn range_of(&self, t: usize, sess: Sess) -> (usize, usize) {
        if t == N_CELL {
            match sess {
                Sess::Day => (0, N_CELL),
                Sess::Pm => (PM_SPLIT, N_CELL),
                Sess::Tl => (TAIL_START, N_CELL),
            }
        } else {
            match sess {
                Sess::Day => (0, N_CELL60),
                Sess::Pm => (PM_SPLIT / 2, N_CELL60),
                Sess::Tl => (TAIL_START / 2, N_CELL60),
            }
        }
    }

    fn series_mat(&self, k: u8) -> Rc<Array2<f64>> {
        if let Some(m) = &self.series.borrow()[k as usize] {
            return m.clone();
        }
        let m = match k {
            0 => self.r30.clone(),
            1 => self.o30.clone(),
            2 => self.r60.clone(),
            3 => self.o60.clone(),
            _ => stack_series(&self.preps, self.n, |p| series_vec(p, k)),
        };
        let rc = Rc::new(m);
        self.series.borrow_mut()[k as usize] = Some(rc.clone());
        rc
    }

    fn centered(&self, k: u8, sess: Sess) -> Rc<(Vec<Vec<f64>>, Vec<f64>)> {
        let key = (k, sess_u8(sess));
        if let Some(c) = self.stats.borrow().get(&key) {
            return c.clone();
        }
        let m = self.series_mat(k);
        let (lo, hi) = self.range_of(m.nrows(), sess);
        let sub = m.slice(ndarray::s![lo..hi, ..]).to_owned();
        let st = Rc::new(centered_cols(&sub));
        self.stats.borrow_mut().insert(key, st.clone());
        st
    }

    fn pearson_sess(&self, k: u8, sess: Sess) -> Array2<f64> {
        let st = self.centered(k, sess);
        pearson_from(&st.0, &st.1)
    }

    fn sim(&self, fam: &str, sess: Sess) -> Rc<Array2<f64>> {
        let fs = fam_src(fam);
        let key = (fs.series, sess_u8(sess));
        if let Some(m) = self.sims.borrow().get(&key) {
            return m.clone();
        }
        let full = self.series_mat(fs.series);
        let t = full.nrows();
        let (lo, hi) = self.range_of(t, sess);
        let sub = full.slice(ndarray::s![lo..hi, ..]).to_owned();
        let m = match fs.kind {
            SimKind::P => self.pearson_sess(fs.series, sess),
            SimKind::PJ => {
                let mut a = sub;
                for v in a.iter_mut() { *v = v.abs(); }
                pearson_mat(&a)
            }
            SimKind::S => sign_mat(&sub, MIN_PAIR),
            SimKind::T => {
                let mv = if t == N_CELL { &self.mkt } else { &self.mkt60 };
                tail_mat(&sub, &mv[lo..hi])
            }
            SimKind::L => leadlag_mat(&sub),
            SimKind::J => jaccard_mat(&sub),
        };
        let rc = Rc::new(m);
        self.sims.borrow_mut().insert(key, rc.clone());
        rc
    }
}

// ============================================================================
// 矩阵直出统计（不建图）
// ============================================================================

fn matrix_stats(s: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = s.nrows();
    let mut simmax = vec![f64::NAN; n];
    let mut simtop3 = vec![f64::NAN; n];
    let mut simtop10 = vec![f64::NAN; n];
    let mut simmean = vec![f64::NAN; n];
    let mut simp90 = vec![f64::NAN; n];
    let mut frac02 = vec![f64::NAN; n];
    let mut frac03 = vec![f64::NAN; n];
    for i in 0..n {
        let mut v: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .filter_map(|j| {
                let x = s[[i, j]];
                if x.is_finite() { Some(x) } else { None }
            })
            .collect();
        if v.is_empty() {
            continue;
        }
        v.sort_by(|a, b| b.total_cmp(a));
        simmax[i] = v[0];
        let t3 = v.len().min(3);
        simtop3[i] = v[..t3].iter().sum::<f64>() / t3 as f64;
        let t10 = v.len().min(10);
        simtop10[i] = v[..t10].iter().sum::<f64>() / t10 as f64;
        simmean[i] = v.iter().sum::<f64>() / v.len() as f64;
        simp90[i] = v[(v.len() as f64 * 0.1) as usize];
        frac02[i] = v.iter().filter(|&&x| x > 0.2).count() as f64 / v.len() as f64;
        frac03[i] = v.iter().filter(|&&x| x > 0.3).count() as f64 / v.len() as f64;
    }
    vec![simmax, simtop3, simtop10, simmean, simp90, frac02, frac03]
}

const MATRIX_STAT_NAMES: [&str; 7] = [
    "simmax", "simtop3", "simtop10", "simmean", "simp90", "simfrac02", "simfrac03",
];

// ============================================================================
// 因子名注册表（唯一真相源）
// ============================================================================

struct GraphSpec {
    tag: String,
    fam: &'static str,
    kind: Kind,
    full: bool,
    sess: Sess,
}
#[derive(Clone, Copy, PartialEq)]
enum Sess { Day, Pm, Tl }
#[derive(Clone, Copy)]
enum Kind {
    Thr(f64),
    KnnU(usize),
    KnnM(usize),
    Mst,
    IndThr(f64),
    DirThr(f64),
}

fn gs(tag: &str, fam: &'static str, kind: Kind, full: bool, sess: Sess) -> GraphSpec {
    GraphSpec { tag: tag.to_string(), fam, kind, full, sess }
}

struct FamDef { fam: &'static str, t1: f64, t2: f64, k: usize, km: usize, has_mst: bool }
fn thr_t(t: f64) -> String { format!("thr{}", (t * 100.0).round() as i64) }

/// day=全量统计 full_all=true；PM/TL 时 thr 两档仍 full、其余缩减（A 方案）
/// full4 = (t1, t2, knnU/M, mst) 各档是否跑 29 统计
/// 实测档位（lead 2026-11-02）：第一档 day 五张全量超 10 分钟，
/// 降为第二档 day 只留 thr(t2)+mst 全量，t1/knnU/knnM 缩减；PM/TL 维持 thr×2 全量。
fn emit_fam(out: &mut Vec<GraphSpec>, fd: &FamDef, sess: Sess, grid: &str,
            f_t1: bool, f_t2: bool, f_knn: bool, f_mst: bool) {
    // grid 是 "g30"/"g60"；pm/tl 前缀用格子数：pm30_/tl30_/pm60_/tl60_
    let gnum = grid.trim_start_matches('g');
    let pre = match sess {
        Sess::Day => format!("{}_{}", grid, fd.fam),
        Sess::Pm => format!("pm{}_{}", gnum, fd.fam),
        Sess::Tl => format!("tl{}_{}", gnum, fd.fam),
    };
    out.push(gs(&format!("{}_{}", pre, thr_t(fd.t1)), fd.fam, Kind::Thr(fd.t1), f_t1, sess));
    out.push(gs(&format!("{}_{}", pre, thr_t(fd.t2)), fd.fam, Kind::Thr(fd.t2), f_t2, sess));
    out.push(gs(&format!("{}_knnU{}", pre, fd.k), fd.fam, Kind::KnnU(fd.k), f_knn, sess));
    out.push(gs(&format!("{}_knnM{}", pre, fd.km), fd.fam, Kind::KnnM(fd.km), f_knn, sess));
    if fd.has_mst {
        out.push(gs(&format!("{}_mst", pre), fd.fam, Kind::Mst, f_mst, sess));
    }
}

const NEW_FAMS: &[FamDef] = &[
    FamDef { fam: "rv", t1: 0.20, t2: 0.30, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rcnt", t1: 0.20, t2: 0.30, k: 30, km: 10, has_mst: true },
    FamDef { fam: "ract", t1: 0.05, t2: 0.10, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rbal", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rdpt", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rspr", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rmid", t1: 0.20, t2: 0.30, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rlag", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rrtn", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rvstd", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rvskew", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rvmax", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "ramtcv", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rgapcv", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rgapstd", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rfhalf", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rrnd", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rtunif", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rimb", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rl1c", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rl1g", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rdsda", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rdsdb", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rdsdn", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
];

/// 老族 PM/TL 段沿用各自阈值（day 段维持原表）
const OLD_PM_DEFS: &[FamDef] = &[
    FamDef { fam: "rd", t1: 0.20, t2: 0.30, k: 30, km: 10, has_mst: true },
    FamDef { fam: "ra", t1: 0.25, t2: 0.35, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rs", t1: 0.08, t2: 0.20, k: 30, km: 10, has_mst: true },
    FamDef { fam: "ro", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
    FamDef { fam: "rd60", t1: 0.20, t2: 0.30, k: 30, km: 10, has_mst: true },
    FamDef { fam: "ra60", t1: 0.25, t2: 0.35, k: 30, km: 10, has_mst: true },
    FamDef { fam: "ro60", t1: 0.15, t2: 0.25, k: 30, km: 10, has_mst: true },
];

static GRAPH_SPECS: Lazy<Vec<GraphSpec>> = Lazy::new(|| {
    let mut v: Vec<GraphSpec> = Vec::new();
    macro_rules! G { ($t:expr, $f:expr, $k:expr, $fl:expr) => { v.push(gs($t, $f, $k, $fl, Sess::Day)); } }

    // ---- rd 家族 g30（主家族, 全量统计) ----
    G!("g30_rd_thr15", "rd", Kind::Thr(0.15), true);
    G!("g30_rd_thr20", "rd", Kind::Thr(0.20), true);
    G!("g30_rd_thr25", "rd", Kind::Thr(0.25), true);
    G!("g30_rd_thr30", "rd", Kind::Thr(0.30), true);
    G!("g30_rd_thr35", "rd", Kind::Thr(0.35), true);
    G!("g30_rd_knnU20", "rd", Kind::KnnU(20), true);
    G!("g30_rd_knnU30", "rd", Kind::KnnU(30), true);
    G!("g30_rd_knnU50", "rd", Kind::KnnU(50), true);
    G!("g30_rd_knnM5", "rd", Kind::KnnM(5), true);
    G!("g30_rd_knnM10", "rd", Kind::KnnM(10), true);
    G!("g30_rd_knnM15", "rd", Kind::KnnM(15), true);
    G!("g30_rd_knnM20", "rd", Kind::KnnM(20), true);
    G!("g30_rd_mst", "rd", Kind::Mst, true);
    G!("g30_rd_indT15", "rd", Kind::IndThr(0.15), true);
    G!("g30_rd_indT20", "rd", Kind::IndThr(0.20), true);
    G!("g30_rd_indT25", "rd", Kind::IndThr(0.25), true);
    // ---- rd 家族 g60（缩减） ----
    G!("g60_rd_thr20", "rd60", Kind::Thr(0.20), false);
    G!("g60_rd_thr30", "rd60", Kind::Thr(0.30), false);
    G!("g60_rd_mst", "rd60", Kind::Mst, false);
    // ---- ra ----
    G!("g30_ra_thr25", "ra", Kind::Thr(0.25), true);
    G!("g30_ra_thr35", "ra", Kind::Thr(0.35), true);
    G!("g30_ra_knnU30", "ra", Kind::KnnU(30), true);
    G!("g30_ra_knnM10", "ra", Kind::KnnM(10), true);
    G!("g30_ra_mst", "ra", Kind::Mst, true);
    G!("g30_ra_indT20", "ra", Kind::IndThr(0.20), true);
    G!("g60_ra_thr30", "ra60", Kind::Thr(0.30), false);
    // ---- rs（符号一致率 0~1 区间, 阈值为比例） ----
    G!("g30_rs_thr08", "rs", Kind::Thr(0.08), true);
    G!("g30_rs_thr15", "rs", Kind::Thr(0.15), true);
    G!("g30_rs_thr20", "rs", Kind::Thr(0.20), true);
    G!("g30_rs_knnU30", "rs", Kind::KnnU(30), true);
    G!("g30_rs_knnM10", "rs", Kind::KnnM(10), true);
    G!("g30_rs_mst", "rs", Kind::Mst, true);
    // ---- ro（主动买压） ----
    G!("g30_ro_thr15", "ro", Kind::Thr(0.15), true);
    G!("g30_ro_thr25", "ro", Kind::Thr(0.25), true);
    G!("g30_ro_knnU30", "ro", Kind::KnnU(30), true);
    G!("g30_ro_knnM10", "ro", Kind::KnnM(10), true);
    G!("g30_ro_mst", "ro", Kind::Mst, true);
    G!("g30_ro_indT20", "ro", Kind::IndThr(0.20), true);
    G!("g60_ro_thr20", "ro60", Kind::Thr(0.20), false);
    // ---- rl（有向） ----
    G!("g30_rl_thr05", "rl", Kind::DirThr(0.05), true);
    G!("g30_rl_thr10", "rl", Kind::DirThr(0.10), true);
    G!("g30_rl_thr15", "rl", Kind::DirThr(0.15), true);
    G!("g30_rl_thr20", "rl", Kind::DirThr(0.20), true);
    // ---- 新族 day 段（全量统计，thr×2+knnU+knnM+mst） ----
    for fd in NEW_FAMS {
        emit_fam(&mut v, fd, Sess::Day, "g30", false, true, false, true);
    }
    // ---- 老族 PM/TL 段（thr 两档 full，其余缩减 9） ----
    for fd in OLD_PM_DEFS {
        let grid = if fd.fam.contains("60") { "g60" } else { "g30" };
        emit_fam(&mut v, fd, Sess::Pm, grid, true, true, false, false);
        emit_fam(&mut v, fd, Sess::Tl, grid, true, true, false, false);
    }
    // ---- 新族 PM/TL 段 ----
    for fd in NEW_FAMS {
        emit_fam(&mut v, fd, Sess::Pm, "g30", true, true, false, false);
        emit_fam(&mut v, fd, Sess::Tl, "g30", true, true, false, false);
    }
    // ---- rl 有向 PM/TL 各两张（day 段已在上面手写，DirThr 12 统计） ----
    for sess in [Sess::Pm, Sess::Tl] {
        let pre = if sess == Sess::Pm { "pm30_rl" } else { "tl30_rl" };
        v.push(gs(&format!("{}_thr10", pre), "rl", Kind::DirThr(0.10), true, sess));
        v.push(gs(&format!("{}_thr20", pre), "rl", Kind::DirThr(0.20), true, sess));
    }
    v
});


/// 全部因子名（顺序 = vals 列序），懒初始化一次
static RELAY_NAMES: Lazy<Vec<String>> = Lazy::new(build_names);

fn build_names() -> Vec<String> {
    let mut v = Vec::with_capacity(1500);
    for sp in GRAPH_SPECS.iter() {
        match sp.kind {
            Kind::DirThr(_) => {
                for s in DIR_STATS {
                    v.push(format!("rlg_{}_{}", sp.tag, s));
                }
            }
            _ => {
                if sp.full {
                    for s in STATS_UNDIR {
                        v.push(format!("rlg_{}_{}", sp.tag, s));
                    }
                } else {
                    for s in REDUCED_STATS {
                        v.push(format!("rlg_{}_{}", sp.tag, s));
                    }
                }
                if let Kind::Mst = sp.kind {
                    for s in MST_EXTRA {
                        v.push(format!("rlg_{}_{}", sp.tag, s));
                    }
                }
            }
        }
    }
    // 行业内附加（在 ind 图上）
    for s in ["ind_str", "ind_deg", "ind_bet", "ind_zclose", "ind_top1"] {
        v.push(format!("rlg_g30_rd_ind_{}", s));
    }
    // 矩阵直出（rd/ra/rs/ro 在 g30, rd/rt 在 g60）
    for fam in ["rd", "ra", "rs", "ro"] {
        for s in MATRIX_STAT_NAMES {
            v.push(format!("rlg_mat30_{}_{}", fam, s));
        }
    }
    for fam in ["rd60", "ro60"] {
        for s in MATRIX_STAT_NAMES {
            v.push(format!("rlg_mat60_{}_{}", fam, s));
        }
    }
    // 新相似度矩阵直出（10 族 × 7 个）
    for fam in ["rv", "rcnt", "ract", "rdpt", "rspr", "rmid", "rlag"] {
        for s in MATRIX_STAT_NAMES {
            v.push(format!("rlg_mat30_{}_{}", fam, s));
        }
    }
    // 上午/下午分图（rd, thr20）10 统计 + delta
    for tag in ["am", "pm"] {
        for s in ["deg", "str", "bet", "harm", "pagerank", "kcore", "dist_mkt", "top1",
                  "compsize", "neighind"] {
            v.push(format!("rlg_{}30_rd_thr20_{}", tag, s));
        }
    }
    for s in ["deg", "str", "bet", "harm", "pagerank", "kcore", "dist_mkt", "top1",
              "compsize", "neighind"] {
        v.push(format!("rlg_delta_rd_thr20_{}", s));
    }
    // 市场相关与跨图汇总
    for s in ["mkt_corr30", "mkt_corr60", "day_ret", "amt_log", "actcells30", "actcells60",
              "rd_bet_rank", "rd_str_rank", "rd_harm_rank", "rd_pr_rank",
              "relay_str_norm", "relay_free_amt", "harm_deg_ratio", "bet_deg_ratio",
              "rd_rs_gap", "rd_ro_gap", "consensus_sim", "graph_entropy",
              "vol_corr30", "act_frac", "pm_rd_bet", "tail_rd_deg",
              "pm_rd_str_mean", "pm_vs_day_str_change"] {
        v.push(format!("rlg_x_{}", s));
    }
    v
}

pub fn relaynet_names() -> Vec<String> {
    RELAY_NAMES.clone()
}

// ============================================================================
// 主计算
// ============================================================================

pub fn compute_relaynet_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    crate::yupei_dist::compute::with_compute_pool(|| compute_relaynet_inner(date))?
}

fn compute_relaynet_inner(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let started = std::time::Instant::now();
    let mut stage = started;
    eprintln!("[rlg-stage] begin date={date} threads={}", rayon::current_num_threads());
    // ---- 1. 枚举 + 并行读 ----
    let codes = crate::data_paths::list_stock_codes(date, "transaction");
    let day_start_us = day_start_us_of(date);
    let preps: Vec<Option<StockPrep>> = codes
        .par_iter()
        .map(|c| {
            read_trade_fast_inner(c, date, false, true, usize::MAX)
                .ok()
                .map(|recs| {
                    let mut p = prep_stock(&recs, day_start_us);
                    // 盘口快照（无 market_data 文件/解析失败 → 全 0）
                    if let Ok(m) = crate::fast_csv_reader::read_market_fast_inner(
                        c, date, false, true, usize::MAX,
                    ) {
                        fill_market(&mut p, &m, day_start_us);
                    }
                    p
                })
        })
        .collect();
    let n = codes.len();
    if n == 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, "无有效股票"));
    }
    eprintln!("[rlg-stage] read_prepare sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // 行业
    let inds = crate::yupei_dist::industry::industry_of(date, &codes)?
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<i64>>();

    eprintln!("[rlg-stage] industry sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 2. 矩阵 ----
    let mut r30 = Array2::<f64>::zeros((N_CELL, n));
    let mut o30 = Array2::<f64>::zeros((N_CELL, n));
    let mut r60 = Array2::<f64>::zeros((N_CELL60, n));
    let mut o60 = Array2::<f64>::zeros((N_CELL60, n));
    let mut amount = vec![f64::NAN; n];
    let mut nact = vec![0usize; n];
    let mut nact60 = vec![0usize; n];
    for (j, p) in preps.iter().enumerate() {
        if let Some(p) = p {
            for i in 0..N_CELL {
                r30[[i, j]] = p.ret30[i];
                o30[[i, j]] = p.ofi[i];
            }
            for i in 0..N_CELL60 {
                r60[[i, j]] = p.ret60[i];
                o60[[i, j]] = p.ofi60[i];
            }
            amount[j] = p.amount;
            nact[j] = p.n_active;
            nact60[j] = p.n_active60;
        }
    }
    // 活跃过滤
    let live: Vec<bool> = (0..n)
        .map(|j| preps[j].is_some() && nact[j] >= MIN_ACTIVE)
        .collect();

    // 市场均收益 / 锚
    let mkt: Vec<f64> = (0..N_CELL)
        .map(|i| {
            let (mut s, mut c) = (0.0, 0usize);
            for j in 0..n {
                if live[j] {
                    s += r30[[i, j]];
                    c += 1;
                }
            }
            if c > 0 { s / c as f64 } else { 0.0 }
        })
        .collect();

    // ---- 3. 相似度矩阵库（懒缓存：序列→中心化→N×N 矩阵） ----
    let mkt60: Vec<f64> = (0..N_CELL60)
        .map(|i| {
            let (mut s, mut c) = (0.0, 0usize);
            for j in 0..n {
                if live[j] {
                    s += r60[[i, j]];
                    c += 1;
                }
            }
            if c > 0 { s / c as f64 } else { 0.0 }
        })
        .collect();
    let bank = SimBank {
        preps, n,
        r30: r30.clone(), o30: o30.clone(), r60: r60.clone(), o60: o60.clone(),
        mkt: mkt.clone(), mkt60,
        series: RefCell::new((0..41).map(|_| None).collect()),
        stats: RefCell::new(std::collections::HashMap::new()),
        sims: RefCell::new(std::collections::HashMap::new()),
    };
    // x 块要用的序列矩阵
    let v30 = bank.series_mat(4);
    let a30 = bank.series_mat(7);

    // 市场相关（个股收益 vs mkt）
    let mm = mkt.iter().sum::<f64>() / N_CELL as f64;
    let msd = mkt.iter().map(|&x| (x - mm) * (x - mm)).sum::<f64>().sqrt();
    let mut mkt_corr = vec![f64::NAN; n];
    for j in 0..n {
        if !live[j] {
            continue;
        }
        let cj = r30.column(j);
        let mj = cj.sum() / N_CELL as f64;
        let sdj = cj.iter().map(|&x| (x - mj) * (x - mj)).sum::<f64>().sqrt();
        if sdj > 0.0 && msd > 0.0 {
            let mut num = 0.0;
            for i in 0..N_CELL {
                num += (cj[i] - mj) * (mkt[i] - mm);
            }
            mkt_corr[j] = num / (sdj * msd);
        }
    }

    eprintln!("[rlg-stage] matrix_setup sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 4. 主图先行（拿 str 定领导节点） ----
    let g_main = build_thresh_limited(bank.sim("rd", Sess::Day).as_ref(), 0.20, &live);
    let str_main: Vec<f64> = g_main
        .adj
        .iter()
        .map(|a| a.iter().map(|&(_, s)| s as f64).sum())
        .collect();
    let mut ord: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
    ord.sort_by(|&a, &b| str_main[b].total_cmp(&str_main[a]));
    let nlead = (live.iter().filter(|&&x| x).count() / 100).max(5);
    let mut lead_set = vec![false; n];
    for &i in ord.iter().take(nlead) {
        lead_set[i] = true;
    }

    eprintln!("[rlg-stage] main_graph sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 5. 逐图算 ----
    let f_total = RELAY_NAMES.len();
    let mut vals = vec![f64::NAN; n * f_total];
    let mut col = 0usize;
    let mut push_cols = |cols: Vec<Vec<f64>>, expect: usize, names: &mut Vec<String>| {
        debug_assert_eq!(cols.len(), expect);
        for c in cols {
            for i in 0..n {
                vals[i * f_total + col] = c[i];
            }
            col += 1;
        }
        let _ = names;
    };
    let mut _dbg_names = Vec::new();

    for sp in GRAPH_SPECS.iter() {
        let graph_started = std::time::Instant::now();
        let sim_rc = bank.sim(sp.fam, sp.sess);
        let graph_compute_started = std::time::Instant::now();
        let sim: &Array2<f64> = sim_rc.as_ref();
        match sp.kind {
            Kind::DirThr(t) => {
                let dg = build_digraph(sim, t);
                let cols = dir_stats(&dg);
                push_cols(cols, DIR_STATS.len(), &mut _dbg_names);
            }
            _ => {
                let g = match sp.kind {
                    Kind::Thr(t) => build_thresh_limited(sim, t, &live),
                    Kind::KnnU(k) => {
                        let g0 = build_knn(sim, k, false);
                        prune_dead(&g0, &live)
                    }
                    Kind::KnnM(k) => {
                        let g0 = build_knn(sim, k, true);
                        prune_dead(&g0, &live)
                    }
                    Kind::Mst => {
                        let g0 = build_mst(sim);
                        prune_dead(&g0, &live)
                    }
                    Kind::IndThr(t) => {
                        let g0 = build_ind_thresh(sim, &inds, t);
                        prune_dead(&g0, &live)
                    }
                    Kind::DirThr(_) => unreachable!(),
                };
                let mut cols = stat_undir(&g, sim, &inds, &mkt_corr, &lead_set, sp.full);
                if let Kind::Mst = sp.kind {
                    for c in mst_extra(&g, &mkt_corr) {
                        cols.push(c);
                    }
                }
                let expect = if sp.full { STATS_UNDIR.len() } else { REDUCED_STATS.len() }
                    + if let Kind::Mst = sp.kind { MST_EXTRA.len() } else { 0 };
                let (mut ne, mut nl) = (0usize, 0usize);
                for i in 0..n {
                    if live[i] && !g.adj[i].is_empty() {
                        nl += 1;
                        ne += g.adj[i].len();
                    }
                }
                eprintln!("[rlg-graph] {} edges={} nodes={}", sp.tag, ne / 2, nl);
                push_cols(cols, expect, &mut _dbg_names);
            }
        }
        eprintln!("[rlg-time] {} sim_sec={:.6} graph_sec={:.6} total_sec={:.6}", sp.tag, graph_compute_started.duration_since(graph_started).as_secs_f64(), graph_compute_started.elapsed().as_secs_f64(), graph_started.elapsed().as_secs_f64());
    }

    eprintln!("[rlg-stage] graphs sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 6. 行业附加 ----
    {
        let g_ind = prune_dead(
            &build_ind_thresh(bank.sim("rd", Sess::Day).as_ref(), &inds, 0.20), &live);
        let mut cols = Vec::new();
        let (mut is_, mut id_, mut ib_, mut iz_, mut it_) = (
            vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
            vec![f64::NAN; n],
        );
        let bet = betweenness_bin(&g_ind, n_src_of(n));
        let (harm, _, _) = dist_stats(&g_ind, n_src_of(n));
        // 行业内 z-close: 对同行业成员, zscore(1/harm)
        let mut byind: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if inds[i] > 0 && live[i] {
                byind.entry(inds[i]).or_default().push(i);
            }
        }
        let mut top1s = vec![f64::NAN; n];
        for i in 0..n {
            if !g_ind.adj[i].is_empty() {
                top1s[i] = g_ind.adj[i].iter().map(|&(_, s)| s as f64).fold(f64::NAN, |a, b| {
                    if a.is_nan() || b > a { b } else { a }
                });
            }
            id_[i] = g_ind.adj[i].len() as f64;
            is_[i] = g_ind.adj[i].iter().map(|&(_, s)| s as f64).sum();
            ib_[i] = bet[i];
            it_[i] = top1s[i];
        }
        for (_ind, mem) in byind {
            let hs: Vec<f64> = mem.iter().map(|&i| harm[i]).filter(|x| x.is_finite()).collect();
            if hs.len() < 5 {
                continue;
            }
            let m = hs.iter().sum::<f64>() / hs.len() as f64;
            let sd = (hs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / hs.len() as f64).sqrt();
            if sd <= 0.0 {
                continue;
            }
            for &i in &mem {
                if harm[i].is_finite() {
                    iz_[i] = (harm[i] - m) / sd;
                }
            }
        }
        cols.push(is_);
        cols.push(id_);
        cols.push(ib_);
        cols.push(iz_);
        cols.push(it_);
        push_cols(cols, 5, &mut _dbg_names);
    }

    eprintln!("[rlg-stage] industry_extras sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 7. 矩阵直出 ----
    for fam in ["rd", "ra", "rs", "ro", "rd60", "ro60"] {
        let cols = matrix_stats(bank.sim(fam, Sess::Day).as_ref());
        push_cols(cols, MATRIX_STAT_NAMES.len(), &mut _dbg_names);
    }
    // 新相似度矩阵直出
    for fam in ["rv", "rcnt", "ract", "rdpt", "rspr", "rmid", "rlag"] {
        let cols = matrix_stats(bank.sim(fam, Sess::Day).as_ref());
        push_cols(cols, MATRIX_STAT_NAMES.len(), &mut _dbg_names);
    }

    eprintln!("[rlg-stage] matrix_stats sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 8. 上午/下午分图 + 差值 ----
    {
        let r_am = r30.slice(ndarray::s![..PM_SPLIT, ..]).to_owned();
        let r_pm = r30.slice(ndarray::s![PM_SPLIT.., ..]).to_owned();
        let s_am = pearson_mat(&r_am);
        let s_pm = pearson_mat(&r_pm);
        let g_am = prune_dead(&build_thresh_limited(&s_am, 0.20, &live), &live);
        let g_pm = prune_dead(&build_thresh_limited(&s_pm, 0.20, &live), &live);
        let mut am_cols: Vec<Vec<f64>> = Vec::new();
        let mut pm_cols: Vec<Vec<f64>> = Vec::new();
        for (g, dst) in [(&g_am, &mut am_cols), (&g_pm, &mut pm_cols)] {
            let mut deg = vec![f64::NAN; n];
            let mut st = vec![f64::NAN; n];
            let mut t1 = vec![f64::NAN; n];
            let mut csz = vec![f64::NAN; n];
            let mut nind = vec![f64::NAN; n];
            for i in 0..n {
                if !live[i] {
                    continue;
                }
                let a = &g.adj[i];
                deg[i] = a.len() as f64;
                st[i] = a.iter().map(|&(_, s)| s as f64).sum();
                if !a.is_empty() {
                    t1[i] = a.iter().map(|&(_, s)| s as f64).fold(f64::NAN, |x, y| {
                        if x.is_nan() || y > x { y } else { x }
                    });
                }
                if inds[i] > 0 && !a.is_empty() {
                    let same = a.iter().filter(|&&(v, _)| inds[v as usize] == inds[i]).count();
                    nind[i] = same as f64 / a.len() as f64;
                }
            }
            let (cid, sizes) = components(g);
            for i in 0..n {
                csz[i] = if cid[i] == u32::MAX { f64::NAN } else { sizes[cid[i] as usize] as f64 };
            }
            let bet = betweenness_bin(g, n_src_of(n));
            let (harm, _, _) = dist_stats(g, n_src_of(n));
            let pr = pagerank(g, 0.85, 50);
            let kc = kcore(g);
            let dm = dist_to_mkt(g, &mkt_corr);
            dst.push(deg);
            dst.push(st);
            dst.push(bet);
            dst.push(harm);
            dst.push(pr);
            dst.push(kc.iter().map(|&x| x as f64).collect());
            dst.push(dm);
            dst.push(t1);
            dst.push(csz);
            dst.push(nind);
        }
        let delta: Vec<Vec<f64>> = (0..am_cols.len())
            .map(|k| {
                (0..n)
                    .map(|i| {
                        let (a, b) = (pm_cols[k][i], am_cols[k][i]);
                        if a.is_finite() && b.is_finite() { a - b } else { f64::NAN }
                    })
                    .collect()
            })
            .collect();
        push_cols(am_cols, 10, &mut _dbg_names);
        push_cols(pm_cols, 10, &mut _dbg_names);
        push_cols(delta, 10, &mut _dbg_names);
    }

    eprintln!("[rlg-stage] am_pm sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 9. 市场相关 / 跨图汇总 / 组合 ----
    {
        let mut cols: Vec<Vec<f64>> = Vec::new();
        // mkt_corr30 / mkt_corr60
        cols.push(mkt_corr.clone());
        let mm60 = {
            let mv: Vec<f64> = (0..N_CELL60)
                .map(|i| {
                    let (mut s, mut c) = (0.0, 0usize);
                    for j in 0..n {
                        if live[j] {
                            s += r60[[i, j]];
                            c += 1;
                        }
                    }
                    if c > 0 { s / c as f64 } else { 0.0 }
                })
                .collect();
            let m = mv.iter().sum::<f64>() / mv.len() as f64;
            let sd = mv.iter().map(|&x| (x - m) * (x - m)).sum::<f64>().sqrt();
            let mut c = vec![f64::NAN; n];
            for j in 0..n {
                if !live[j] { continue; }
                let cj = r60.column(j);
                let mj = cj.sum() / N_CELL60 as f64;
                let sdj = cj.iter().map(|&x| (x - mj) * (x - mj)).sum::<f64>().sqrt();
                if sdj > 0.0 && sd > 0.0 {
                    let mut num = 0.0;
                    for i in 0..N_CELL60 {
                        num += (cj[i] - mj) * (mv[i] - m);
                    }
                    c[j] = num / (sdj * sd);
                }
            }
            c
        };
        cols.push(mm60);
        cols.push(
            (0..n)
                .map(|i| bank.preps[i].as_ref().map(|p| p.day_ret).unwrap_or(f64::NAN))
                .collect(),
        );
        cols.push(amount.iter().map(|&x| if x.is_finite() && x > 0.0 { x.ln() } else { f64::NAN }).collect());
        cols.push(nact.iter().map(|&x| x as f64).collect());
        cols.push(nact60.iter().map(|&x| x as f64).collect());
        // rank 系列在汇总完所有块后再补 → 先占位
        for _ in 0..4 {
            cols.push(vec![f64::NAN; n]);
        }
        // 组合列
        // relay_str_norm / relay_free_amt / harm_deg_ratio / bet_deg_ratio /
        // rd_rs_gap / rd_ro_gap / consensus_sim / graph_entropy 在主图块里取
        // 这里先占位, 最后用主图列覆盖
        for _ in 0..8 {
            cols.push(vec![f64::NAN; n]);
        }
        // vol_corr30 / act_frac / pm_rd_bet / tail_rd_deg / pm_str_mean / pm_vs_day 占位后回填
        for _ in 0..6 {
            cols.push(vec![f64::NAN; n]);
        }
        push_cols(cols, 24, &mut _dbg_names);
    }

    // 用 vals 里已算的列回填组合因子与排名因子
    // 主图 g30_rd_thr20 的列偏移:
    let off_main = col_offset_of("g30_rd_thr20");
    let str_col = off_main + 1;
    let deg_col = off_main + 0;
    let harm_col = off_main + 9;
    let bet_col = off_main + 12;
    let pr_col = off_main + 14;
    // rl 有向图 netlead 在 rl 块
    let off_rl = col_offset_of("g30_rl_thr10");
    let netlead_col = off_rl + 2;
    // ind bet
    let off_ind = col_offset_of("g30_rd_indT20");
    let ind_bet_col = off_ind + 12;
    let x_base = col_offset_of("__x__");
    // 回填 x 段（col = x_base..x_base+18）
    let mut tmp = vec![0.0f64; n];
    // 排名函数: 非 NaN 值秩 pct
    fn rankpct(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
        idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![f64::NAN; v.len()];
        let m = idx.len();
        if m == 0 {
            return r;
        }
        let mut i = 0;
        while i < m {
            let mut j = i;
            while j + 1 < m && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let p = ((i + 1 + j + 1) as f64 / 2.0) / m as f64;
            for &k in &idx[i..=j] {
                r[k] = p;
            }
            i = j + 1;
        }
        r
    }
    // rd_bet_rank / rd_str_rank / rd_harm_rank / rd_pr_rank
    {
        let get = |c| (0..n).map(|i| vals[i * f_total + c]).collect::<Vec<f64>>();
        let r1 = rankpct(&get(bet_col));
        let r2 = rankpct(&get(str_col));
        let r3 = rankpct(&get(harm_col));
        let r4 = rankpct(&get(pr_col));
        for (k, r) in [r1, r2, r3, r4].iter().enumerate() {
            for i in 0..n {
                vals[i * f_total + x_base + 6 + k] = r[i];
            }
        }
    }
    // 组合
    for i in 0..n {
        let b = vals[i * f_total + bet_col];
        let s = vals[i * f_total + str_col];
        let h = vals[i * f_total + harm_col];
        let d = vals[i * f_total + deg_col];
        let nl = vals[i * f_total + netlead_col];
        let amt = vals[i * f_total + x_base + 3];
        // relay_str_norm = bet / (str+1)
        tmp[i] = if b.is_finite() && s.is_finite() && s > 0.0 { b / s } else { f64::NAN };
        vals[i * f_total + x_base + 10] = tmp[i];
        // relay_free_amt = bet / (amt_log+1)
        vals[i * f_total + x_base + 11] =
            if b.is_finite() && amt.is_finite() { b / (amt + 1.0) } else { f64::NAN };
        // harm_deg_ratio
        vals[i * f_total + x_base + 12] =
            if h.is_finite() && d.is_finite() && d > 0.0 { h / d } else { f64::NAN };
        // bet_deg_ratio
        vals[i * f_total + x_base + 13] =
            if b.is_finite() && d.is_finite() && d > 0.0 { b / d } else { f64::NAN };
        let _ = nl;
    }
    // rd_rs_gap = 平均(rd - rs 标准化差): 用 simmean
    {
        let off_mrd = col_offset_of("mat30_rd");
        let off_mrs = col_offset_of("mat30_rs");
        let off_mro = col_offset_of("mat30_ro");
        let simmean_off = 3;
        for i in 0..n {
            let rd = vals[i * f_total + off_mrd + simmean_off];
            let rs = vals[i * f_total + off_mrs + simmean_off];
            let ro = vals[i * f_total + off_mro + simmean_off];
            vals[i * f_total + x_base + 14] =
                if rd.is_finite() && rs.is_finite() { rd - rs } else { f64::NAN };
            vals[i * f_total + x_base + 15] =
                if rd.is_finite() && ro.is_finite() { rd - ro } else { f64::NAN };
        }
    }
    // consensus_sim = 四个主图 strength 的一致性（rd/ra/rs/ro 在 thr25 的 str 均值）
    {
        let mut sum = vec![0.0f64; n];
        let mut cnt = vec![0usize; n];
        for tag in ["g30_rd_thr25", "g30_ra_thr25", "g30_rs_thr15", "g30_ro_thr25"] {
            let off = col_offset_of(tag) + 1;
            for i in 0..n {
                let v = vals[i * f_total + off];
                if v.is_finite() {
                    sum[i] += v;
                    cnt[i] += 1;
                }
            }
        }
        for i in 0..n {
            vals[i * f_total + x_base + 16] =
                if cnt[i] > 0 { sum[i] / cnt[i] as f64 } else { f64::NAN };
        }
    }
    // graph_entropy = 邻居相似度在主图上的熵（已有 entsim 列直接用其变形?）
    // 用主图 entsim 列
    {
        let off = off_main + 5; // entsim
        for i in 0..n {
            vals[i * f_total + x_base + 17] = vals[i * f_total + off];
        }
    }
    // vol_corr30 = 个股 vol30 与市场总成交量的 pearson
    {
        let mv: Vec<f64> = (0..N_CELL)
            .map(|i| {
                let mut s = 0.0;
                for j in 0..n {
                    if live[j] {
                        s += v30[[i, j]];
                    }
                }
                s
            })
            .collect();
        let mm = mv.iter().sum::<f64>() / N_CELL as f64;
        let msd = mv.iter().map(|&x| (x - mm) * (x - mm)).sum::<f64>().sqrt();
        for i in 0..n {
            if !live[i] {
                continue;
            }
            let cj = v30.column(i);
            let mj = cj.sum() / N_CELL as f64;
            let sdj = cj.iter().map(|&x| (x - mj) * (x - mj)).sum::<f64>().sqrt();
            if sdj > 0.0 && msd > 0.0 {
                let mut num = 0.0;
                for k in 0..N_CELL {
                    num += (cj[k] - mj) * (mv[k] - mm);
                }
                vals[i * f_total + x_base + 18] = num / (sdj * msd);
            }
        }
    }
    // act_frac = 活跃格占比
    for i in 0..n {
        vals[i * f_total + x_base + 19] = nact[i] as f64 / N_CELL as f64;
    }
    // pm_rd_bet = PM 段 rd_thr20 图的 bet（full 统计第 13 列）
    {
        let off = col_offset_of("pm30_rd_thr20") + 12;
        for i in 0..n {
            vals[i * f_total + x_base + 20] = vals[i * f_total + off];
        }
    }
    // tail_rd_deg = 尾盘版 rd_thr20 图的 deg
    {
        let off = col_offset_of("tl30_rd_thr20");
        for i in 0..n {
            vals[i * f_total + x_base + 21] = vals[i * f_total + off];
        }
    }
    // pm_rd_str_mean = rd 族 PM 段 5 张图 str 的均值
    {
        let mut sum = vec![0.0f64; n];
        let mut cnt = vec![0usize; n];
        for tag in ["pm30_rd_thr20", "pm30_rd_thr30", "pm30_rd_knnU30",
                    "pm30_rd_knnM10", "pm30_rd_mst"] {
            let off = col_offset_of(tag) + 1;
            for i in 0..n {
                let v = vals[i * f_total + off];
                if v.is_finite() {
                    sum[i] += v;
                    cnt[i] += 1;
                }
            }
        }
        for i in 0..n {
            vals[i * f_total + x_base + 22] =
                if cnt[i] > 0 { sum[i] / cnt[i] as f64 } else { f64::NAN };
        }
    }
    // pm_vs_day_str_change = PM 与 day 同图(rd_thr20) str 之差
    {
        let off_pm = col_offset_of("pm30_rd_thr20") + 1;
        let off_day = col_offset_of("g30_rd_thr20") + 1;
        for i in 0..n {
            let (a, b) = (vals[i * f_total + off_pm], vals[i * f_total + off_day]);
            vals[i * f_total + x_base + 23] =
                if a.is_finite() && b.is_finite() { a - b } else { f64::NAN };
        }
    }

    eprintln!("[rlg-stage] summary sec={:.6} elapsed={:.6}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    stage = std::time::Instant::now();
    // ---- 转 f32 输出 ----
    assert_eq!(col, f_total, "relaynet 列数与注册表不一致: {col} != {f_total}");
    let out: Vec<f32> = vals.iter().map(|&x| x as f32).collect();
    eprintln!("[rlg-stage] output sec={:.6} total={:.6} stocks={} factors={}", stage.elapsed().as_secs_f64(), started.elapsed().as_secs_f64(), n, f_total);
    Ok((codes, out))
}

/// 把 non-live 节点的所有边剪掉（图已建好后再清，比建图时过滤省事）
fn prune_dead(g: &UGraph, live: &[bool]) -> UGraph {
    let n = g.n;
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        if !live[i] {
            continue;
        }
        for &(v, s) in &g.adj[i] {
            if live[v as usize] {
                adj[i].push((v, s));
            }
        }
    }
    UGraph { n, adj }
}

/// 找某个图标签的输出列偏移（= 它在 GRAPH_SPECS 里前面所有图的因子数之和 + 行业块等顺序）
fn col_offset_of(tag: &str) -> usize {
    let mut off = 0usize;
    for sp in GRAPH_SPECS.iter() {
        if sp.tag == tag {
            return off;
        }
        off += spec_n_factors(sp);
    }
    // 图之后顺序: 行业附加5 + 矩阵直出(6+7)*7 + 上下午20 + delta10 + x24
    let fam_new_mat = ["rv", "rcnt", "ract", "rdpt", "rspr", "rmid", "rlag"];
    match tag {
        "__ind__" => off,
        "mat30_rd" => off + 5,
        "mat30_ra" => off + 5 + 7,
        "mat30_rs" => off + 5 + 14,
        "mat30_ro" => off + 5 + 21,
        "mat60_rd60" => off + 5 + 28,
        "mat60_ro60" => off + 5 + 35,
        "__am__" => off + 5 + 91,
        "__pm__" => off + 5 + 91 + 10,
        "__delta__" => off + 5 + 91 + 20,
        "__x__" => off + 5 + 91 + 30,
        _ => {
            if let Some(f) = tag.strip_prefix("mat30_") {
                if let Some(k) = fam_new_mat.iter().position(|&x| x == f) {
                    return off + 5 + 42 + k * 7;
                }
            }
            usize::MAX
        }
    }
}

fn spec_n_factors(sp: &GraphSpec) -> usize {
    match sp.kind {
        Kind::DirThr(_) => DIR_STATS.len(),
        _ => {
            let base = if sp.full { STATS_UNDIR.len() } else { 9 };
            base + if let Kind::Mst = sp.kind { MST_EXTRA.len() } else { 0 }
        }
    }
}

// ============================================================================
// PyO3
// ============================================================================

#[pyfunction]
pub fn py_relaynet(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_relaynet_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_relaynet_names() -> Vec<String> {
    relaynet_names()
}

/// Harness-only exact comparison of the matrix kernels.
pub fn matrix_probe(input: ndarray::Array2<f64>, kind: u8) -> ndarray::Array2<f64> {
    match kind { 0 => pearson_mat(&input), 1 => sign_mat(&input, MIN_PAIR),
        2 => leadlag_mat(&input), _ => jaccard_mat(&input) }
}

/// Harness-only whole-stat oracle comparison on supplied undirected edges.
pub fn stat_probe(n: usize, edges: Vec<(usize, usize, f32)>) -> Vec<Vec<f64>> {
    let mut g = UGraph { n, adj: vec![Vec::new(); n] };
    for (i, j, s) in edges { g.adj[i].push((j as u32, s)); g.adj[j].push((i as u32, s)); }
    for a in &mut g.adj { a.sort_by_key(|&(j, _)| j); }
    let sim = Array2::<f64>::zeros((n, n));
    let inds: Vec<i64> = (0..n).map(|i| (i % 3 + 1) as i64).collect();
    let mkt: Vec<f64> = (0..n).map(|i| i as f64 / n.max(1) as f64).collect();
    let leaders: Vec<bool> = (0..n).map(|i| i % 7 == 0).collect();
    stat_undir(&g, &sim, &inds, &mkt, &leaders, true)
}

// Sandbox validation: every prep series plus metadata, before graph/matrix work.
pub fn prep_probe(code: &str, date: i64) -> std::io::Result<Vec<f64>> {
    let start = day_start_us_of(date);
    let recs = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    let mut p = prep_stock(&recs, start);
    if let Ok(m) = crate::fast_csv_reader::read_market_fast_inner(code, date, false, true, usize::MAX) {
        fill_market(&mut p, &m, start);
    }
    let mut out = vec![f64::NAN; 41 * N_CELL + 4];
    for k in 0..41usize {
        let s: &[f64] = match k {
            0 => &p.ret30, 1 => &p.ofi, 2 => &p.ret60, 3 => &p.ofi60,
            _ => series_vec(&p, k as u8),
        };
        out[k * N_CELL..k * N_CELL + s.len()].copy_from_slice(s);
    }
    out[41 * N_CELL..].copy_from_slice(&[p.n_active as f64, p.n_active60 as f64, p.amount, p.day_ret]);
    Ok(out)
}
