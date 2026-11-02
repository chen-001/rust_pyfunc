//! 接力网络（relaynet_six_people）横截面因子：同一天全市场股票当节点，
//! 用逐笔对齐到 30 秒格子的价格变动构造多种"共动相似度"，
//! 稀疏化成多张图后在图上跑 Dijkstra/中心性/社区，逐股票输出因子。
//!
//! 数据流（单日）：
//!   1. `list_stock_codes` 枚举 symbol_map ∩ 当日 Level2 文件
//!   2. rayon 读逐笔 → 每股 (last_price[474], ofi[474], active_cnt)
//!   3. 按因子族与时段构造收益、符号、成交、盘口和领先滞后相似度矩阵
//!   4. 按 GRAPH_SPECS 建立 477 张阈值、kNN、最大生成森林、行业内和有向图
//!   5. 完整无向图 51 项、缩减图 9 项、MST 附加 5 项、有向图 12 项
//!      共输出 13,574 列；距离与中介遍历全部源节点
//!      固定源块和加法顺序保证计算确定性
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

/// 30 秒格子: 9:30→11:30(上午240格) 接着下午平移后 11:30→13:27(234格), 共 474
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
    // relaynet exact prep: unused series stay zero
    let mut bo: Vec<(u32, i64)> = Vec::with_capacity(recs.len());
    let mut so: Vec<(u32, i64)> = Vec::with_capacity(recs.len());
    let mut buysell: Vec<(f64, f64)> = vec![(0.0, 0.0); N_CELL];
    let mut ctrades: Vec<Vec<(i64, f64, f64)>> = (0..N_CELL).map(|_| Vec::new()).collect();
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
            bo.push((c30 as u32, r.bid_order));
            so.push((c30 as u32, r.ask_order));
            if sgn > 0.0 {
                buysell[c30].0 += t;
            } else if sgn < 0.0 {
                buysell[c30].1 += t;
            }
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
    let mut big30 = vec![0.0f64; N_CELL];
    let mut act30 = vec![0.0f64; N_CELL];
    for c in 0..N_CELL {
        if tvacc[c] > 0.0 {
            act30[c] = 1.0;
        }
    }
    // 每格不同订单数 / 主导买卖单占比：按 (cell, order) 排序后扫一遍
    let mut buyord30 = vec![0.0f64; N_CELL];
    let mut sellord30 = vec![0.0f64; N_CELL];
    let mut dom30 = vec![0.0f64; N_CELL];
    let mut domsell30 = vec![0.0f64; N_CELL];
    bo.sort_unstable();
    let mut i = 0;
    while i < bo.len() {
        let (c, o) = (bo[i].0, bo[i].1);
        while i < bo.len() && bo[i].0 == c && bo[i].1 == o {
            i += 1;
        }
        buyord30[c as usize] += 1.0;
    }
    so.sort_unstable();
    let mut j = 0;
    while j < so.len() {
        let (c, o) = (so[j].0, so[j].1);
        while j < so.len() && so[j].0 == c && so[j].1 == o {
            j += 1;
        }
        sellord30[c as usize] += 1.0;
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
    let (
        mut vstd30,
        mut vskew30,
        mut vmaxshare30,
        mut vgini30,
        mut amtcv30,
        mut gapcv30,
        mut gapstd30,
        mut fhalf30,
        mut rndsec30,
        mut tspan30,
        mut tunif30,
    ) = (
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
        vec![0.0f64; N_CELL],
    );
    for c in 0..N_CELL {
        let tr = &ctrades[c];
        let m = tr.len();
        if m == 0 {
            continue;
        }
        let vols: Vec<f64> = tr.iter().map(|&(_, v, _)| v).collect();
        let amts: Vec<f64> = tr.iter().map(|&(_, _, t)| t).collect();
        let (mv, ma) = (
            vols.iter().sum::<f64>() / m as f64,
            amts.iter().sum::<f64>() / m as f64,
        );
        vstd30[c] = (vols.iter().map(|&x| (x - mv) * (x - mv)).sum::<f64>() / m as f64).sqrt();
        let sdv = vstd30[c];
        if sdv > 0.0 {
            vskew30[c] = vols.iter().map(|&x| ((x - mv) / sdv).powi(3)).sum::<f64>() / m as f64;
        }
        if tvacc[c] > 0.0 {
            vmaxshare30[c] = amts.iter().cloned().fold(0.0f64, f64::max) / tvacc[c];
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
            let sdg =
                (gaps.iter().map(|&x| (x - mg) * (x - mg)).sum::<f64>() / gaps.len() as f64).sqrt();
            gapstd30[c] = sdg;
            if mg > 0.0 {
                gapcv30[c] = sdg / mg;
            }
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
        last_px,
        ofi,
        n_active,
        n_active60,
        amount,
        day_ret,
        ret30,
        ret60,
        ofi60,
        vol30: volacc.to_vec(),
        tv30: tvacc.to_vec(),
        cnt30: cntacc.to_vec(),
        big30,
        act30,
        buyord30,
        sellord30,
        dom30,
        dimb30: vec![0.0; N_CELL],
        spr30: vec![0.0; N_CELL],
        midr30: vec![0.0; N_CELL],
        rba30: vec![0.0; N_CELL],
        ordbal30,
        actlag30,
        rtn30,
        domsell30,
        imb30,
        vstd30,
        vskew30,
        vmaxshare30,
        vgini30,
        amtcv30,
        gapcv30,
        gapstd30,
        fhalf30,
        rndsec30,
        tspan30,
        tunif30,
        aslope30: vec![0.0; N_CELL],
        bslope30: vec![0.0; N_CELL],
        dskewa30: vec![0.0; N_CELL],
        dskewb30: vec![0.0; N_CELL],
        l1conc30: vec![0.0; N_CELL],
        l1gap30: vec![0.0; N_CELL],
        dsdask30: vec![0.0; N_CELL],
        dsdbid30: vec![0.0; N_CELL],
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
fn fill_market(
    p: &mut StockPrep,
    mrecs: &[crate::fast_csv_reader::MarketRecord],
    day_start_us: i64,
) {
    if mrecs.is_empty() {
        return;
    }
    let mut dnum = [0.0f64; N_CELL];
    let mut snum = [0.0f64; N_CELL];
    let mut dcnt = [0usize; N_CELL];
    let mut scnt = [0usize; N_CELL];
    let mut mid = [f64::NAN; N_CELL];
    // 十档形状的格内和/计数/深度平方和
    let mut l1b = [0.0f64; N_CELL];
    let mut l1a = [0.0f64; N_CELL];
    let mut shcnt = [0usize; N_CELL];
    let (mut ta1, mut ta2, mut tb1, mut tb2, mut tn1, mut tn2, mut tncnt) = (
        [0.0f64; N_CELL],
        [0.0f64; N_CELL],
        [0.0f64; N_CELL],
        [0.0f64; N_CELL],
        [0.0f64; N_CELL],
        [0.0f64; N_CELL],
        [0usize; N_CELL],
    );
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
    let bits = relay_candidate_bits(s, k, Some(allow));
    relay_graph_from_bits(s, &bits, false, Some(t))
}

fn build_knn(s: &Array2<f64>, k: usize, mutual: bool) -> UGraph {
    let bits = relay_candidate_bits(s, k, None);
    relay_graph_from_bits(s, &bits, mutual, None)
}

fn build_mst(s: &Array2<f64>) -> UGraph {
    let n = s.nrows();
    let mut intree = vec![false; n];
    let mut best = vec![f64::NEG_INFINITY; n];
    let mut parent = vec![usize::MAX; n];
    for _ in 0..n {
        let mut u = usize::MAX;
        let mut value = f64::NEG_INFINITY;
        for j in 0..n {
            if !intree[j] && best[j] > value {
                value = best[j];
                u = j;
            }
        }
        // A disconnected finite-similarity component needs a fresh Prim root.
        if u == usize::MAX {
            u = (0..n).find(|&j| !intree[j]).unwrap();
        }
        intree[u] = true;
        for j in 0..n {
            if !intree[j] {
                let v = s[[u, j]];
                if v.is_finite() && v > best[j] {
                    best[j] = v;
                    parent[j] = u;
                }
            }
        }
    }
    let mut adj = vec![Vec::new(); n];
    for j in 0..n {
        if parent[j] != usize::MAX {
            let v = s[[j, parent[j]]] as f32;
            adj[j].push((parent[j] as u32, v));
            adj[parent[j]].push((j as u32, v));
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
        if du != d[u] {
            continue;
        }
        for &(v, sim) in &g.adj[u] {
            let v = v as usize;
            let nd = du + (1.0 - sim as f64).max(1e-9);
            if nd < d[v] {
                d[v] = nd;
                heap.push(State { d: nd, u: v });
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

fn dist_stats(g: &UGraph) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (h, f, e, _, _, _, _) = dist_stats_q(g);
    (h, f, e)
}

/// 全节点精确最短路统计，不抽样或填补其他节点。

/// A forest has one positive-cost path per reachable pair. Its weighted and
/// binary betweenness equal the count of pairs in distinct branches at a node.
fn forest_bet(g: &UGraph) -> Option<Vec<f64>> {
    let n = g.n;
    let mut parent = vec![usize::MAX; n];
    let mut size = vec![1usize; n];
    let mut bet = vec![0.0; n];
    let mut order = Vec::with_capacity(n);
    let mut components = 0usize;
    for root in 0..n {
        if parent[root] != usize::MAX {
            continue;
        }
        components += 1;
        let first = order.len();
        parent[root] = root;
        order.push(root);
        let mut head = first;
        while head < order.len() {
            let u = order[head];
            head += 1;
            for &(v, _) in &g.adj[u] {
                let v = v as usize;
                if parent[v] == usize::MAX {
                    parent[v] = u;
                    order.push(v);
                }
            }
        }
        let count = order.len() - first;
        for &u in order[first..].iter().rev() {
            if u != root {
                size[parent[u]] += size[u];
            }
        }
        for &u in &order[first..] {
            let mut sum = 0u64;
            let mut pairs = 0u64;
            for &(v, _) in &g.adj[u] {
                let v = v as usize;
                let branch = if parent[v] == u {
                    size[v]
                } else {
                    count - size[u]
                } as u64;
                pairs += sum * branch;
                sum += branch;
            }
            bet[u] = pairs as f64;
        }
    }
    let edges = g.adj.iter().map(|a| a.len()).sum::<usize>() / 2;
    if edges == n.saturating_sub(components) {
        Some(bet)
    } else {
        None
    }
}

struct ExactPaths {
    columns: [Vec<f64>; 7],
    bet: Vec<f64>,
}
impl ExactPaths {
    fn distances(
        self,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let [h, f, e, q25, q75, q50, q90] = self.columns;
        (h, f, e, q25, q75, q50, q90)
    }
}

/// Reused for a fixed chunk of sources, so allocation and reduction order do
/// not depend on worker scheduling. No per-node predecessor vectors are needed.
struct PathScratch {
    dist: Vec<f64>,
    sigma: Vec<f64>,
    delta: Vec<f64>,
    order: Vec<usize>,
    heap: BinaryHeap<State>,
    values: Vec<f64>,
    parent: Vec<u32>,
    head: Vec<u32>,
    pred_from: Vec<u32>,
    pred_next: Vec<u32>,
}
impl PathScratch {
    fn new(n: usize) -> Self {
        Self {
            dist: vec![f64::INFINITY; n],
            sigma: vec![0.0; n],
            delta: vec![0.0; n],
            order: Vec::with_capacity(n),
            heap: BinaryHeap::with_capacity(n),
            values: Vec::with_capacity(n),
            parent: vec![u32::MAX; n],
            head: vec![u32::MAX; n],
            pred_from: Vec::with_capacity(n),
            pred_next: Vec::with_capacity(n),
        }
    }
    fn source(
        &mut self,
        g: &UGraph,
        src: usize,
        tree: bool,
        weighted: bool,
        bet: &mut [f64],
    ) -> [f64; 7] {
        self.dist.fill(f64::INFINITY);
        self.dist[src] = 0.0;
        self.order.clear();
        self.values.clear();
        self.heap.clear();
        if weighted && !tree {
            self.sigma.fill(0.0);
            self.sigma[src] = 1.0;
            self.delta.fill(0.0);
            {
                self.parent.fill(u32::MAX);
                self.head.fill(u32::MAX);
                self.pred_from.clear();
                self.pred_next.clear();
            }
        }
        if tree {
            self.order.push(src);
            let mut head = 0;
            while head < self.order.len() {
                let u = self.order[head];
                head += 1;
                let du = self.dist[u];
                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    if !self.dist[v].is_finite() {
                        self.dist[v] = du + (1.0 - sim as f64).max(1e-9);
                        self.order.push(v);
                    }
                }
            }
        } else {
            self.heap.push(State { d: 0.0, u: src });
            while let Some(State { d: du, u }) = self.heap.pop() {
                if du != self.dist[u] {
                    continue;
                }
                if weighted {
                    self.order.push(u);
                }
                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    let nd = du + (1.0 - sim as f64).max(1e-9);
                    if nd < self.dist[v] {
                        self.dist[v] = nd;
                        self.heap.push(State { d: nd, u: v });
                        if weighted {
                            self.sigma[v] = self.sigma[u];
                            {
                                self.parent[v] = u as u32;
                                self.head[v] = u32::MAX;
                            }
                        }
                    } else if weighted && nd == self.dist[v] {
                        self.sigma[v] += self.sigma[u];
                        {
                            self.pred_from.push(u as u32);
                            self.pred_next.push(self.head[v]);
                            self.head[v] = (self.pred_from.len() - 1) as u32;
                        }
                    }
                }
            }
        }
        if weighted && !tree {
            for &u in self.order.iter().rev() {
                {
                    let first = self.parent[u];
                    if first != u32::MAX {
                        let v = first as usize;
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                    }
                    let mut link = self.head[u];
                    while link != u32::MAX {
                        let v = self.pred_from[link as usize] as usize;
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                        link = self.pred_next[link as usize];
                    }
                }
                if u != src {
                    bet[u] += self.delta[u];
                }
            }
        }
        let (mut h, mut f, mut e) = (0.0, 0.0, 0.0f64);
        // Preserve the original node-index summation order exactly.
        for (j, &d) in self.dist.iter().enumerate() {
            if j != src && d.is_finite() && d > 0.0 {
                h += 1.0 / d;
                f += d;
                e = e.max(d);
                self.values.push(d);
            }
        }
        if self.values.is_empty() {
            return [f64::NAN; 7];
        }
        self.values.sort_unstable_by(|a, b| a.total_cmp(b));
        let count = self.values.len() as f64;
        [
            h / count,
            f / count,
            e,
            quantile_sorted(&self.values, 0.25),
            quantile_sorted(&self.values, 0.75),
            quantile_sorted(&self.values, 0.50),
            quantile_sorted(&self.values, 0.90),
        ]
    }
}

fn exact_paths(g: &UGraph, weighted: bool) -> ExactPaths {
    exact_paths_screen4(g, weighted)
}

fn exact_paths_mode(g: &UGraph, weighted: bool) -> ExactPaths {
    let n = g.n;
    let forest = forest_bet(g);
    let tree = forest.is_some();
    // All n sources are used, including isolates (NaN distances, zero bet).
    // Fixed consecutive blocks give deterministic reduction across thread counts.
    const BLOCK: usize = 16;
    let parts: Vec<(Vec<[f64; 7]>, Vec<f64>)> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|block| {
            let lo = block * BLOCK;
            let hi = (lo + BLOCK).min(n);
            let mut scratch = PathScratch::new(n);
            let mut bet = vec![0.0; n];
            let rows = (lo..hi)
                .map(|src| scratch.source(g, src, tree, weighted, &mut bet))
                .collect();
            (rows, bet)
        })
        .collect();
    let mut columns: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::with_capacity(n));
    let mut bet = forest.unwrap_or_else(|| vec![0.0; n]);
    for (rows, source_bet) in parts {
        for row in rows {
            for j in 0..7 {
                columns[j].push(row[j]);
            }
        }
        if weighted && !tree {
            for (a, b) in bet.iter_mut().zip(source_bet) {
                *a += b;
            }
        }
    }
    if weighted && !tree {
        for b in &mut bet {
            *b *= 0.5;
        }
    }
    ExactPaths { columns, bet }
}

fn dist_stats_q(
    g: &UGraph,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    exact_paths(g, false).distances()
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

fn betweenness_bin(g: &UGraph) -> Vec<f64> {
    binary_direction_run(g)
}

fn pagerank(g: &UGraph, d: f64, iters: usize) -> Vec<f64> {
    let n = g.n;
    let strength: Vec<f64> = g
        .adj
        .iter()
        .map(|a| a.iter().map(|&(_, s)| s as f64).sum())
        .collect();
    // The corrected push algorithm adds contributions in source-index order.
    // A sorted incoming row gives that same floating-point order independently per target.
    let incoming: Vec<Vec<(u32, f32)>> = g
        .adj
        .iter()
        .map(|a| {
            let mut row = a.clone();
            row.sort_unstable_by_key(|&(v, _)| v);
            row
        })
        .collect();
    let mut pr = vec![1.0 / n as f64; n];
    let mut next = vec![0.0; n];
    for _ in 0..iters {
        let mut dangling = 0.0;
        for i in 0..n {
            if strength[i] <= 0.0 {
                dangling += pr[i];
            }
        }
        next.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut value = (1.0 - d) / n as f64;
            for &(src, weight) in &incoming[i] {
                let src = src as usize;
                if strength[src] <= 0.0 {
                    continue;
                }
                value += d * pr[src] * weight as f64 / strength[src];
            }
            *out = value + d * dangling / n as f64;
        });
        std::mem::swap(&mut pr, &mut next);
    }
    pr
}

fn eigenvec(g: &UGraph, iters: usize) -> Vec<f64> {
    let n = g.n;
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut y = vec![0.0; n];
    for _ in 0..iters {
        y.par_iter_mut().enumerate().for_each(|(i, out)| {
            *out = g.adj[i]
                .iter()
                .map(|&(v, s)| s as f64 * x[v as usize])
                .sum();
        });
        // Keep the normalization sum in its original index order.
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 0.0 {
            break;
        }
        x.par_iter_mut()
            .zip(y.par_iter())
            .for_each(|(out, &value)| *out = value / norm);
    }
    x
}

fn kcore(g: &UGraph) -> Vec<u32> {
    // Degree buckets: every edge is visited once, with no repeated market-wide scan.
    let n = g.n;
    let mut degree: Vec<usize> = g.adj.iter().map(Vec::len).collect();
    let max_degree = degree.iter().copied().max().unwrap_or(0);
    let mut bins = vec![0usize; max_degree + 1];
    for &d in &degree {
        bins[d] += 1;
    }
    let mut start = 0;
    for count in &mut bins {
        let old = *count;
        *count = start;
        start += old;
    }
    let mut positions = vec![0usize; n];
    let mut vertices = vec![0usize; n];
    for v in 0..n {
        positions[v] = bins[degree[v]];
        vertices[positions[v]] = v;
        bins[degree[v]] += 1;
    }
    for d in (1..=max_degree).rev() {
        bins[d] = bins[d - 1];
    }
    bins[0] = 0;
    for i in 0..n {
        let v = vertices[i];
        for &(u, _) in &g.adj[v] {
            let u = u as usize;
            if degree[u] > degree[v] {
                let d = degree[u];
                let pu = positions[u];
                let pw = bins[d];
                let w = vertices[pw];
                vertices.swap(pu, pw);
                positions[u] = pw;
                positions[w] = pu;
                bins[d] += 1;
                degree[u] -= 1;
            }
        }
    }
    degree.into_iter().map(|v| v as u32).collect()
}

fn communities(g: &UGraph) -> Vec<u32> {
    let n = g.n;
    let mut lab: Vec<u32> = (0..n as u32).collect();
    let mut weights = vec![0.0f64; n];
    let mut marks = vec![false; n];
    let mut touched = Vec::with_capacity(n);
    for _ in 0..100 {
        let mut changed = false;
        for i in 0..n {
            if g.adj[i].is_empty() {
                continue;
            }
            touched.clear();
            for &(v, s) in &g.adj[i] {
                let l = lab[v as usize] as usize;
                if !marks[l] {
                    weights[l] = 0.0;
                    marks[l] = true;
                    touched.push(l);
                }
                weights[l] += s as f64;
            }
            let mut best = touched[0];
            for &l in &touched {
                let cmp = weights[l]
                    .partial_cmp(&weights[best])
                    .unwrap_or(Ordering::Equal);
                if cmp == Ordering::Greater || (cmp == Ordering::Equal && l < best) {
                    best = l;
                }
                marks[l] = false;
            }
            if lab[i] != best as u32 {
                lab[i] = best as u32;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    lab
}

// ---------------------------------------------------------------------------
// 每图统计打包
// ---------------------------------------------------------------------------

const STATS_UNDIR: [&str; 51] = [
    "deg",
    "str",
    "top1",
    "top3",
    "meansim",
    "entsim",
    "kcore",
    "reach2",
    "compsize",
    "harm",
    "farness",
    "ecc",
    "bet",
    "betw",
    "pagerank",
    "eigv",
    "clusw",
    "comm_size",
    "comm_inshare",
    "dist_mkt",
    "dist_lead",
    "neighind",
    "ndeg",
    "hhi",
    "jacc",
    "leafnb",
    "d25",
    "d75",
    "assort",
    // 抗极端值扩展层（22 个，全部用分位数/top-k 均值/比值，不用单点极值）
    "str_p25",
    "str_p50",
    "str_p90",
    "str_t10m",
    "str_cv",
    "ndeg_p50",
    "reach3",
    "pdl50",
    "dist_p90",
    "dist_iqr",
    "bet_t10",
    "pr_log",
    "tri_p75",
    "corefrac",
    "comm_top_sim",
    "comm_edges_out",
    "comm_deg_in",
    "dist_mkt_top10",
    "lead_bet_mean",
    "local_density",
    "deg_rank",
    "mstreach",
];
const REDUCED_STATS: [&str; 9] = [
    "deg", "str", "top1", "top3", "meansim", "entsim", "compsize", "dist_mkt", "neighind",
];
const MST_EXTRA: [&str; 5] = ["submass", "is_art", "treedepth", "hops_mkt", "leaf"];
const DIR_STATS: [&str; 12] = [
    "outstr",
    "instr",
    "netlead",
    "outdeg",
    "indeg",
    "leadshare",
    "netpr",
    "prlead",
    "outharm",
    "inharm",
    "outreach",
    "inreach",
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

    {
        cols.extend(undir_basic_parallel_exact(g));
    }

    if !full {
        // 缩减包：只出前 6 个 + compsize + dist_mkt + neighind
        let (cid, sizes) = components(g);
        cols.push(
            cid.iter()
                .map(|&c| {
                    if c == u32::MAX {
                        f64::NAN
                    } else {
                        sizes[c as usize] as f64
                    }
                })
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

    // Exact reusable neighborhood bitsets. No paths or nodes are sampled.
    let nw = (n + 63) / 64;
    let mut nbit = vec![vec![0u64; nw]; n];
    for i in 0..n {
        for &(v, _) in &g.adj[i] {
            nbit[i][v as usize / 64] |= 1u64 << (v as usize % 64);
        }
    }
    let balls: Vec<Vec<u64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut ball = nbit[i].clone();
            for &(v, _) in &g.adj[i] {
                for (b, &w) in ball.iter_mut().zip(&nbit[v as usize]) {
                    *b |= w;
                }
            }
            ball
        })
        .collect();
    cols.push(
        (0..n)
            .map(|i| {
                if g.adj[i].is_empty() {
                    0.0
                } else {
                    balls[i]
                        .iter()
                        .map(|w| w.count_ones() as usize)
                        .sum::<usize>() as f64
                        - 1.0
                }
            })
            .collect(),
    );

    let (cid, sizes) = components(g);
    let lcc_id = sizes
        .iter()
        .enumerate()
        .max_by_key(|&(_, &z)| z)
        .map(|(i, _)| i as u32);
    cols.push(
        cid.iter()
            .map(|&c| {
                if c == u32::MAX {
                    f64::NAN
                } else {
                    sizes[c as usize] as f64
                }
            })
            .collect(),
    );

    let exact = exact_paths(g, true);
    let [harm, farness, ecc, d25, d75, pdl50, dist_p90] = exact.columns;
    cols.push(harm);
    cols.push(farness);
    cols.push(ecc);

    cols.push(betweenness_bin(g));
    cols.push(exact.bet);
    cols.push(pagerank(g, 0.85, 50));
    cols.push(eigenvec(g, 50));
    let (clustering, triangle_p75_cache) = { clustering_triangle_pair_exact(g) };
    cols.push(clustering);

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
    // A two-hop ball covering its component has exactly the component's edges.
    let mut component_edges = vec![0usize; sizes.len()];
    for i in 0..n {
        component_edges[cid[i] as usize] += g.adj[i].len();
    }
    for e in &mut component_edges {
        *e /= 2;
    }
    let degu: Vec<f64> = g.adj.iter().map(|a| a.len() as f64).collect();
    // 每个节点的邻居均度（assort 外延与 ndeg 列共用）
    let ndeg_arr: Vec<f64> = g
        .adj
        .iter()
        .map(|a| {
            if a.is_empty() {
                f64::NAN
            } else {
                a.iter().map(|&(v, _)| degu[v as usize]).sum::<f64>() / a.len() as f64
            }
        })
        .collect();
    let str_v: Vec<f64> = cols[1].clone();
    // ---- 抗极端值扩展层需要的共享中间量 ----
    let kcr: Vec<u32> = cols[6].iter().map(|&x| x as u32).collect();
    let mut kcr_sorted: Vec<u32> = kcr.clone();
    kcr_sorted.sort_unstable();
    let kc_med = if kcr_sorted.is_empty() {
        0.0
    } else {
        kcr_sorted[kcr_sorted.len() / 2] as f64
    };
    let betw_arr: Vec<f64> = cols[13].clone(); // betw（已有列）
    let pr_arr: Vec<f64> = cols[14].clone(); // pagerank
    let dm_arr: Vec<f64> = cols[19].clone(); // dist_mkt
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
            o[10] = sw[m - kt..].iter().sum::<f64>() / kt as f64; // str_t10m
            let mw = str_v[i] / di;
            let sd = (sw.iter().map(|&x| (x - mw) * (x - mw)).sum::<f64>() / di).sqrt();
            o[11] = if mw.abs() > 1e-12 {
                sd / mw.abs()
            } else {
                f64::NAN
            }; // str_cv
               // ---- 邻居结构 ----
            o[0] = a.iter().map(|&(v, _)| degu[v as usize]).sum::<f64>() / di; // ndeg
            let mut nb_deg: Vec<f64> = a.iter().map(|&(v, _)| degu[v as usize]).collect();
            nb_deg.sort_by(|x, y| x.total_cmp(y));
            let dp = nb_deg.len();
            o[12] = nb_deg[dp / 2]; // ndeg_p50
                                    // Exact bitset union/intersection replaces repeated edge scans.
            {
                let ball = &balls[i];
                let ball_n: usize = ball.iter().map(|w| w.count_ones() as usize).sum();
                let component = cid[i] as usize;
                let (cnt3, e_in) = if ball_n == sizes[component] as usize {
                    (ball_n, component_edges[component])
                } else {
                    let mut b3 = ball.clone();
                    let mut twice_edges = 0usize;
                    for (wi, &word) in ball.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let j = wi * 64 + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            for w in 0..nw {
                                b3[w] |= nbit[j][w];
                                twice_edges += (ball[w] & nbit[j][w]).count_ones() as usize;
                            }
                        }
                    }
                    (
                        b3.iter().map(|w| w.count_ones() as usize).sum(),
                        twice_edges / 2,
                    )
                };
                o[13] = (cnt3 as i64
                    - 1
                    - if a.iter().any(|&(v, _)| v as usize == i) {
                        1
                    } else {
                        0
                    }) as f64;
                if ball_n >= 2 {
                    o[26] = e_in as f64 / (ball_n * (ball_n - 1) / 2) as f64;
                }
            }
            // ---- hhi / jacc / leafnb / assort（原有 4 个） ----
            let swsum = str_v[i];
            if swsum > 0.0 {
                o[1] = a
                    .iter()
                    .map(|&(_, x)| {
                        let q = x as f64 / swsum;
                        q * q
                    })
                    .sum::<f64>();
            }
            let mut js = 0.0;
            let mut leaf = 0.0;
            for &(v, _) in a {
                let vi = v as usize;
                let mut inter = 0usize;
                for w in 0..nw {
                    inter += (nbit[i][w] & nbit[vi][w]).count_ones() as usize;
                }
                let uni = a.len() + g.adj[vi].len() - inter;
                if uni > 0 {
                    js += inter as f64 / uni as f64;
                }
                if g.adj[vi].len() == 1 {
                    leaf += 1.0;
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
                        xa.push(dv as f64);
                        ya.push(yv);
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
                    if vx > 0.0 && vy > 0.0 {
                        o[6] = num / (vx * vy).sqrt();
                    }
                }
            }
            // tri_p75：节点参与三角形强度（(sij·sjk·ski)^(1/3)）的 75 分位
            {
                o[19] = triangle_p75_cache[i];
            }
            // corefrac：邻居中 kcore>=全局中位数的占比
            o[20] = a
                .iter()
                .filter(|&&(v, _)| kcr[v as usize] as f64 >= kc_med)
                .count() as f64
                / di;
            // ---- 中心性稳健版 ----
            o[14] = pdl50[i];
            o[15] = dist_p90[i];
            o[4] = d25[i];
            o[5] = d75[i];
            if d25[i].is_finite() && d75[i].is_finite() {
                o[16] = d75[i] - d25[i];
            }
            // bet_t10：邻居 betw 最强 10% 的均值
            {
                let mut bw: Vec<f64> = a
                    .iter()
                    .map(|&(v, _)| betw_arr[v as usize])
                    .filter(|x| x.is_finite())
                    .collect();
                if !bw.is_empty() {
                    bw.sort_by(|x, y| y.total_cmp(x));
                    let kb = ((bw.len() as f64 * 0.10).ceil() as usize).max(1);
                    o[17] = bw[..kb].iter().sum::<f64>() / kb as f64;
                }
            }
            if pr_arr[i].is_finite() && pr_arr[i] > 0.0 {
                o[18] = pr_arr[i].ln();
            }
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
                if s_all > 0.0 {
                    o[22] = 1.0 - s_in / s_all;
                } // comm_edges_out
                o[23] = c_in as f64 / di; // comm_deg_in
                if !w_in.is_empty() {
                    w_in.sort_by(|x, y| y.total_cmp(x));
                    let kb = (w_in.len() / 2).max(1).min(3);
                    o[21] = w_in[..kb].iter().sum::<f64>() / kb as f64; // comm_top_sim
                }
            }
            // ---- 锚点/市场 ----
            {
                // dist_mkt_top10：邻居"经它到市场锚点"距离(dm[v]+w)最小的 10% 均值
                let mut dv: Vec<f64> = a
                    .iter()
                    .map(|&(v, w)| dm_arr[v as usize] + (1.0 - w as f64).max(1e-9))
                    .filter(|x| x.is_finite())
                    .collect();
                if !dv.is_empty() {
                    dv.sort_by(|x, y| x.total_cmp(y));
                    let kb = ((dv.len() as f64 * 0.10).ceil() as usize).max(1);
                    o[24] = dv[..kb].iter().sum::<f64>() / kb as f64;
                }
                // lead_bet_mean：邻居里属于 lead_set 的 betw 均值
                let bs: Vec<f64> = a
                    .iter()
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
        cols.push(
            newstats
                .iter()
                .map(|t| match k {
                    0 => t[0],
                    1 => t[1],
                    2 => t[2],
                    3 => t[3],
                    4 => t[4],
                    5 => t[5],
                    _ => t[6],
                })
                .collect(),
        );
    }
    // 新 21 个（7..27 共 21 个）
    for k in 7..28 {
        cols.push(newstats.iter().map(|t| t[k]).collect());
    }
    // mstreach：三步球占全市场比例（补充 reach2 绝对数）
    let n_f = n as f64;
    cols.push(
        newstats
            .iter()
            .map(|t| {
                if t[13].is_finite() {
                    t[13] / n_f
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );
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
        if du != d[u] {
            continue;
        }
        for &(v, sim) in &g.adj[u] {
            let w = (1.0 - sim as f64).max(1e-9);
            if du + w < d[v as usize] {
                d[v as usize] = du + w;
                heap.push(State {
                    d: du + w,
                    u: v as usize,
                });
            }
        }
    }
    d.iter()
        .map(|&x| if x.is_finite() { x } else { f64::NAN })
        .collect()
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
        if du != d[u] {
            continue;
        }
        for &(v, sim) in &g.adj[u] {
            let w = (1.0 - sim as f64).max(1e-9);
            if du + w < d[v as usize] {
                d[v as usize] = du + w;
                heap.push(State {
                    d: du + w,
                    u: v as usize,
                });
            }
        }
    }
    d.iter()
        .map(|&x| if x.is_finite() { x } else { f64::NAN })
        .collect()
}

// ---------------------------------------------------------------------------
// MST 附加统计
// ---------------------------------------------------------------------------

fn mst_extra(g: &UGraph, mkt_corr: &[f64], precomputed_harm: Option<&[f64]>) -> Vec<Vec<f64>> {
    let n = g.n;
    // treedepth: 找一个中心根 = 调和中心最大的点，BFS 深度
    let owned_harm;
    let harm = if let Some(h) = precomputed_harm {
        h
    } else {
        owned_harm = dist_stats(g).0;
        &owned_harm
    };
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
    // is_art 保留既有分叉节点定义（度 >= 3），不是严格割点判定。
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
            hops[i] = if dist[i] >= 0 {
                dist[i] as f64
            } else {
                f64::NAN
            };
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

    let (pr, pr2) = dir_pagerank_pair_exact(g, &outstr, &instr);

    let netpr: Vec<f64> = pr.iter().zip(&pr2).map(|(a, b)| a - b).collect();
    let prlead: Vec<f64> = pr.iter().zip(&pr2).map(|(a, b)| a / b.max(1e-12)).collect();

    // 出/入调和距离：从每个节点沿对应方向求全部加权最短路。

    let (outh, inh) = dir_harmonic_pair_exact(g);

    // outreach/inreach: 出/入 k 跳可达计数（BFS 到 2 跳）

    let outreach = dir_two_hop_exact(&g.out_adj);
    let inreach = dir_two_hop_exact(&g.in_adj);

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
                if du != d[u] {
                    continue;
                }
                for &(v, sim) in &adj[u] {
                    let w = (1.0 - sim as f64).max(1e-9);
                    if du + w < d[v as usize] {
                        d[v as usize] = du + w;
                        heap.push(State {
                            d: du + w,
                            u: v as usize,
                        });
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
            if cnt == 0 {
                f64::NAN
            } else {
                h / cnt as f64
            }
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

/// 用预先中心化的列算皮尔逊。SIMD 的不同通道对应不同股票对，
/// 每个通道仍按原来的时间顺序逐项相加，不改变浮点归约顺序、不使用 FMA。
fn pearson_from(cols: &[Vec<f64>], sd: &[f64]) -> Array2<f64> {
    let n = cols.len();
    let t = cols.first().map_or(0, Vec::len);
    const LANES: usize = 8;
    let blocks = (n + LANES - 1) / LANES;
    let mut packed = vec![0.0; blocks * t * LANES];
    for j in 0..n {
        let base = (j / LANES) * t * LANES + j % LANES;
        for k in 0..t {
            packed[base + k * LANES] = cols[j][k];
        }
    }
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n];
            if sd[i] > 0.0 {
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                    unsafe {
                        pearson_row_avx2(&cols[i], &packed, sd, i, &mut row);
                    }
                    return row;
                }
                for j in i + 1..n {
                    if sd[j] <= 0.0 {
                        continue;
                    }
                    let mut num = 0.0;
                    for k in 0..t {
                        num += cols[i][k] * cols[j][k];
                    }
                    row[j] = num / (sd[i] * sd[j]);
                }
            }
            row
        })
        .collect();
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    for (i, row) in rows.into_iter().enumerate() {
        for j in i + 1..n {
            out[[i, j]] = row[j];
            out[[j, i]] = row[j];
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pearson_row_avx2(ci: &[f64], packed: &[f64], sd: &[f64], i: usize, row: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = sd.len();
    let t = ci.len();
    for block in (i + 1) / 8..(n + 7) / 8 {
        let p = packed.as_ptr().add(block * t * 8);
        let mut a = _mm256_setzero_pd();
        let mut b = _mm256_setzero_pd();
        for k in 0..t {
            let x = _mm256_set1_pd(*ci.get_unchecked(k));
            let pa = _mm256_mul_pd(x, _mm256_loadu_pd(p.add(k * 8)));
            let pb = _mm256_mul_pd(x, _mm256_loadu_pd(p.add(k * 8 + 4)));
            a = _mm256_add_pd(a, pa);
            b = _mm256_add_pd(b, pb);
        }
        let mut nums = [0.0f64; 8];
        _mm256_storeu_pd(nums.as_mut_ptr(), a);
        _mm256_storeu_pd(nums.as_mut_ptr().add(4), b);
        for lane in 0..8 {
            let j = block * 8 + lane;
            if j > i && j < n && !(sd[j] <= 0.0) {
                row[j] = nums[lane] / (sd[i] * sd[j]);
            }
        }
    }
}

/// 列标准化皮尔逊: rets (T,N), nan=0, sd<=0 列全 NaN
fn pearson_mat(rets: &Array2<f64>) -> Array2<f64> {
    let (cols, sd) = centered_cols(rets);
    pearson_from(&cols, &sd)
}

/// 符号一致率：两个位图分别保留非零和正数判定，连 NaN 的旧判定也一致。
fn sign_mat(rets: &Array2<f64>, min_pair: usize) -> Array2<f64> {
    let (t, n) = (rets.nrows(), rets.ncols());
    let nw = (t + 63) / 64;
    let mut nonzero = vec![0u64; n * nw];
    let mut positive = vec![0u64; n * nw];
    for k in 0..t {
        let mask = 1u64 << (k % 64);
        for j in 0..n {
            let v = rets[[k, j]];
            if v != 0.0 {
                nonzero[j * nw + k / 64] |= mask;
            }
            if v > 0.0 {
                positive[j * nw + k / 64] |= mask;
            }
        }
    }
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n];
            for j in i + 1..n {
                let (mut tot, mut different) = (0usize, 0usize);
                for w in 0..nw {
                    let common = nonzero[i * nw + w] & nonzero[j * nw + w];
                    tot += common.count_ones() as usize;
                    different += (common & (positive[i * nw + w] ^ positive[j * nw + w]))
                        .count_ones() as usize;
                }
                if tot >= min_pair {
                    row[j] = (tot - different) as f64 / tot as f64;
                }
            }
            row
        })
        .collect();
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    for (i, row) in rows.into_iter().enumerate() {
        for j in i + 1..n {
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

/// 领先-滞后反对称矩阵。仅算上三角：交换股票后前向和后向严格互换。
/// 精确相消保留 +0；非零分子除法下溢到零时仍保留正确的正负号。
fn leadlag_mat(rets: &Array2<f64>) -> Array2<f64> {
    let (t, n) = (rets.nrows(), rets.ncols());
    let (cols, sd) = centered_cols(rets);
    let rows: Vec<(Vec<f64>, Vec<usize>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n];
            let mut exact_zeros = Vec::new();
            if sd[i] <= 0.0 {
                return (row, exact_zeros);
            }
            let ci = &cols[i];
            for j in i + 1..n {
                if sd[j] <= 0.0 {
                    continue;
                }
                let cj = &cols[j];
                let (mut fwd, mut bwd) = (0.0, 0.0);
                for k in 0..t.saturating_sub(1) {
                    fwd += ci[k] * cj[k + 1];
                }
                for k in 1..t {
                    bwd += ci[k] * cj[k - 1];
                }
                let numerator = fwd - bwd;
                row[j] = numerator / (sd[i] * sd[j]);
                if numerator == 0.0 && row[j] == 0.0 {
                    exact_zeros.push(j);
                }
            }
            (row, exact_zeros)
        })
        .collect();
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    for (i, (row, exact_zeros)) in rows.into_iter().enumerate() {
        for j in i + 1..n {
            out[[i, j]] = row[j];
            out[[j, i]] = -row[j];
        }
        for j in exact_zeros {
            out[[j, i]] = 0.0;
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
fn stack_series(
    preps: &[Option<StockPrep>],
    n: usize,
    f: impl Fn(&StockPrep) -> &Vec<f64>,
) -> Array2<f64> {
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
        4 => &p.vol30,
        5 => &p.tv30,
        6 => &p.cnt30,
        7 => &p.act30,
        8 => &p.dom30,
        9 => &p.buyord30,
        10 => &p.sellord30,
        11 => &p.dimb30,
        12 => &p.spr30,
        13 => &p.midr30,
        14 => &p.ordbal30,
        16 => &p.actlag30,
        17 => &p.rtn30,
        18 => &p.domsell30,
        19 => &p.imb30,
        20 => &p.vstd30,
        21 => &p.vskew30,
        22 => &p.vmaxshare30,
        23 => &p.vgini30,
        24 => &p.amtcv30,
        25 => &p.gapcv30,
        26 => &p.gapstd30,
        27 => &p.fhalf30,
        28 => &p.rndsec30,
        29 => &p.tspan30,
        30 => &p.tunif30,
        31 => &p.aslope30,
        32 => &p.bslope30,
        33 => &p.dskewa30,
        34 => &p.dskewb30,
        35 => &p.l1conc30,
        36 => &p.l1gap30,
        37 => &p.dsdask30,
        38 => &p.dsdbid30,
        39 => &p.dsdnet30,
        40 => &p.big30,
        _ => &p.rba30,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SimKind {
    P,
    PJ,
    S,
    T,
    L,
    J,
}
struct FamSrc {
    series: u8,
    kind: SimKind,
}

/// 相似度族 → (序列id, 算法)
fn fam_src(fam: &str) -> FamSrc {
    match fam {
        "rd" => FamSrc {
            series: 0,
            kind: SimKind::P,
        },
        "ra" => FamSrc {
            series: 0,
            kind: SimKind::PJ,
        },
        "rs" => FamSrc {
            series: 0,
            kind: SimKind::S,
        },
        "ro" => FamSrc {
            series: 1,
            kind: SimKind::P,
        },
        "rl" => FamSrc {
            series: 0,
            kind: SimKind::L,
        },
        "rd60" => FamSrc {
            series: 2,
            kind: SimKind::P,
        },
        "ra60" => FamSrc {
            series: 2,
            kind: SimKind::PJ,
        },
        "ro60" => FamSrc {
            series: 3,
            kind: SimKind::P,
        },
        "rv" => FamSrc {
            series: 4,
            kind: SimKind::P,
        },
        "rcnt" => FamSrc {
            series: 6,
            kind: SimKind::P,
        },
        "ract" => FamSrc {
            series: 7,
            kind: SimKind::J,
        },
        "rdpt" => FamSrc {
            series: 11,
            kind: SimKind::P,
        },
        "rspr" => FamSrc {
            series: 12,
            kind: SimKind::P,
        },
        "rmid" => FamSrc {
            series: 13,
            kind: SimKind::P,
        },
        "rbal" => FamSrc {
            series: 14,
            kind: SimKind::P,
        },
        "rlag" => FamSrc {
            series: 16,
            kind: SimKind::P,
        },
        "rrtn" => FamSrc {
            series: 17,
            kind: SimKind::P,
        },
        "rvstd" => FamSrc {
            series: 20,
            kind: SimKind::P,
        },
        "rvskew" => FamSrc {
            series: 21,
            kind: SimKind::P,
        },
        "rvmax" => FamSrc {
            series: 22,
            kind: SimKind::P,
        },
        "ramtcv" => FamSrc {
            series: 24,
            kind: SimKind::P,
        },
        "rgapcv" => FamSrc {
            series: 25,
            kind: SimKind::P,
        },
        "rgapstd" => FamSrc {
            series: 26,
            kind: SimKind::P,
        },
        "rfhalf" => FamSrc {
            series: 27,
            kind: SimKind::P,
        },
        "rrnd" => FamSrc {
            series: 28,
            kind: SimKind::P,
        },
        "rtunif" => FamSrc {
            series: 30,
            kind: SimKind::P,
        },
        "rimb" => FamSrc {
            series: 19,
            kind: SimKind::P,
        },
        "rl1c" => FamSrc {
            series: 35,
            kind: SimKind::P,
        },
        "rl1g" => FamSrc {
            series: 36,
            kind: SimKind::P,
        },
        "rdsda" => FamSrc {
            series: 37,
            kind: SimKind::P,
        },
        "rdsdb" => FamSrc {
            series: 38,
            kind: SimKind::P,
        },
        "rdsdn" => FamSrc {
            series: 39,
            kind: SimKind::P,
        },
        _ => FamSrc {
            series: 0,
            kind: SimKind::P,
        },
    }
}

/// 相似度矩阵库：序列 (T,N) + 中心化统计 + 成品种矩阵，三层懒缓存
struct SimBank {
    preps: Vec<Option<StockPrep>>,
    n: usize,
    r30: Array2<f64>,
    o30: Array2<f64>,
    r60: Array2<f64>,
    o60: Array2<f64>,
    mkt: Vec<f64>,
    mkt60: Vec<f64>,
    series: RefCell<Vec<Option<Rc<Array2<f64>>>>>,
    stats: RefCell<std::collections::HashMap<(u8, u8), Rc<(Vec<Vec<f64>>, Vec<f64>)>>>,
    sims: RefCell<std::collections::HashMap<(u8, u8, SimKind), Rc<Array2<f64>>>>,
}

fn sess_u8(s: Sess) -> u8 {
    match s {
        Sess::Day => 0,
        Sess::Pm => 1,
        Sess::Tl => 2,
    }
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
        let key = (fs.series, sess_u8(sess), fs.kind);
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
                for v in a.iter_mut() {
                    *v = v.abs();
                }
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
    let rows: Vec<[f64; 7]> = (0..n)
        .into_par_iter()
        .map_init(
            || Vec::<f64>::with_capacity(n),
            |v, i| {
                v.clear();
                v.extend(s.row(i).iter().enumerate().filter_map(|(j, &x)| {
                    if j != i && x.is_finite() {
                        Some(x)
                    } else {
                        None
                    }
                }));
                if v.is_empty() {
                    return [f64::NAN; 7];
                }
                v.sort_by(|a, b| b.total_cmp(a));
                let t3 = v.len().min(3);
                let t10 = v.len().min(10);
                [
                    v[0],
                    v[..t3].iter().sum::<f64>() / t3 as f64,
                    v[..t10].iter().sum::<f64>() / t10 as f64,
                    v.iter().sum::<f64>() / v.len() as f64,
                    v[(v.len() as f64 * 0.1) as usize],
                    v.iter().filter(|&&x| x > 0.2).count() as f64 / v.len() as f64,
                    v.iter().filter(|&&x| x > 0.3).count() as f64 / v.len() as f64,
                ]
            },
        )
        .collect();
    (0..7)
        .map(|k| rows.iter().map(|r| r[k]).collect())
        .collect()
}

const MATRIX_STAT_NAMES: [&str; 7] = [
    "simmax",
    "simtop3",
    "simtop10",
    "simmean",
    "simp90",
    "simfrac02",
    "simfrac03",
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
enum Sess {
    Day,
    Pm,
    Tl,
}
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
    GraphSpec {
        tag: tag.to_string(),
        fam,
        kind,
        full,
        sess,
    }
}

struct FamDef {
    fam: &'static str,
    t1: f64,
    t2: f64,
    k: usize,
    km: usize,
    has_mst: bool,
}
fn thr_t(t: f64) -> String {
    format!("thr{}", (t * 100.0).round() as i64)
}

/// 按既有因子注册表选择完整/缩减统计包，不改变输出列定义。
fn emit_fam(
    out: &mut Vec<GraphSpec>,
    fd: &FamDef,
    sess: Sess,
    grid: &str,
    f_t1: bool,
    f_t2: bool,
    f_knn: bool,
    f_mst: bool,
) {
    // grid 是 "g30"/"g60"；pm/tl 前缀用格子数：pm30_/tl30_/pm60_/tl60_
    let gnum = grid.trim_start_matches('g');
    let pre = match sess {
        Sess::Day => format!("{}_{}", grid, fd.fam),
        Sess::Pm => format!("pm{}_{}", gnum, fd.fam),
        Sess::Tl => format!("tl{}_{}", gnum, fd.fam),
    };
    out.push(gs(
        &format!("{}_{}", pre, thr_t(fd.t1)),
        fd.fam,
        Kind::Thr(fd.t1),
        f_t1,
        sess,
    ));
    out.push(gs(
        &format!("{}_{}", pre, thr_t(fd.t2)),
        fd.fam,
        Kind::Thr(fd.t2),
        f_t2,
        sess,
    ));
    out.push(gs(
        &format!("{}_knnU{}", pre, fd.k),
        fd.fam,
        Kind::KnnU(fd.k),
        f_knn,
        sess,
    ));
    out.push(gs(
        &format!("{}_knnM{}", pre, fd.km),
        fd.fam,
        Kind::KnnM(fd.km),
        f_knn,
        sess,
    ));
    if fd.has_mst {
        out.push(gs(&format!("{}_mst", pre), fd.fam, Kind::Mst, f_mst, sess));
    }
}

const NEW_FAMS: &[FamDef] = &[
    FamDef {
        fam: "rv",
        t1: 0.20,
        t2: 0.30,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rcnt",
        t1: 0.20,
        t2: 0.30,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "ract",
        t1: 0.05,
        t2: 0.10,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rbal",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rdpt",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rspr",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rmid",
        t1: 0.20,
        t2: 0.30,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rlag",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rrtn",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rvstd",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rvskew",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rvmax",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "ramtcv",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rgapcv",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rgapstd",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rfhalf",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rrnd",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rtunif",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rimb",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rl1c",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rl1g",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rdsda",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rdsdb",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rdsdn",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
];

/// 老族 PM/TL 段沿用各自阈值（day 段维持原表）
const OLD_PM_DEFS: &[FamDef] = &[
    FamDef {
        fam: "rd",
        t1: 0.20,
        t2: 0.30,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "ra",
        t1: 0.25,
        t2: 0.35,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rs",
        t1: 0.08,
        t2: 0.20,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "ro",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "rd60",
        t1: 0.20,
        t2: 0.30,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "ra60",
        t1: 0.25,
        t2: 0.35,
        k: 30,
        km: 10,
        has_mst: true,
    },
    FamDef {
        fam: "ro60",
        t1: 0.15,
        t2: 0.25,
        k: 30,
        km: 10,
        has_mst: true,
    },
];

static GRAPH_SPECS: Lazy<Vec<GraphSpec>> = Lazy::new(|| {
    let mut v: Vec<GraphSpec> = Vec::new();
    macro_rules! G {
        ($t:expr, $f:expr, $k:expr, $fl:expr) => {
            v.push(gs($t, $f, $k, $fl, Sess::Day));
        };
    }

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
        let pre = if sess == Sess::Pm {
            "pm30_rl"
        } else {
            "tl30_rl"
        };
        v.push(gs(
            &format!("{}_thr10", pre),
            "rl",
            Kind::DirThr(0.10),
            true,
            sess,
        ));
        v.push(gs(
            &format!("{}_thr20", pre),
            "rl",
            Kind::DirThr(0.20),
            true,
            sess,
        ));
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
        for s in [
            "deg", "str", "bet", "harm", "pagerank", "kcore", "dist_mkt", "top1", "compsize",
            "neighind",
        ] {
            // The standalone PM split block coexists with the registered PM
            // graph. Distinguish its names so FactorStoreReader can open all
            // 13,574 columns; column positions and numerical values stay fixed.
            let prefix = if tag == "pm" { "pm30_split" } else { "am30" };
            v.push(format!("rlg_{}_rd_thr20_{}", prefix, s));
        }
    }
    for s in [
        "deg", "str", "bet", "harm", "pagerank", "kcore", "dist_mkt", "top1", "compsize",
        "neighind",
    ] {
        v.push(format!("rlg_delta_rd_thr20_{}", s));
    }
    // 市场相关与跨图汇总
    for s in [
        "mkt_corr30",
        "mkt_corr60",
        "day_ret",
        "amt_log",
        "actcells30",
        "actcells60",
        "rd_bet_rank",
        "rd_str_rank",
        "rd_harm_rank",
        "rd_pr_rank",
        "relay_str_norm",
        "relay_free_amt",
        "harm_deg_ratio",
        "bet_deg_ratio",
        "rd_rs_gap",
        "rd_ro_gap",
        "consensus_sim",
        "graph_entropy",
        "vol_corr30",
        "act_frac",
        "pm_rd_bet",
        "tail_rd_deg",
        "pm_rd_str_mean",
        "pm_vs_day_str_change",
    ] {
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
                        c,
                        date,
                        false,
                        true,
                        usize::MAX,
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

    // 行业
    let inds = crate::yupei_dist::industry::industry_of(date, &codes)?
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<i64>>();

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
            if c > 0 {
                s / c as f64
            } else {
                0.0
            }
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
            if c > 0 {
                s / c as f64
            } else {
                0.0
            }
        })
        .collect();
    let bank = SimBank {
        preps,
        n,
        r30: r30.clone(),
        o30: o30.clone(),
        r60: r60.clone(),
        o60: o60.clone(),
        mkt: mkt.clone(),
        mkt60,
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

    // ---- 4. 主图先行（拿 str 定领导节点） ----
    let mut candidate_cache = RelayCandidateCache::new(&live);
    let main_sim = bank.sim("rd", Sess::Day);
    let main_bits = candidate_cache.get(&main_sim, Kind::Thr(0.20));
    let g_main = match main_bits.as_deref() {
        Some(bits) => relay_graph_from_bits(main_sim.as_ref(), bits, false, Some(0.20)),
        None => build_thresh_limited(main_sim.as_ref(), 0.20, &live),
    };
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

    // ---- 5. 逐图算 ----
    let f_total = RELAY_NAMES.len();
    let mut vals = vec![f64::NAN; n * f_total];
    let mut col = 0usize;
    let mut push_cols = |cols: Vec<Vec<f64>>, expect: usize, names: &mut Vec<String>| {
        debug_assert_eq!(cols.len(), expect);
        {
            for i in 0..n {
                let row = &mut vals[i * f_total + col..i * f_total + col + cols.len()];
                for (k, source) in cols.iter().enumerate() {
                    row[k] = source[i];
                }
            }
            col += cols.len();
        }
        let _ = names;
    };
    let mut _dbg_names = Vec::new();
    let mut graph_stats_cache: BTreeMap<(bool, u64), Vec<(Vec<Vec<(u32, f32)>>, Vec<Vec<f64>>)>> =
        BTreeMap::new();

    // GRAPH_BATCH_LOOP_BEGIN
    const GRAPH_BATCH: usize = 4;
    {
        for specs in GRAPH_SPECS.chunks(GRAPH_BATCH) {
            // SimBank owns non-Send Rc/RefCell caches. Populate them on this
            // thread, then share only immutable matrix references with Rayon.
            let matrices: Vec<Rc<Array2<f64>>> =
                specs.iter().map(|sp| bank.sim(sp.fam, sp.sess)).collect();
            let refs: Vec<&Array2<f64>> = matrices.iter().map(|m| m.as_ref()).collect();
            let candidate_owners: Vec<_> = matrices
                .iter()
                .zip(specs)
                .map(|(matrix, sp)| candidate_cache.get(matrix, sp.kind))
                .collect();
            let candidate_refs: Vec<_> = candidate_owners
                .iter()
                .map(|b| b.as_deref().map(Vec::as_slice))
                .collect();
            let outputs = relay_graph_batch(
                specs,
                &refs,
                &live,
                &inds,
                &mkt_corr,
                &lead_set,
                &mut graph_stats_cache,
                Some(&candidate_refs),
            );
            for (i, output) in outputs.into_iter().enumerate() {
                let sp = &specs[i];

                let expect = spec_n_factors(sp);
                push_cols(output.columns, expect, &mut _dbg_names);
            }
        }
    }
    // GRAPH_BATCH_LOOP_END

    // ---- 6. 行业附加 ----
    {
        let g_ind = prune_dead(
            &build_ind_thresh(bank.sim("rd", Sess::Day).as_ref(), &inds, 0.20),
            &live,
        );
        let mut cols = Vec::new();
        let (mut is_, mut id_, mut ib_, mut iz_, mut it_) = (
            vec![f64::NAN; n],
            vec![f64::NAN; n],
            vec![f64::NAN; n],
            vec![f64::NAN; n],
            vec![f64::NAN; n],
        );
        let bet = betweenness_bin(&g_ind);
        let (harm, _, _) = dist_stats(&g_ind);
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
                top1s[i] = g_ind.adj[i]
                    .iter()
                    .map(|&(_, s)| s as f64)
                    .fold(f64::NAN, |a, b| if a.is_nan() || b > a { b } else { a });
            }
            id_[i] = g_ind.adj[i].len() as f64;
            is_[i] = g_ind.adj[i].iter().map(|&(_, s)| s as f64).sum();
            ib_[i] = bet[i];
            it_[i] = top1s[i];
        }
        for (_ind, mem) in byind {
            let hs: Vec<f64> = mem
                .iter()
                .map(|&i| harm[i])
                .filter(|x| x.is_finite())
                .collect();
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
                        if x.is_nan() || y > x {
                            y
                        } else {
                            x
                        }
                    });
                }
                if inds[i] > 0 && !a.is_empty() {
                    let same = a
                        .iter()
                        .filter(|&&(v, _)| inds[v as usize] == inds[i])
                        .count();
                    nind[i] = same as f64 / a.len() as f64;
                }
            }
            let (cid, sizes) = components(g);
            for i in 0..n {
                csz[i] = if cid[i] == u32::MAX {
                    f64::NAN
                } else {
                    sizes[cid[i] as usize] as f64
                };
            }
            let bet = betweenness_bin(g);
            let (harm, _, _) = dist_stats(g);
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
                        if a.is_finite() && b.is_finite() {
                            a - b
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
            .collect();
        push_cols(am_cols, 10, &mut _dbg_names);
        push_cols(pm_cols, 10, &mut _dbg_names);
        push_cols(delta, 10, &mut _dbg_names);
    }

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
                    if c > 0 {
                        s / c as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let m = mv.iter().sum::<f64>() / mv.len() as f64;
            let sd = mv.iter().map(|&x| (x - m) * (x - m)).sum::<f64>().sqrt();
            let mut c = vec![f64::NAN; n];
            for j in 0..n {
                if !live[j] {
                    continue;
                }
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
                .map(|i| {
                    bank.preps[i]
                        .as_ref()
                        .map(|p| p.day_ret)
                        .unwrap_or(f64::NAN)
                })
                .collect(),
        );
        cols.push(
            amount
                .iter()
                .map(|&x| {
                    if x.is_finite() && x > 0.0 {
                        x.ln()
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        );
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
        tmp[i] = if b.is_finite() && s.is_finite() && s > 0.0 {
            b / s
        } else {
            f64::NAN
        };
        vals[i * f_total + x_base + 10] = tmp[i];
        // relay_free_amt = bet / (amt_log+1)
        vals[i * f_total + x_base + 11] = if b.is_finite() && amt.is_finite() {
            b / (amt + 1.0)
        } else {
            f64::NAN
        };
        // harm_deg_ratio
        vals[i * f_total + x_base + 12] = if h.is_finite() && d.is_finite() && d > 0.0 {
            h / d
        } else {
            f64::NAN
        };
        // bet_deg_ratio
        vals[i * f_total + x_base + 13] = if b.is_finite() && d.is_finite() && d > 0.0 {
            b / d
        } else {
            f64::NAN
        };
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
            vals[i * f_total + x_base + 14] = if rd.is_finite() && rs.is_finite() {
                rd - rs
            } else {
                f64::NAN
            };
            vals[i * f_total + x_base + 15] = if rd.is_finite() && ro.is_finite() {
                rd - ro
            } else {
                f64::NAN
            };
        }
    }
    // consensus_sim = 四个主图 strength 的一致性（rd/ra/rs/ro 在 thr25 的 str 均值）
    {
        let mut sum = vec![0.0f64; n];
        let mut cnt = vec![0usize; n];
        for tag in [
            "g30_rd_thr25",
            "g30_ra_thr25",
            "g30_rs_thr15",
            "g30_ro_thr25",
        ] {
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
            vals[i * f_total + x_base + 16] = if cnt[i] > 0 {
                sum[i] / cnt[i] as f64
            } else {
                f64::NAN
            };
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
        for tag in [
            "pm30_rd_thr20",
            "pm30_rd_thr30",
            "pm30_rd_knnU30",
            "pm30_rd_knnM10",
            "pm30_rd_mst",
        ] {
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
            vals[i * f_total + x_base + 22] = if cnt[i] > 0 {
                sum[i] / cnt[i] as f64
            } else {
                f64::NAN
            };
        }
    }
    // pm_vs_day_str_change = PM 与 day 同图(rd_thr20) str 之差
    {
        let off_pm = col_offset_of("pm30_rd_thr20") + 1;
        let off_day = col_offset_of("g30_rd_thr20") + 1;
        for i in 0..n {
            let (a, b) = (vals[i * f_total + off_pm], vals[i * f_total + off_day]);
            vals[i * f_total + x_base + 23] = if a.is_finite() && b.is_finite() {
                a - b
            } else {
                f64::NAN
            };
        }
    }

    // ---- 转 f32 输出 ----
    assert_eq!(
        col, f_total,
        "relaynet 列数与注册表不一致: {col} != {f_total}"
    );
    let out: Vec<f32> = vals.iter().map(|&x| x as f32).collect();
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
            base + if let Kind::Mst = sp.kind {
                MST_EXTRA.len()
            } else {
                0
            }
        }
    }
}

// ============================================================================
// PyO3
// ============================================================================

#[pyfunction]
pub fn py_relaynet_six_people(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_relaynet_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_relaynet_six_people_names() -> Vec<String> {
    relaynet_names()
}

fn relay_candidate_bits(s: &Array2<f64>, k: usize, allow: Option<&[bool]>) -> Vec<Vec<u64>> {
    let n = s.nrows();
    let nw = (n + 63) / 64;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut bits = vec![0u64; nw];
            if k == 0 || allow.map_or(false, |a| !a[i]) {
                return bits;
            }
            {
                // The worst retained candidate is minimum similarity, then largest code index.
                let mut heap: BinaryHeap<(std::cmp::Reverse<OrderedF>, u32)> =
                    BinaryHeap::with_capacity(k);
                for j in 0..n {
                    if j == i || allow.map_or(false, |a| !a[j]) {
                        continue;
                    }
                    let v = s[[i, j]];
                    if !v.is_finite() {
                        continue;
                    }
                    if heap.len() < k {
                        heap.push((std::cmp::Reverse(OrderedF(v)), j as u32));
                    } else {
                        let worst = heap.peek().unwrap();
                        if v > worst.0 .0 .0 || (v == worst.0 .0 .0 && (j as u32) < worst.1) {
                            heap.pop();
                            heap.push((std::cmp::Reverse(OrderedF(v)), j as u32));
                        }
                    }
                }
                for (_, j) in heap {
                    bits[j as usize / 64] |= 1u64 << (j as usize % 64);
                }
            }
            bits
        })
        .collect()
}

fn relay_graph_from_bits(
    s: &Array2<f64>,
    bits: &[Vec<u64>],
    mutual: bool,
    threshold: Option<f64>,
) -> UGraph {
    let n = s.nrows();
    let nw = (n + 63) / 64;
    // Transpose only retained candidates; the inverse rows then have contiguous reads.
    let mut reverse = vec![vec![0u64; nw]; n];
    for i in 0..n {
        for (word, &raw) in bits[i].iter().enumerate() {
            let mut remaining = raw;
            while remaining != 0 {
                let j = word * 64 + remaining.trailing_zeros() as usize;
                reverse[j][i / 64] |= 1u64 << (i % 64);
                remaining &= remaining - 1;
            }
        }
    }
    let adj = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut a = Vec::new();
            for word in 0..nw {
                let mut keep = if mutual {
                    bits[i][word] & reverse[i][word]
                } else {
                    bits[i][word] | reverse[i][word]
                };
                // Increasing words/bits give exactly the original increasing-neighbor order.
                while keep != 0 {
                    let j = word * 64 + keep.trailing_zeros() as usize;
                    keep &= keep - 1;
                    if i == j {
                        continue;
                    }
                    // The original undirected builder reads the upper triangle for both directions.
                    let v = s[[i.min(j), i.max(j)]];
                    if v.is_finite() && threshold.map_or(true, |t| v > t) {
                        a.push((j as u32, v as f32));
                    }
                }
            }
            a
        })
        .collect();
    UGraph { n, adj }
}

fn dir_costs_are_reversible_exact(g: &DGraph) -> bool {
    const SCALE: f64 = 4294967296.0;
    const LIMIT: f64 = 9007199254740992.0;
    let mut max_scaled = 0.0f64;
    for adj in &g.out_adj {
        for &(_, sim) in adj {
            let scaled = (1.0 - sim as f64).max(1e-9) * SCALE;
            if !scaled.is_finite() || scaled <= 0.0 || scaled.fract() != 0.0 {
                return false;
            }
            max_scaled = max_scaled.max(scaled);
        }
    }
    max_scaled * (g.n as f64) < LIMIT
}

fn dir_harmonic_pair_exact(g: &DGraph) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if g.out_adj.iter().all(Vec::is_empty) {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }
    if !dir_costs_are_reversible_exact(g) {
        return (dir_harmonic(g, false), dir_harmonic(g, true));
    }
    let mut distances = vec![f64::INFINITY; n * n];
    let mut outgoing = vec![f64::NAN; n];
    distances
        .par_chunks_mut(n)
        .zip(outgoing.par_iter_mut())
        .enumerate()
        .for_each(|(s, (d, out))| {
            d[s] = 0.0;
            if g.out_adj[s].is_empty() {
                return;
            }
            let mut heap = BinaryHeap::new();
            heap.push(State { d: 0.0, u: s });
            while let Some(State { d: du, u }) = heap.pop() {
                if du != d[u] {
                    continue;
                }
                for &(v, sim) in &g.out_adj[u] {
                    let w = (1.0 - sim as f64).max(1e-9);
                    let candidate = du + w;
                    if candidate < d[v as usize] {
                        d[v as usize] = candidate;
                        heap.push(State {
                            d: candidate,
                            u: v as usize,
                        });
                    }
                }
            }
            let (mut h, mut count) = (0.0, 0usize);
            for (j, &x) in d.iter().enumerate() {
                if j != s && x.is_finite() && x > 0.0 {
                    h += 1.0 / x;
                    count += 1;
                }
            }
            if count > 0 {
                *out = h / count as f64;
            }
        });
    let mut incoming = vec![f64::NAN; n];
    const TILE: usize = 32;
    incoming
        .par_chunks_mut(TILE)
        .enumerate()
        .for_each(|(block, target)| {
            let lo = block * TILE;
            let mut sums = [0.0f64; TILE];
            let mut counts = [0usize; TILE];
            for source in 0..n {
                let row = &distances[source * n + lo..source * n + lo + target.len()];
                for (lane, &x) in row.iter().enumerate() {
                    if source != lo + lane && x.is_finite() && x > 0.0 {
                        sums[lane] += 1.0 / x;
                        counts[lane] += 1;
                    }
                }
            }
            for lane in 0..target.len() {
                if counts[lane] > 0 {
                    target[lane] = sums[lane] / counts[lane] as f64;
                }
            }
        });
    (outgoing, incoming)
}

fn dir_two_hop_exact(adj: &[Vec<(u32, f32)>]) -> Vec<f64> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let nw = n.div_ceil(64);
    let mut bits = vec![0u64; n * nw];
    bits.par_chunks_mut(nw).enumerate().for_each(|(i, row)| {
        for &(v, _) in &adj[i] {
            row[v as usize / 64] |= 1u64 << (v as usize % 64);
        }
    });
    (0..n)
        .into_par_iter()
        .map_init(
            || vec![0u64; nw],
            |ball, i| {
                ball.copy_from_slice(&bits[i * nw..(i + 1) * nw]);
                ball[i / 64] |= 1u64 << (i % 64);
                for &(v, _) in &adj[i] {
                    let v = v as usize;
                    if adj[v].len() < nw {
                        for &(w, _) in &adj[v] {
                            ball[w as usize / 64] |= 1u64 << (w as usize % 64);
                        }
                    } else {
                        for (out, &word) in ball.iter_mut().zip(&bits[v * nw..(v + 1) * nw]) {
                            *out |= word;
                        }
                    }
                }
                ball.iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>() as f64
                    - 1.0
            },
        )
        .collect()
}

fn dir_pagerank_pair_exact(g: &DGraph, outstr: &[f64], instr: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    // Sorted incoming source IDs reproduce the corrected serial push sum order.
    let incoming: Vec<_> = g
        .in_adj
        .iter()
        .map(|a| {
            let mut a = a.clone();
            a.sort_by_key(|&(v, _)| v);
            a
        })
        .collect();
    let reverse_incoming: Vec<_> = g
        .out_adj
        .iter()
        .map(|a| {
            let mut a = a.clone();
            a.sort_by_key(|&(v, _)| v);
            a
        })
        .collect();
    let mut pr = vec![1.0 / n as f64; n];
    let mut pr2 = pr.clone();
    let mut next = vec![0.0; n];
    let mut next2 = vec![0.0; n];
    for _ in 0..50 {
        let (mut dangling, mut dangling2) = (0.0, 0.0);
        for i in 0..n {
            if outstr[i] <= 0.0 {
                dangling += pr[i];
            }
            if instr[i] <= 0.0 {
                dangling2 += pr2[i];
            }
        }
        next.par_iter_mut()
            .zip(next2.par_iter_mut())
            .enumerate()
            .for_each(|(i, (out, out2))| {
                let mut value = 0.15 / n as f64;
                for &(src, w) in &incoming[i] {
                    let src = src as usize;
                    if outstr[src] <= 0.0 {
                        continue;
                    }
                    value += 0.85 * pr[src] * w as f64 / outstr[src];
                }
                *out = value + 0.85 * dangling / n as f64;
                let mut value = 0.15 / n as f64;
                for &(src, w) in &reverse_incoming[i] {
                    let src = src as usize;
                    if instr[src] <= 0.0 {
                        continue;
                    }
                    value += 0.85 * pr2[src] * w as f64 / instr[src];
                }
                *out2 = value + 0.85 * dangling2 / n as f64;
            });
        std::mem::swap(&mut pr, &mut next);
        std::mem::swap(&mut pr2, &mut next2);
    }
    (pr, pr2)
}

type RelayGraphStatsCache = BTreeMap<(bool, u64), Vec<(Vec<Vec<(u32, f32)>>, Vec<Vec<f64>>)>>;

enum RelayGraphWork {
    Undirected(UGraph),
    Directed(DGraph),
}
struct RelayGraphOutput {
    columns: Vec<Vec<f64>>,
}
fn relay_graph_hash(g: &UGraph) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    g.n.hash(&mut h);
    for row in &g.adj {
        row.len().hash(&mut h);
        for &(v, w) in row {
            v.hash(&mut h);
            w.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Parallelize independent graphs within the existing bounded compute pool.
/// All cache lookups/inserts and output assembly remain in specification order.
fn relay_graph_batch(
    specs: &[GraphSpec],
    sims: &[&Array2<f64>],
    live: &[bool],
    inds: &[i64],
    mkt: &[f64],
    leaders: &[bool],
    cache: &mut RelayGraphStatsCache,
    candidates: Option<&[Option<&[Vec<u64>]>]>,
) -> Vec<RelayGraphOutput> {
    let prepared: Vec<(RelayGraphWork, u64)> = specs
        .par_iter()
        .enumerate()
        .map(|(i, sp)| {
            let sim = sims[i];
            let bits = candidates.and_then(|entries| entries[i]);
            let graph = match sp.kind {
                Kind::DirThr(t) => RelayGraphWork::Directed(build_digraph(sim, t)),
                _ => {
                    let g = match sp.kind {
                        Kind::Thr(t) => match bits {
                            Some(b) => relay_graph_from_bits(sim, b, false, Some(t)),
                            None => build_thresh_limited(sim, t, live),
                        },
                        Kind::KnnU(k) => prune_dead(
                            &match bits {
                                Some(b) => relay_graph_from_bits(sim, b, false, None),
                                None => build_knn(sim, k, false),
                            },
                            live,
                        ),
                        Kind::KnnM(k) => prune_dead(
                            &match bits {
                                Some(b) => relay_graph_from_bits(sim, b, true, None),
                                None => build_knn(sim, k, true),
                            },
                            live,
                        ),
                        Kind::Mst => prune_dead(&build_mst(sim), live),
                        Kind::IndThr(t) => prune_dead(&build_ind_thresh(sim, inds, t), live),
                        Kind::DirThr(_) => unreachable!(),
                    };
                    RelayGraphWork::Undirected(g)
                }
            };
            let hash = match &graph {
                RelayGraphWork::Undirected(g) => relay_graph_hash(g),
                _ => 0,
            };
            (graph, hash)
        })
        .collect();
    let count = specs.len();
    let mut base: Vec<Option<Vec<Vec<f64>>>> = (0..count).map(|_| None).collect();
    let mut duplicate = vec![None; count];
    let mut cached = vec![false; count];
    for i in 0..count {
        if let RelayGraphWork::Undirected(g) = &prepared[i].0 {
            let key = (specs[i].full, prepared[i].1);
            if let Some((_, cols)) = cache.get(&key).and_then(|bucket| {
                bucket
                    .iter()
                    .find(|(adj, _)| relay_graph_adjacency_equal(adj, &g.adj))
            }) {
                base[i] = Some(cols.clone());
                cached[i] = true;
                continue;
            }
            duplicate[i]=(0..i).find(|&j| {
                specs[j].full==specs[i].full && prepared[j].1==prepared[i].1 &&
                matches!(&prepared[j].0,RelayGraphWork::Undirected(previous) if relay_graph_adjacency_equal(&previous.adj, &g.adj))
            });
            cached[i] = duplicate[i].is_some();
        }
    }
    let mut computed: Vec<Option<Vec<Vec<f64>>>> = (0..count)
        .into_par_iter()
        .map(|i| {
            if cached[i] {
                return None;
            }
            let cols = match &prepared[i].0 {
                RelayGraphWork::Undirected(g) => {
                    stat_undir(g, sims[i], inds, mkt, leaders, specs[i].full)
                }
                RelayGraphWork::Directed(g) => dir_stats(g),
            };
            Some(cols)
        })
        .collect();
    for i in 0..count {
        if let Some(j) = duplicate[i] {
            base[i] = Some(base[j].as_ref().expect("earlier batch result").clone());
        } else if let Some(cols) = computed[i].take() {
            if let RelayGraphWork::Undirected(g) = &prepared[i].0 {
                cache
                    .entry((specs[i].full, prepared[i].1))
                    .or_default()
                    .push((g.adj.clone(), cols.clone()));
            }
            base[i] = Some(cols);
        }
    }
    // MST extras reuse full-graph harmonics exactly as in the serial path.
    base.into_par_iter()
        .enumerate()
        .map(|(i, cols)| {
            let mut cols = cols.expect("computed graph result");
            if let RelayGraphWork::Undirected(g) = &prepared[i].0 {
                if matches!(specs[i].kind, Kind::Mst) {
                    let extra = mst_extra(
                        g,
                        mkt,
                        if specs[i].full {
                            Some(cols[9].as_slice())
                        } else {
                            None
                        },
                    );
                    cols.extend(extra);
                }
            }
            RelayGraphOutput { columns: cols }
        })
        .collect()
}

fn relay_graph_adjacency_equal(a: &[Vec<(u32, f32)>], b: &[Vec<(u32, f32)>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|(&(vx, wx), &(vy, wy))| vx == vy && wx.to_bits() == wy.to_bits())
        })
}

struct BinaryDirectionGraph {
    offsets: Vec<usize>,
    neighbors: Vec<u32>,
    component: Vec<u32>,
    sizes: Vec<u32>,
    component_edges: Vec<usize>,
    sorted: bool,
}
impl BinaryDirectionGraph {
    fn new(g: &UGraph) -> Self {
        let (component, sizes) = components(g);
        let mut component_edges = vec![0usize; sizes.len()];
        let mut offsets = Vec::with_capacity(g.n + 1);
        let mut neighbors = Vec::with_capacity(g.adj.iter().map(Vec::len).sum());
        let mut sorted = true;
        offsets.push(0);
        for (i, row) in g.adj.iter().enumerate() {
            sorted &= row.windows(2).all(|pair| pair[0].0 < pair[1].0);
            component_edges[component[i] as usize] += row.len();
            neighbors.extend(row.iter().map(|&(v, _)| v));
            offsets.push(neighbors.len());
        }
        Self {
            offsets,
            neighbors,
            component,
            sizes,
            component_edges,
            sorted,
        }
    }
}

struct BinaryDirectionScratch {
    dist: Vec<i32>,
    sigma: Vec<f64>,
    delta: Vec<f64>,
    queue: Vec<usize>,
    rank: Vec<usize>,
    parents: Vec<usize>,
    found: Vec<(usize, usize)>,
    pred_head: Vec<u32>,
    pred_from: Vec<u32>,
    pred_next: Vec<u32>,
}
impl BinaryDirectionScratch {
    fn new(n: usize) -> Self {
        Self {
            dist: vec![-1; n],
            sigma: vec![0.0; n],
            delta: vec![0.0; n],
            queue: Vec::with_capacity(n),
            rank: vec![usize::MAX; n],
            parents: Vec::new(),
            found: Vec::with_capacity(n),
            pred_head: vec![u32::MAX; n],
            pred_from: Vec::new(),
            pred_next: Vec::new(),
        }
    }
    fn bfs(&mut self, g: &BinaryDirectionGraph, src: usize) {
        self.dist.fill(-1);
        self.sigma.fill(0.0);
        self.delta.fill(0.0);
        self.queue.clear();
        self.rank.fill(usize::MAX);
        {
            self.pred_head.fill(u32::MAX);
            self.pred_from.clear();
            self.pred_next.clear();
        }
        self.dist[src] = 0;
        self.sigma[src] = 1.0;
        self.queue.push(src);
        self.rank[src] = 0;
        let cid = g.component[src] as usize;
        let mut remaining = g.sizes[cid] as usize - 1;
        let mut unvisited_edges = g.component_edges[cid] - (g.offsets[src + 1] - g.offsets[src]);
        let mut begin = 0;
        while begin < self.queue.len() {
            if remaining == 0 {
                break;
            }
            let end = self.queue.len();
            let depth = self.dist[self.queue[begin]];
            let frontier_edges: usize = self.queue[begin..end]
                .iter()
                .map(|&v| g.offsets[v + 1] - g.offsets[v])
                .sum();
            if g.sorted && remaining > 0 && frontier_edges > unvisited_edges.saturating_mul(4) {
                self.found.clear();
                for v in 0..self.dist.len() {
                    if self.dist[v] >= 0 || g.component[v] as usize != cid {
                        continue;
                    }
                    self.parents.clear();
                    for &u in &g.neighbors[g.offsets[v]..g.offsets[v + 1]] {
                        let u = u as usize;
                        if self.dist[u] == depth {
                            self.parents.push(u);
                        }
                    }
                    if self.parents.is_empty() {
                        continue;
                    }
                    // The push kernel visits parents in queue order, not vertex-ID order.
                    self.parents.sort_unstable_by_key(|&u| self.rank[u]);
                    let mut paths = 0.0;
                    for &u in &self.parents {
                        paths += self.sigma[u];
                    }
                    self.sigma[v] = paths;
                    self.dist[v] = depth + 1;
                    {
                        for &u in &self.parents {
                            self.pred_from.push(u as u32);
                            self.pred_next.push(self.pred_head[v]);
                            self.pred_head[v] = (self.pred_from.len() - 1) as u32;
                        }
                    }
                    self.found.push((self.rank[self.parents[0]], v));
                    remaining -= 1;
                    unvisited_edges -= g.offsets[v + 1] - g.offsets[v];
                }
                // With strictly sorted adjacency, discovery is lexicographic:
                // first parent queue position, then target vertex ID.
                self.found.sort_unstable();
                for &(_, v) in &self.found {
                    self.rank[v] = self.queue.len();
                    self.queue.push(v);
                }
            } else {
                for qi in begin..end {
                    let u = self.queue[qi];
                    let sigma_u = self.sigma[u];
                    for &v in &g.neighbors[g.offsets[u]..g.offsets[u + 1]] {
                        let v = v as usize;
                        if self.dist[v] < 0 {
                            self.dist[v] = depth + 1;
                            self.rank[v] = self.queue.len();
                            self.queue.push(v);
                            remaining -= 1;
                            unvisited_edges -= g.offsets[v + 1] - g.offsets[v];
                        }
                        if self.dist[v] == depth + 1 {
                            self.sigma[v] += sigma_u;
                            {
                                self.pred_from.push(u as u32);
                                self.pred_next.push(self.pred_head[v]);
                                self.pred_head[v] = (self.pred_from.len() - 1) as u32;
                            }
                        }
                    }
                }
            }
            begin = end;
        }
    }
}

fn binary_direction_run(g: &UGraph) -> Vec<f64> {
    if let Some(bet) = forest_bet(g) {
        return bet;
    }
    let n = g.n;
    let graph = BinaryDirectionGraph::new(g);
    const BLOCK: usize = 16;
    let parts: Vec<Vec<f64>> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|block| {
            let mut scratch = BinaryDirectionScratch::new(n);
            let mut bet = vec![0.0; n];
            for src in block * BLOCK..((block + 1) * BLOCK).min(n) {
                if graph.offsets[src] == graph.offsets[src + 1] {
                    continue;
                }
                scratch.bfs(&graph, src);
                for &u in scratch.queue.iter().rev() {
                    // Each predecessor is a distinct target. Preserve the reverse
                    // queue order of additions into each target's delta exactly.
                    let mut link = scratch.pred_head[u];
                    let sigma_u = scratch.sigma[u];
                    let factor = 1.0 + scratch.delta[u];
                    while link != u32::MAX {
                        let v = scratch.pred_from[link as usize] as usize;
                        scratch.delta[v] += (scratch.sigma[v] / sigma_u) * factor;
                        link = scratch.pred_next[link as usize];
                    }
                    if u != src {
                        bet[u] += scratch.delta[u];
                    }
                }
            }
            bet
        })
        .collect();
    let mut bet = vec![0.0; n];
    for part in parts {
        for (a, b) in bet.iter_mut().zip(part) {
            *a += b;
        }
    }
    for b in &mut bet {
        *b *= 0.5;
    }
    bet
}

fn undir_basic_parallel_exact(g: &UGraph) -> Vec<Vec<f64>> {
    let rows: Vec<[f64; 6]> = (0..g.n)
        .into_par_iter()
        .map_init(Vec::<f64>::new, |sims, i| {
            let a = &g.adj[i];
            let mut row = [
                a.len() as f64,
                a.iter().map(|&(_, x)| x as f64).sum(),
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ];
            if !a.is_empty() {
                sims.clear();
                sims.extend(a.iter().map(|&(_, x)| x as f64));
                sims.sort_by(|x, y| y.total_cmp(x));
                row[2] = sims[0];
                row[3] = sims.iter().take(3).sum::<f64>() / sims.len().min(3) as f64;
                row[4] = row[1] / a.len() as f64;
                // Keep this sorted sum distinct from the adjacency-order strength.
                let ss: f64 = sims.iter().sum();
                if ss > 0.0 {
                    row[5] = -sims
                        .iter()
                        .map(|&p| {
                            let q = p / ss;
                            if q > 0.0 {
                                q * q.ln()
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                }
            }
            row
        })
        .collect();
    (0..6)
        .map(|k| rows.iter().map(|row| row[k]).collect())
        .collect()
}

fn triangle_p75_select_exact(values: &mut [f64]) -> f64 {
    let pos = 0.75 * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    let (left, upper, _) = values.select_nth_unstable_by(hi, |x, y| x.total_cmp(y));
    let high = *upper;
    let low = if lo == hi {
        high
    } else {
        *left.select_nth_unstable_by(lo, |x, y| x.total_cmp(y)).1
    };
    low + (high - low) * (pos - lo as f64)
}

fn clustering_triangle_pair_exact(g: &UGraph) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    let rows: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map_init(
            || (vec![0.0f32; n], vec![usize::MAX; n], Vec::<f64>::new()),
            |(weights, stamps, triw), i| {
                triw.clear();
                let di = g.deg(i);
                if di < 2 {
                    return (0.0, f64::NAN);
                }
                for &(v, weight) in &g.adj[i] {
                    weights[v as usize] = weight;
                    stamps[v as usize] = i;
                }
                let mut acc = 0.0;
                // This is the original clustering loop, in its original order and
                // with its original product association. Quantile values are only
                // retained when k>j, exactly the original tri_p75 subset.
                for &(j, sij) in &g.adj[i] {
                    for &(k, sjk) in &g.adj[j as usize] {
                        if k as usize != i && stamps[k as usize] == i {
                            let ski = weights[k as usize];
                            let prod = sij as f64 * sjk as f64 * ski as f64;
                            if prod > 0.0 {
                                let value = prod.powf(1.0 / 3.0);
                                acc += value;
                                if k > j {
                                    triw.push(value);
                                }
                            }
                        }
                    }
                }
                let p75 = if triw.is_empty() {
                    f64::NAN
                } else {
                    triangle_p75_select_exact(triw)
                };
                (acc / (di as f64 * (di - 1) as f64), p75)
            },
        )
        .collect();
    rows.into_iter().unzip()
}

struct RelayCandidateCache<'a> {
    live: &'a [bool],
    entries: BTreeMap<(usize, usize, bool), (Rc<Array2<f64>>, Rc<Vec<Vec<u64>>>)>,
}
impl<'a> RelayCandidateCache<'a> {
    fn new(live: &'a [bool]) -> Self {
        Self {
            live,
            entries: BTreeMap::new(),
        }
    }
    fn get(&mut self, matrix: &Rc<Array2<f64>>, kind: Kind) -> Option<Rc<Vec<Vec<u64>>>> {
        let n = matrix.nrows();
        let (k, limited) = match kind {
            Kind::Thr(_) => ((n / 25).max(200).min(n.saturating_sub(1)).max(1), true),
            Kind::KnnU(k) | Kind::KnnM(k) => (k, false),
            _ => return None,
        };
        let key = (Rc::as_ptr(matrix) as usize, k, limited);
        if let Some((_, bits)) = self.entries.get(&key) {
            return Some(bits.clone());
        }
        let bits = Rc::new(relay_candidate_bits(
            matrix.as_ref(),
            k,
            if limited { Some(self.live) } else { None },
        ));
        self.entries.insert(key, (matrix.clone(), bits.clone()));
        Some(bits)
    }
}

/// Screening rejects only candidate > current target distance, retaining ties.
/// Targets are unique within a row (checked by the guard), and accepted lanes
/// execute the unchanged scalar update in ascending lane order. Rejected edges
/// cannot become useful after another relaxation decreases a target distance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn weighted_screen4(du: f64, a: &[(u32, f32)], d: *const f64) -> u32 {
    use std::arch::x86_64::*;
    let index = _mm_set_epi32(a[3].0 as i32, a[2].0 as i32, a[1].0 as i32, a[0].0 as i32);
    let sim = _mm256_cvtps_pd(_mm_set_ps(a[3].1, a[2].1, a[1].1, a[0].1));
    let cost = _mm256_max_pd(
        _mm256_sub_pd(_mm256_set1_pd(1.0), sim),
        _mm256_set1_pd(1e-9),
    );
    let candidate = _mm256_add_pd(_mm256_set1_pd(du), cost);
    let target = _mm256_i32gather_pd::<8>(d, index);
    _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LE_OQ>(candidate, target)) as u32
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn weighted_screen4(_: f64, _: &[(u32, f32)], _: *const f64) -> u32 {
    15
}

fn weighted_screen_guard(g: &UGraph) -> bool {
    if g.n > i32::MAX as usize {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if !std::is_x86_feature_detected!("avx2") {
        return false;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        return false;
    }
    let mut seen = vec![usize::MAX; g.n];
    for (i, row) in g.adj.iter().enumerate() {
        for &(v, _) in row {
            let v = v as usize;
            if v >= g.n || seen[v] == i {
                return false;
            }
            seen[v] = i;
        }
    }
    true
}
impl PathScratch {
    fn source_screen4<const WEIGHTED: bool>(
        &mut self,
        g: &UGraph,
        src: usize,
        tree: bool,
        bet: &mut [f64],
    ) -> [f64; 7] {
        self.dist.fill(f64::INFINITY);
        self.dist[src] = 0.0;
        self.order.clear();
        self.values.clear();
        self.heap.clear();
        if WEIGHTED && !tree {
            self.sigma.fill(0.0);
            self.sigma[src] = 1.0;
            self.delta.fill(0.0);
            {
                self.parent.fill(u32::MAX);
                self.head.fill(u32::MAX);
                self.pred_from.clear();
                self.pred_next.clear();
            }
        }
        if tree {
            self.order.push(src);
            let mut head = 0;
            while head < self.order.len() {
                let u = self.order[head];
                head += 1;
                let du = self.dist[u];
                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    if !self.dist[v].is_finite() {
                        self.dist[v] = du + (1.0 - sim as f64).max(1e-9);
                        self.order.push(v);
                    }
                }
            }
        } else {
            self.heap.push(State { d: 0.0, u: src });
            while let Some(State { d: du, u }) = self.heap.pop() {
                if du != self.dist[u] {
                    continue;
                }
                if WEIGHTED {
                    self.order.push(u);
                }
                for chunk in g.adj[u].chunks(4) {
                    let mut mask = if chunk.len() == 4 {
                        unsafe { weighted_screen4(du, chunk, self.dist.as_ptr()) }
                    } else {
                        (1u32 << chunk.len()) - 1
                    };
                    while mask != 0 {
                        let lane = mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        let (v, sim) = chunk[lane];
                        let v = v as usize;
                        let nd = du + (1.0 - sim as f64).max(1e-9);
                        if nd < self.dist[v] {
                            self.dist[v] = nd;
                            self.heap.push(State { d: nd, u: v });
                            if WEIGHTED {
                                self.sigma[v] = self.sigma[u];
                                {
                                    self.parent[v] = u as u32;
                                    self.head[v] = u32::MAX;
                                }
                            }
                        } else if WEIGHTED && nd == self.dist[v] {
                            self.sigma[v] += self.sigma[u];
                            {
                                self.pred_from.push(u as u32);
                                self.pred_next.push(self.head[v]);
                                self.head[v] = (self.pred_from.len() - 1) as u32;
                            }
                        }
                    }
                }
            }
        }
        if WEIGHTED && !tree {
            for &u in self.order.iter().rev() {
                {
                    let first = self.parent[u];
                    if first != u32::MAX {
                        let v = first as usize;
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                    }
                    let mut link = self.head[u];
                    while link != u32::MAX {
                        let v = self.pred_from[link as usize] as usize;
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                        link = self.pred_next[link as usize];
                    }
                }
                if u != src {
                    bet[u] += self.delta[u];
                }
            }
        }
        let (mut h, mut f, mut e) = (0.0, 0.0, 0.0f64);
        // Preserve the original node-index summation order exactly.
        for (j, &d) in self.dist.iter().enumerate() {
            if j != src && d.is_finite() && d > 0.0 {
                h += 1.0 / d;
                f += d;
                e = e.max(d);
                self.values.push(d);
            }
        }
        if self.values.is_empty() {
            return [f64::NAN; 7];
        }
        self.values.sort_unstable_by(|a, b| a.total_cmp(b));
        let count = self.values.len() as f64;
        [
            h / count,
            f / count,
            e,
            quantile_sorted(&self.values, 0.25),
            quantile_sorted(&self.values, 0.75),
            quantile_sorted(&self.values, 0.50),
            quantile_sorted(&self.values, 0.90),
        ]
    }
}
fn exact_paths_screen4(g: &UGraph, weighted: bool) -> ExactPaths {
    if !weighted_screen_guard(g) {
        return exact_paths_mode(g, weighted);
    }
    let n = g.n;
    let forest = forest_bet(g);
    let tree = forest.is_some();
    // All n sources are used, including isolates (NaN distances, zero bet).
    // Fixed consecutive blocks give deterministic reduction across thread counts.
    const BLOCK: usize = 16;
    let parts: Vec<(Vec<[f64; 7]>, Vec<f64>)> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|block| {
            let lo = block * BLOCK;
            let hi = (lo + BLOCK).min(n);
            let mut scratch = PathScratch::new(n);
            let mut bet = vec![0.0; n];
            let rows = (lo..hi)
                .map(|src| {
                    if weighted {
                        scratch.source_screen4::<true>(g, src, tree, &mut bet)
                    } else {
                        scratch.source_screen4::<false>(g, src, tree, &mut bet)
                    }
                })
                .collect();
            (rows, bet)
        })
        .collect();
    let mut columns: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::with_capacity(n));
    let mut bet = forest.unwrap_or_else(|| vec![0.0; n]);
    for (rows, source_bet) in parts {
        for row in rows {
            for j in 0..7 {
                columns[j].push(row[j]);
            }
        }
        if weighted && !tree {
            for (a, b) in bet.iter_mut().zip(source_bet) {
                *a += b;
            }
        }
    }
    if weighted && !tree {
        for b in &mut bet {
            *b *= 0.5;
        }
    }
    ExactPaths { columns, bet }
}
