//! 可观测挂单比例指标：基于盘口快照的可观测挂单占比序列，
//! 划分低占比时段，在每个时段内计算盘口与逐笔成交的多维统计特征。
//!
//! 对 side ∈ {bid, ask} 各跑一遍完整流程，再对 method ∈ {A, B} 各划分一次，
//! 共 4 套配置 (side, method)。每套返回 (seg_2d, pre5_2d) 两个二维数组：
//!   seg_2d:  行=该配置的小段, 列=147个指标
//!   pre5_2d: 行=同上小段, 列同上, 每行用「起点前 pre_minutes 分钟」窗口算
//!
//! 方法A：连续低于分位点的快照合并成一段；段的时间范围 = 段首快照 → 段末快照的
//!        下一次快照；段内快照数 < drop_min 则丢弃。
//! 方法B：穿越点（前一次 ≥ 分位点、本次 < 分位点），每点取其后 forward_sec 秒作为一段。

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};

// ============================================================================
// 参数
// ============================================================================

/// 可观测挂单比例指标的全部可调参数。
#[derive(Clone, Debug)]
pub struct ObsOrderParams {
    /// 分位点（全天 obs_ratio 序列的分位数阈值），默认 0.10
    pub q: f32,
    /// 方法A 段最小快照数（不足则丢弃），默认 10
    pub drop_min: usize,
    /// 方法B 向后秒数，默认 60
    pub forward_sec: f32,
    /// 起点前窗口分钟数，默认 5
    pub pre_minutes: f32,
    /// 滚动窗口快照数（过去含当前），默认 20
    pub rolling_n: usize,
    /// 自相关滞后阶数，默认 1
    pub autocorr_lag: usize,
}

impl Default for ObsOrderParams {
    fn default() -> Self {
        Self {
            q: 0.10,
            drop_min: 10,
            forward_sec: 60.0,
            pre_minutes: 5.0,
            rolling_n: 20,
            autocorr_lag: 1,
        }
    }
}

/// 每套配置的内部计算列数（盘口序列60 + 盘口标量13 + 逐笔序列65 + 逐笔标量9）。
pub const NCOLS: usize = 147;

/// 冗余分析后删除的列索引：
/// 第一轮 33 列（跨股时序 第3小|corr|>=0.6），第二轮 26 列（截面|corr|>=0.5），
/// 第三轮 17 列（am_b/s/d/da 族 NaN>96%，计算逻辑缺陷致除零）。
pub const DELETE_COLS: [usize; 76] = [
    2, 10, 11, 12, 13, 14, 15, 16, 24, 28, 33, 35, 41, 45, 46, 48, 49, 51, 55, 56, 57, 61, 64, 65,
    66, 67, 68, 69, 70, 72, 78, 79, 81, 85, 89, 91, 92, 94, 99, 101, 102, 103, 104, 105, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
    129, 130, 131, 132, 134, 135, 136, 137, 139, 141, 143,
];

/// 输出列数 = NCOLS - DELETE_COLS.len()。
pub const NCOLS_OUT: usize = 71;

/// 保留列索引（NCOLS 中不在 DELETE_COLS 的）。
pub fn keep_cols() -> Vec<usize> {
    let del: std::collections::HashSet<usize> = DELETE_COLS.iter().copied().collect();
    (0..NCOLS).filter(|i| !del.contains(i)).collect()
}

// ============================================================================
// 小段定义
// ============================================================================

/// 一个时间小段：用快照索引范围 [lo, hi)（含 lo 不含 hi）和时间范围 [t0, t1]。
#[derive(Clone, Debug)]
struct Segment {
    /// 段内首快照索引（含）
    mkt_lo: usize,
    /// 段内末快照索引（不含）
    mkt_hi: usize,
    /// 时间范围起点（epoch 秒，含）
    t0: f32,
    /// 时间范围终点（epoch 秒，含）
    t1: f32,
}

// ============================================================================
// 辅助统计函数（序列聚合用）
// ============================================================================

/// 对序列做 5 种聚合：mean, std, skew, autocorr(lag), trend。
/// 返回长度 5 的数组，点数不足时对应位置为 NaN。
/// - mean: n>=1
/// - std: n>=2（样本标准差 ddof=1，与 pandas 一致）
/// - skew: n>=3
/// - autocorr: n>=lag+1
/// - trend: n>=2（与 [1..n] 的 Pearson 相关）
#[inline]
fn aggregate5(vals: &[f32], autocorr_lag: usize) -> [f32; 5] {
    let n = vals.len();
    let mut out = [f32::NAN; 5];
    if n == 0 {
        return out;
    }
    // mean
    let sum: f32 = vals.iter().copied().sum();
    let mean = sum / n as f32;
    out[0] = mean;
    // std (ddof=1)
    if n >= 2 {
        let mut sq = 0.0f32;
        for &v in vals {
            let d = v - mean;
            sq += d * d;
        }
        out[1] = (sq / (n - 1) as f32).sqrt();
    }
    // skew (pandas: 偏度 G1，无偏校正)
    if n >= 3 {
        let std1 = out[1];
        if std1 > 0.0 && std1.is_finite() {
            let m2: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
            let m3: f32 = vals.iter().map(|&v| (v - mean).powi(3)).sum::<f32>() / n as f32;
            let g1 = m3 / m2.powf(1.5);
            // 无偏校正（pandas scipy skew 校正因子）
            let nf = n as f32;
            let g1_adj = g1 * ((nf - 1.0).powf(1.5)) / ((nf - 2.0) * nf.sqrt());
            out[2] = if g1_adj.is_finite() { g1_adj } else { f32::NAN };
        }
    }
    // autocorr(lag)
    if n > autocorr_lag && autocorr_lag >= 1 {
        let a = &vals[..n - autocorr_lag];
        let b = &vals[autocorr_lag..];
        let na = a.len();
        let ma = a.iter().sum::<f32>() / na as f32;
        let mb = b.iter().sum::<f32>() / na as f32;
        let mut cov = 0.0f32;
        let mut va = 0.0f32;
        let mut vb = 0.0f32;
        for k in 0..na {
            let da = a[k] - ma;
            let db = b[k] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        let denom = (va * vb).sqrt();
        out[3] = if denom > 0.0 { cov / denom } else { f32::NAN };
    }
    // trend: Pearson(序列, [1..n])
    if n >= 2 {
        let xs: Vec<f32> = (1..=n).map(|x| x as f32).collect();
        let mx = xs.iter().sum::<f32>() / n as f32;
        let mut cov = 0.0f32;
        let mut vx = 0.0f32;
        let mut vy = 0.0f32;
        for k in 0..n {
            let dx = xs[k] - mx;
            let dy = vals[k] - mean;
            cov += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        let denom = (vx * vy).sqrt();
        out[4] = if denom > 0.0 { cov / denom } else { f32::NAN };
    }
    out
}

/// 线性相关（Pearson），长度不足返回 NaN。
#[inline]
fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 {
        return f32::NAN;
    }
    let ma = a[..n].iter().sum::<f32>() / n as f32;
    let mb = b[..n].iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for k in 0..n {
        let da = a[k] - ma;
        let db = b[k] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        f32::NAN
    }
}

/// 对方阵 data (n×d) 计算协方差矩阵的行列式。
/// data 按行存储：每行是一个 d 维样本。
/// 样本数 n <= d 时行列式无法计算，返回 NaN。
#[inline]
fn cov_det(data: &[f32], n: usize, d: usize) -> f32 {
    if n < d || d == 0 || n == 0 {
        return f32::NAN;
    }
    // 列均值
    let mut col_mean = vec![0.0f32; d];
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for j in 0..d {
            col_mean[j] += row[j];
        }
    }
    for j in 0..d {
        col_mean[j] /= n as f32;
    }
    // 协方差矩阵 (ddof=0，与归一化协方差约定一致)
    let mut cov = vec![0.0f32; d * d];
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for j in 0..d {
            let dj = row[j] - col_mean[j];
            for k in j..d {
                let dk = row[k] - col_mean[k];
                cov[j * d + k] += dj * dk;
            }
        }
    }
    for j in 0..d {
        for k in 0..d {
            if k < j {
                cov[j * d + k] = cov[k * d + j];
            } else {
                cov[j * d + k] /= n as f32;
            }
        }
    }
    // LU 分解求行列式
    lu_det(&cov, d)
}

/// LU 分解（不带主元，对正定协方差矩阵足够；非正定则返回 NaN）。
#[inline]
fn lu_det(a: &[f32], n: usize) -> f32 {
    let mut m = a.to_vec();
    let mut det = 1.0f32;
    for k in 0..n {
        let pivot = m[k * n + k];
        if pivot.abs() < 1e-30 || !pivot.is_finite() {
            return 0.0;
        }
        det *= pivot;
        for i in (k + 1)..n {
            let factor = m[i * n + k] / pivot;
            for j in k..n {
                m[i * n + j] -= factor * m[k * n + j];
            }
        }
    }
    det
}

/// 归一化协方差行列式：每列 z-score（ddof=0）后求协方差行列式。
/// data 按行存储 n×d。
#[inline]
fn norm_cov_det(data: &[f32], n: usize, d: usize) -> f32 {
    if n < d || d == 0 || n == 0 {
        return f32::NAN;
    }
    // 列均值
    let mut col_mean = vec![0.0f32; d];
    let mut col_std = vec![0.0f32; d];
    for j in 0..d {
        let mut s = 0.0f32;
        for i in 0..n {
            s += data[i * d + j];
        }
        col_mean[j] = s / n as f32;
    }
    for j in 0..d {
        let mut s = 0.0f32;
        let mj = col_mean[j];
        for i in 0..n {
            let d_ij = data[i * d + j] - mj;
            s += d_ij * d_ij;
        }
        col_std[j] = (s / n as f32).sqrt();
    }
    // z-score（std=0 时该列置 0，避免除零）
    let mut z = vec![0.0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let std_j = col_std[j];
            z[i * d + j] = if std_j > 0.0 {
                (data[i * d + j] - col_mean[j]) / std_j
            } else {
                0.0
            };
        }
    }
    // 归一化后的协方差 = 相关矩阵，用 lu_det
    cov_det(&z, n, d)
}

// ============================================================================
// 预计算：全天快照序列上的盘口指标
// ============================================================================

/// 盘口失衡：(bid_sum - ask_sum) / (bid_sum + ask_sum)，按 4 种档位口径。
/// 返回 (n, 4) 列序：[1档, 1-5档, 1-10档, 6-10档]
fn precompute_imbalance(market: &[MarketRecord]) -> Vec<[f32; 4]> {
    market
        .iter()
        .map(|m| {
            // 1 档
            let a1 = m.ask_vols[0];
            let b1 = m.bid_vols[0];
            let imb1 = if a1 + b1 > 0.0 {
                (b1 - a1) / (a1 + b1)
            } else {
                f32::NAN
            };
            // 1-5 档
            let a5: f32 = m.ask_vols[..5].iter().sum::<f32>();
            let b5: f32 = m.bid_vols[..5].iter().sum::<f32>();
            let imb5 = if a5 + b5 > 0.0 {
                (b5 - a5) / (a5 + b5)
            } else {
                f32::NAN
            };
            // 1-10 档
            let a10: f32 = m.ask_vols.iter().sum::<f32>();
            let b10: f32 = m.bid_vols.iter().sum::<f32>();
            let imb10 = if a10 + b10 > 0.0 {
                (b10 - a10) / (a10 + b10)
            } else {
                f32::NAN
            };
            // 6-10 档
            let a610: f32 = m.ask_vols[5..].iter().sum::<f32>();
            let b610: f32 = m.bid_vols[5..].iter().sum::<f32>();
            let imb610 = if a610 + b610 > 0.0 {
                (b610 - a610) / (a610 + b610)
            } else {
                f32::NAN
            };
            [imb1, imb5, imb10, imb610]
        })
        .collect()
}

/// 10 档挂单 trend：每快照内 10 档量与 [1..10] 的 corr。
/// same_side=true 时用该侧 10 档，false 时用对手侧。
/// 返回长度 n 的序列。
fn precompute_vol_trend(market: &[MarketRecord], same_side: Side) -> Vec<f32> {
    let xs: [f32; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    market
        .iter()
        .map(|m| {
            let vols = match same_side {
                Side::Bid => &m.bid_vols,
                Side::Ask => &m.ask_vols,
            };
            corr(&xs, vols)
        })
        .collect()
}

/// 10 档挂单 std：每快照内 10 档量的 std（ddof=1）。
fn precompute_vol_std(market: &[MarketRecord], same_side: Side) -> Vec<f32> {
    market
        .iter()
        .map(|m| {
            let vols = match same_side {
                Side::Bid => &m.bid_vols,
                Side::Ask => &m.ask_vols,
            };
            let n = 10usize;
            let mean = vols.iter().sum::<f32>() / n as f32;
            let mut sq = 0.0f32;
            for &v in vols {
                let d = v - mean;
                sq += d * d;
            }
            (sq / (n - 1) as f32).sqrt()
        })
        .collect()
}

/// 滚动 rolling_n 窗内 同侧 10 档挂单量的协方差行列式（含 |det|）。
/// 窗口=过去 rolling_n 次快照（含当前），前 rolling_n-1 个为 NaN。
/// 返回 (n, 2)：[det, |det|]
fn precompute_roll10_det(
    market: &[MarketRecord],
    same_side: Side,
    rolling_n: usize,
) -> Vec<[f32; 2]> {
    let n = market.len();
    let mut out = vec![[f32::NAN; 2]; n];
    if rolling_n == 0 || n < rolling_n {
        return out;
    }
    // 预组织 (rolling_n, 10) 数据
    let mut buf = vec![0.0f32; rolling_n * 10];
    for end in rolling_n..=n {
        let start = end - rolling_n;
        for (i, idx) in (start..end).enumerate() {
            let vols = match same_side {
                Side::Bid => &market[idx].bid_vols,
                Side::Ask => &market[idx].ask_vols,
            };
            buf[i * 10..(i + 1) * 10].copy_from_slice(vols);
        }
        let det = cov_det(&buf, rolling_n, 10);
        let row = end - 1;
        out[row] = [det, det.abs()];
    }
    out
}

/// 滚动 rolling_n 窗内 双方 20 档挂单量（ask1-10 ⊕ bid1-10）协方差行列式（含 |det|）。
/// 返回 (n, 2)：[det, |det|]
fn precompute_roll20_det(market: &[MarketRecord], rolling_n: usize) -> Vec<[f32; 2]> {
    let n = market.len();
    let mut out = vec![[f32::NAN; 2]; n];
    if rolling_n == 0 || n < rolling_n {
        return out;
    }
    let mut buf = vec![0.0f32; rolling_n * 20];
    for end in rolling_n..=n {
        let start = end - rolling_n;
        for (i, idx) in (start..end).enumerate() {
            let m = &market[idx];
            buf[i * 20..(i + 1) * 20][..10].copy_from_slice(&m.ask_vols);
            buf[i * 20..(i + 1) * 20][10..].copy_from_slice(&m.bid_vols);
        }
        let det = cov_det(&buf, rolling_n, 20);
        let row = end - 1;
        out[row] = [det, det.abs()];
    }
    out
}

/// 价格波动率：std(last_prc[i]/last_prc[i-1] - 1)。
/// 这是标量，对每段算一次。
fn price_volatility(market: &[MarketRecord], lo: usize, hi: usize) -> f32 {
    if hi <= lo + 1 {
        return f32::NAN;
    }
    let mut rets = Vec::with_capacity(hi - lo - 1);
    for i in lo + 1..hi {
        let prev = market[i - 1].last_prc;
        let curr = market[i].last_prc;
        if prev > 0.0 {
            rets.push(curr / prev - 1.0);
        }
    }
    if rets.len() < 2 {
        return f32::NAN;
    }
    let n = rets.len();
    let mean = rets.iter().sum::<f32>() / n as f32;
    let sq: f32 = rets.iter().map(|&v| (v - mean).powi(2)).sum();
    (sq / (n - 1) as f32).sqrt()
}

/// 间隔成交量：volume[i] - volume[i-1]，返回 [lo, hi) 区间内的序列。
fn interval_volumes(market: &[MarketRecord], lo: usize, hi: usize) -> Vec<f32> {
    if hi <= lo + 1 {
        return vec![];
    }
    (lo + 1..hi)
        .map(|i| market[i].volume - market[i - 1].volume)
        .collect()
}

// ============================================================================
// 逐笔成交预计算
// ============================================================================

/// 按 1 秒聚合小段内的成交，返回 (秒时间戳, 主买量, 主卖量, 笔数, 金额)。
/// 主买 flag=66，主卖 flag=83，撤单已在读取时过滤。
/// trade 已按时间排序。
fn aggregate_by_second(
    trade: &[TradeRecord],
    t0: f32,
    t1: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // 返回 (secs, buy_vols, sell_vols, counts, amounts, buy_amounts)
    // secs: 该秒的时间戳（向下取整到整数秒）
    // buy_vols/sell_vols: 该秒主买/主卖成交量
    // counts: 该秒成交笔数（主买+主卖）
    // amounts: 该秒总成交金额
    // buy_amounts: 该秒主买成交金额
    if trade.is_empty() {
        return (vec![], vec![], vec![], vec![], vec![], vec![]);
    }
    // 二分定位 [t0, t1] 范围
    let lo = trade.partition_point(|t| t.time_sec < t0);
    let mut hi = trade.partition_point(|t| t.time_sec <= t1);
    if hi == 0 || lo >= trade.len() || lo >= hi {
        return (vec![], vec![], vec![], vec![], vec![], vec![]);
    }
    hi = hi.min(trade.len());

    // 按秒聚合
    let mut secs: Vec<f32> = Vec::new();
    let mut buy_vols: Vec<f32> = Vec::new();
    let mut sell_vols: Vec<f32> = Vec::new();
    let mut counts: Vec<f32> = Vec::new();
    let mut amounts: Vec<f32> = Vec::new();
    let mut buy_amounts: Vec<f32> = Vec::new();

    let mut cur_sec = f32::NAN;
    for i in lo..hi {
        let t = &trade[i];
        let sec = t.time_sec.floor();
        if sec != cur_sec {
            secs.push(sec);
            buy_vols.push(0.0);
            sell_vols.push(0.0);
            counts.push(0.0);
            amounts.push(0.0);
            buy_amounts.push(0.0);
            cur_sec = sec;
        }
        let idx = secs.len() - 1;
        if t.flag == 66 {
            buy_vols[idx] += t.volume;
            buy_amounts[idx] += t.turnover;
        } else if t.flag == 83 {
            sell_vols[idx] += t.volume;
        }
        counts[idx] += 1.0;
        amounts[idx] += t.turnover;
    }
    (secs, buy_vols, sell_vols, counts, amounts, buy_amounts)
}

// ============================================================================
// 小段内全部指标计算
// ============================================================================

/// 方向枚举：bid 流程同侧=bid，ask 流程同侧=ask。
#[derive(Clone, Copy)]
enum Side {
    Bid,
    Ask,
}

/// 计算一个小段内的全部 147 列指标。
/// 预计算的序列（imbalance, vol_trend, vol_std, roll10_det, roll20_det）传入以避免重复。
fn compute_segment_features(
    seg: &Segment,
    market: &[MarketRecord],
    trade: &[TradeRecord],
    side: Side,
    p: &ObsOrderParams,
    imb: &[[f32; 4]],      // 全天盘口失衡
    vt_same: &[f32],       // 全天 同侧 vol trend
    vt_opp: &[f32],        // 全天 对手侧 vol trend
    vs_same: &[f32],       // 全天 同侧 vol std
    vs_opp: &[f32],        // 全天 对手侧 vol std
    r10_same: &[[f32; 2]], // 全天 同侧 rolling 10 det
    r10_opp: &[[f32; 2]],  // 全天 对手侧 rolling 10 det
    r20: &[[f32; 2]],      // 全天 双方 rolling 20 det
) -> Vec<f32> {
    let lo = seg.mkt_lo;
    let hi = seg.mkt_hi;
    let mut out = Vec::with_capacity(NCOLS);

    // ============ 盘口序列类（×5聚合）============
    // P1 盘口失衡 4 档 ×5 = 20
    for dim in 0..4 {
        let col: Vec<f32> = imb[lo..hi].iter().map(|x| x[dim]).collect();
        let agg = aggregate5(&col, p.autocorr_lag);
        out.extend_from_slice(&agg);
    }
    // P2 10档trend 同侧/对手 ×5 = 10
    for src in [vt_same, vt_opp] {
        let agg = aggregate5(&src[lo..hi], p.autocorr_lag);
        out.extend_from_slice(&agg);
    }
    // P3 10档std 同侧/对手 ×5 = 10
    for src in [vs_same, vs_opp] {
        let agg = aggregate5(&src[lo..hi], p.autocorr_lag);
        out.extend_from_slice(&agg);
    }
    // P7 rolling10 同侧 det/|det| ×5 = 10
    for v in [0, 1] {
        let col: Vec<f32> = r10_same[lo..hi].iter().map(|x| x[v]).collect();
        let agg = aggregate5(&col, p.autocorr_lag);
        out.extend_from_slice(&agg);
    }
    // P8 rolling20 双方 det/|det| ×5 = 10
    for v in [0, 1] {
        let col: Vec<f32> = r20[lo..hi].iter().map(|x| x[v]).collect();
        let agg = aggregate5(&col, p.autocorr_lag);
        out.extend_from_slice(&agg);
    }
    // 小计 60

    // ============ 盘口标量类（每段1值）============
    // P4 价格波动率 = 1
    out.push(price_volatility(market, lo, hi));
    // P5 间隔成交量 std = 1
    {
        let iv = interval_volumes(market, lo, hi);
        if iv.len() >= 2 {
            let n = iv.len();
            let mean = iv.iter().sum::<f32>() / n as f32;
            let sq: f32 = iv.iter().map(|&v| (v - mean).powi(2)).sum();
            out.push((sq / (n - 1) as f32).sqrt());
        } else {
            out.push(f32::NAN);
        }
    }
    // P6 间隔成交量 trend = 1
    {
        let iv = interval_volumes(market, lo, hi);
        let xs: Vec<f32> = (1..=iv.len()).map(|x| x as f32).collect();
        out.push(corr(&xs, &iv));
    }
    // S1 小段10档协方差行列式 8 版本
    {
        // 同侧 det / |det|
        let d_same = segment_vol_cov_det(market, lo, hi, side, 10);
        let d_opp = segment_vol_cov_det(market, lo, hi, opp_side(side), 10);
        out.push(d_same); // 1
        out.push(d_same.abs()); // 2
        out.push(d_opp); // 3
        out.push(d_opp.abs()); // 4
        out.push(d_same - d_opp); // 5 det 差
        out.push((d_same - d_opp).abs()); // 6 |det 差|
        out.push(d_same.abs() - d_opp.abs()); // 7 |det| 差
        out.push((d_same.abs() - d_opp.abs()).abs()); // 8 ||det| 差|
    }
    // S2 小段双方20档协方差行列式 det / |det|
    {
        let d = segment_vol_cov_both_det(market, lo, hi);
        out.push(d);
        out.push(d.abs());
    }
    // 小计 13

    // ============ 逐笔序列类（×5聚合）============
    // 先聚合成交到秒
    let (secs, buy_vols, sell_vols, counts, amounts, buy_amounts) =
        aggregate_by_second(trade, seg.t0, seg.t1);

    // 该小段逐笔成交（用于订单编号差值）
    let t_lo = trade.partition_point(|t| t.time_sec < seg.t0);
    let t_hi = trade.partition_point(|t| t.time_sec <= seg.t1);

    // T1 订单编号差值 ask_order - bid_order 原始 / |·| ×5 = 10
    {
        let n_trades = t_hi.saturating_sub(t_lo);
        let mut diff_raw = Vec::with_capacity(n_trades.min(3_000_000));
        let mut diff_abs = Vec::with_capacity(n_trades.min(3_000_000));
        for i in t_lo..t_hi {
            let d = trade[i].ask_order as f32 - trade[i].bid_order as f32;
            diff_raw.push(d);
            diff_abs.push(d.abs());
        }
        out.extend_from_slice(&aggregate5(&diff_raw, p.autocorr_lag));
        out.extend_from_slice(&aggregate5(&diff_abs, p.autocorr_lag));
    }
    // T2 按秒主买占比 ×5 = 5
    {
        let ratio: Vec<f32> = (0..secs.len())
            .map(|i| {
                let denom = buy_vols[i] + sell_vols[i];
                if denom > 0.0 {
                    buy_vols[i] / denom
                } else {
                    f32::NAN
                }
            })
            .collect();
        out.extend_from_slice(&aggregate5(&ratio, p.autocorr_lag));
    }
    // T3 按秒成交笔数 买/卖/差/|差|/全量 ×5 = 25
    {
        // 买笔数：buy_vols>0 的秒算（这里用 buy_amounts>0 或 buy_vols>0 判断有主买）
        // 但按秒聚合时 counts 已是全量笔数。买/卖笔数需重新统计。
        let mut buy_counts = vec![0.0f32; secs.len()];
        let mut sell_counts = vec![0.0f32; secs.len()];
        let mut cur_sec = f32::NAN;
        let mut cur_idx = 0usize;
        for i in t_lo..t_hi {
            let t = &trade[i];
            let sec = t.time_sec.floor();
            if sec != cur_sec {
                // 找到对应 idx
                while cur_idx < secs.len() && secs[cur_idx] < sec {
                    cur_idx += 1;
                }
                if cur_idx < secs.len() && secs[cur_idx] == sec {
                    cur_sec = sec;
                } else {
                    continue;
                }
            }
            if t.flag == 66 {
                buy_counts[cur_idx] += 1.0;
            } else if t.flag == 83 {
                sell_counts[cur_idx] += 1.0;
            }
        }
        // 差/|差|
        let diff: Vec<f32> = (0..secs.len())
            .map(|i| buy_counts[i] - sell_counts[i])
            .collect();
        let diff_abs: Vec<f32> = diff.iter().map(|&v| v.abs()).collect();
        for src in [
            &buy_counts[..],
            &sell_counts[..],
            &diff[..],
            &diff_abs[..],
            &counts[..],
        ] {
            out.extend_from_slice(&aggregate5(src, p.autocorr_lag));
        }
    }
    // T4 按秒每笔平均成交金额 买/卖/差/|差|/全量 ×5 = 25
    {
        // 买均额 = buy_amounts / buy_counts；卖均额需 sell_amounts/sell_counts
        let mut sell_amounts = vec![0.0f32; secs.len()];
        let mut cur_sec = f32::NAN;
        let mut cur_idx = 0usize;
        for i in t_lo..t_hi {
            let t = &trade[i];
            let sec = t.time_sec.floor();
            if sec != cur_sec {
                while cur_idx < secs.len() && secs[cur_idx] < sec {
                    cur_idx += 1;
                }
                if cur_idx < secs.len() && secs[cur_idx] == sec {
                    cur_sec = sec;
                } else {
                    continue;
                }
            }
            if t.flag == 83 {
                sell_amounts[cur_idx] += t.turnover;
            }
        }
        // buy_counts/sell_counts 重新算（避免上一块局部变量）
        let mut bc2 = vec![0.0f32; secs.len()];
        let mut sc2 = vec![0.0f32; secs.len()];
        let mut cur_sec = f32::NAN;
        let mut cur_idx = 0usize;
        for i in t_lo..t_hi {
            let t = &trade[i];
            let sec = t.time_sec.floor();
            if sec != cur_sec {
                while cur_idx < secs.len() && secs[cur_idx] < sec {
                    cur_idx += 1;
                }
                if cur_idx < secs.len() && secs[cur_idx] == sec {
                    cur_sec = sec;
                } else {
                    continue;
                }
            }
            if t.flag == 66 {
                bc2[cur_idx] += 1.0;
            } else if t.flag == 83 {
                sc2[cur_idx] += 1.0;
            }
        }
        let buy_avg: Vec<f32> = (0..secs.len())
            .map(|i| {
                if bc2[i] > 0.0 {
                    buy_amounts[i] / bc2[i]
                } else {
                    f32::NAN
                }
            })
            .collect();
        let sell_avg: Vec<f32> = (0..secs.len())
            .map(|i| {
                if sc2[i] > 0.0 {
                    sell_amounts[i] / sc2[i]
                } else {
                    f32::NAN
                }
            })
            .collect();
        let diff: Vec<f32> = (0..secs.len()).map(|i| buy_avg[i] - sell_avg[i]).collect();
        let diff_abs: Vec<f32> = diff.iter().map(|&v| v.abs()).collect();
        let all_avg: Vec<f32> = (0..secs.len())
            .map(|i| {
                if counts[i] > 0.0 {
                    amounts[i] / counts[i]
                } else {
                    f32::NAN
                }
            })
            .collect();
        for src in [
            &buy_avg[..],
            &sell_avg[..],
            &diff[..],
            &diff_abs[..],
            &all_avg[..],
        ] {
            out.extend_from_slice(&aggregate5(src, p.autocorr_lag));
        }
    }
    // 小计 65

    // ============ 逐笔标量类（每段1值）============
    // S3 成交[时间戳,量,价]归一化协方差行列式 det / |det|
    // S4 主买 det / |det|
    // S5 主卖 det / |det|
    // S6 主买det - 主卖det / |差|
    // S7 主买总占比
    {
        // 全量
        let d_all = norm_cov_det_trade(trade, t_lo, t_hi, None);
        out.push(d_all);
        out.push(d_all.abs());
        // 主买
        let d_buy = norm_cov_det_trade(trade, t_lo, t_hi, Some(true));
        out.push(d_buy);
        out.push(d_buy.abs());
        // 主卖
        let d_sell = norm_cov_det_trade(trade, t_lo, t_hi, Some(false));
        out.push(d_sell);
        out.push(d_sell.abs());
        // 差
        out.push(d_buy - d_sell);
        out.push((d_buy - d_sell).abs());
        // 主买总占比
        let total_buy: f32 = (t_lo..t_hi)
            .filter(|&i| trade[i].flag == 66)
            .map(|i| trade[i].volume)
            .sum();
        let total_all: f32 = (t_lo..t_hi).map(|i| trade[i].volume).sum();
        out.push(if total_all > 0.0 {
            total_buy / total_all
        } else {
            f32::NAN
        });
    }
    // 小计 9

    debug_assert_eq!(out.len(), NCOLS);
    out
}

/// 小段内单侧 N 档挂单量的协方差行列式。
/// 把段内每个快照的同侧（或对手侧）N 档量作为一行（n×N），算协方差行列式。
fn segment_vol_cov_det(
    market: &[MarketRecord],
    lo: usize,
    hi: usize,
    side: Side,
    n_tiers: usize,
) -> f32 {
    let n = hi - lo;
    if n < n_tiers || n == 0 {
        return f32::NAN;
    }
    let mut buf = vec![0.0f32; n * n_tiers];
    for (i, idx) in (lo..hi).enumerate() {
        let vols = match side {
            Side::Bid => &market[idx].bid_vols,
            Side::Ask => &market[idx].ask_vols,
        };
        buf[i * n_tiers..(i + 1) * n_tiers].copy_from_slice(&vols[..n_tiers]);
    }
    cov_det(&buf, n, n_tiers)
}

/// 小段内双方 2*N 档（ask 前 N + bid 前 N）挂单量的协方差行列式。
fn segment_vol_cov_both_det(market: &[MarketRecord], lo: usize, hi: usize) -> f32 {
    let n = hi - lo;
    let d = 20;
    if n < d || n == 0 {
        return f32::NAN;
    }
    let mut buf = vec![0.0f32; n * d];
    for (i, idx) in (lo..hi).enumerate() {
        let m = &market[idx];
        buf[i * d..(i + 1) * d][..10].copy_from_slice(&m.ask_vols);
        buf[i * d..(i + 1) * d][10..].copy_from_slice(&m.bid_vols);
    }
    cov_det(&buf, n, d)
}

/// 成交记录归一化协方差行列式。
/// flag_filter = None(全量) / Some(true)(仅主买) / Some(false)(仅主卖)
/// 列序固定 [时间戳(float秒), 成交量, 价格]
fn norm_cov_det_trade(
    trade: &[TradeRecord],
    lo: usize,
    hi: usize,
    flag_filter: Option<bool>,
) -> f32 {
    let d = 3;
    if hi <= lo {
        return f32::NAN;
    }
    // 限制预分配上限，避免极端成交量的股票导致 capacity overflow / OOM
    let cap = ((hi - lo) * d).min(3_000_000);
    let mut buf = Vec::with_capacity(cap);
    for i in lo..hi {
        let t = &trade[i];
        let keep = match flag_filter {
            None => true,
            Some(true) => t.flag == 66,
            Some(false) => t.flag == 83,
        };
        if keep {
            buf.push(t.time_sec);
            buf.push(t.volume);
            buf.push(t.price);
        }
    }
    let n = buf.len() / d;
    norm_cov_det(&buf, n, d)
}

#[inline]
fn opp_side(side: Side) -> Side {
    match side {
        Side::Bid => Side::Ask,
        Side::Ask => Side::Bid,
    }
}

// ============================================================================
// 小段划分
// ============================================================================

/// 计算分位点（与 numpy.quantile 的 linear 插值一致）。
fn quantile(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// 方法A：连续低于阈值的快照合并成段。
/// 段时间范围 = 段首快照 → 段末快照的下一次快照时刻。
/// 段内快照数 < drop_min 丢弃。
fn method_a_segments(
    market: &[MarketRecord],
    obs_ratio: &[f32],
    threshold: f32,
    drop_min: usize,
) -> Vec<Segment> {
    let n = market.len();
    let mut segs = Vec::new();
    let mut i = 0;
    while i < n {
        if obs_ratio[i] < threshold {
            let start = i;
            while i < n && obs_ratio[i] < threshold {
                i += 1;
            }
            let end = i; // [start, end) 都低于阈值
            let count = end - start;
            if count >= drop_min {
                let t0 = market[start].time_sec;
                // 段末快照 = end-1；下一次快照时刻 = market[end]（若存在）
                let t1 = if end < n {
                    market[end].time_sec
                } else {
                    // 最后一段：用最后一次快照的时间
                    market[end - 1].time_sec
                };
                segs.push(Segment {
                    mkt_lo: start,
                    mkt_hi: end,
                    t0,
                    t1,
                });
            }
        } else {
            i += 1;
        }
    }
    segs
}

/// 方法B：穿越点（前一次 ≥ 阈值、本次 < 阈值），每点取其后 forward_sec 秒作为一段。
fn method_b_segments(
    market: &[MarketRecord],
    obs_ratio: &[f32],
    threshold: f32,
    forward_sec: f32,
) -> Vec<Segment> {
    let n = market.len();
    let mut segs = Vec::new();
    for i in 1..n {
        if obs_ratio[i - 1] >= threshold && obs_ratio[i] < threshold {
            let t0 = market[i].time_sec;
            let t1 = t0 + forward_sec;
            // 找段内快照范围 [mkt_lo, mkt_hi)
            let mkt_lo = i;
            let mut mkt_hi = i;
            while mkt_hi < n && market[mkt_hi].time_sec <= t1 {
                mkt_hi += 1;
            }
            segs.push(Segment {
                mkt_lo,
                mkt_hi,
                t0,
                t1,
            });
        }
    }
    segs
}

// ============================================================================
// 主计算入口
// ============================================================================

/// 计算可观测挂单比例序列并划分小段，对每套配置计算 (seg_2d, pre5_2d)。
///
/// 返回 dict: { (side_str, method_str): (seg_2d_flat, seg_rows, pre5_2d_flat, pre5_rows), ... }
/// seg_2d_flat 为 row-major 展平的 (rows × NCOLS) f64。
pub fn compute_observable_order_metrics(
    code: &str,
    date: i64,
    params: &ObsOrderParams,
) -> std::io::Result<
    [(Vec<f32>, usize, Vec<f32>, usize); 4], // bid_A, bid_B, ask_A, ask_B
> {
    // 1. 读取数据（撤单已在读取时过滤）
    let trade = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    let market = read_market_fast_inner(code, date, false, true, usize::MAX)?;

    // 2. 预计算全天盘口序列
    let imb = precompute_imbalance(&market);
    // 可观测挂单比例
    let obs_bid: Vec<f32> = market
        .iter()
        .map(|m| {
            let obs: f32 = m.bid_vols.iter().sum();
            if m.total_bid_vol > 0.0 {
                obs / m.total_bid_vol
            } else {
                f32::NAN
            }
        })
        .collect();
    let obs_ask: Vec<f32> = market
        .iter()
        .map(|m| {
            let obs: f32 = m.ask_vols.iter().sum();
            if m.total_ask_vol > 0.0 {
                obs / m.total_ask_vol
            } else {
                f32::NAN
            }
        })
        .collect();

    // 3. 分位点（对非 NaN 值排序后取分位）
    let thr_bid = {
        let mut s: Vec<f32> = obs_bid.iter().copied().filter(|v| v.is_finite()).collect();
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        quantile(&s, params.q)
    };
    let thr_ask = {
        let mut s: Vec<f32> = obs_ask.iter().copied().filter(|v| v.is_finite()).collect();
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        quantile(&s, params.q)
    };

    // 4. 预计算 rolling 序列（同侧/对手侧，对 bid 和 ask 流程共用底层 bid/ask 数据）
    let vt_bid = precompute_vol_trend(&market, Side::Bid);
    let vt_ask = precompute_vol_trend(&market, Side::Ask);
    let vs_bid = precompute_vol_std(&market, Side::Bid);
    let vs_ask = precompute_vol_std(&market, Side::Ask);
    let r10_bid = precompute_roll10_det(&market, Side::Bid, params.rolling_n);
    let r10_ask = precompute_roll10_det(&market, Side::Ask, params.rolling_n);
    let r20 = precompute_roll20_det(&market, params.rolling_n);

    // 5. 对 4 套配置各算一遍
    let pre5_sec = params.pre_minutes * 60.0;
    let configs: [(Side, &str); 4] = [
        (Side::Bid, "A"),
        (Side::Bid, "B"),
        (Side::Ask, "A"),
        (Side::Ask, "B"),
    ];
    let mut results: [(Vec<f32>, usize, Vec<f32>, usize); 4] = [
        (vec![], 0, vec![], 0),
        (vec![], 0, vec![], 0),
        (vec![], 0, vec![], 0),
        (vec![], 0, vec![], 0),
    ];

    for (ci, &(side, method)) in configs.iter().enumerate() {
        let (obs, thr) = match side {
            Side::Bid => (&obs_bid, thr_bid),
            Side::Ask => (&obs_ask, thr_ask),
        };
        let segs = match method {
            "A" => method_a_segments(&market, obs, thr, params.drop_min),
            _ => method_b_segments(&market, obs, thr, params.forward_sec),
        };
        let nrows = segs.len();
        if nrows == 0 {
            results[ci] = (vec![], 0, vec![], 0);
            continue;
        }
        let (vt_same, vt_opp, vs_same, vs_opp, r10_same, r10_opp) = match side {
            Side::Bid => (&vt_bid, &vt_ask, &vs_bid, &vs_ask, &r10_bid, &r10_ask),
            Side::Ask => (&vt_ask, &vt_bid, &vs_ask, &vs_bid, &r10_ask, &r10_bid),
        };
        // seg_2d
        let mut seg_flat = Vec::with_capacity(nrows * NCOLS);
        let mut pre5_flat = Vec::with_capacity(nrows * NCOLS);
        for seg in &segs {
            let row = compute_segment_features(
                seg, &market, &trade, side, params, &imb, vt_same, vt_opp, vs_same, vs_opp,
                r10_same, r10_opp, &r20,
            );
            seg_flat.extend_from_slice(&row);
            // pre5：起点前 pre_minutes 分钟
            let pre5_t0 = seg.t0 - pre5_sec;
            let pre5_t1 = seg.t0;
            // 在快照中找 [pre5_t0, pre5_t1] 的范围
            let pre_lo = market.partition_point(|m| m.time_sec < pre5_t0);
            let mut pre_hi = market.partition_point(|m| m.time_sec <= pre5_t1);
            if pre_hi > pre_lo {
                pre_hi = pre_hi.min(market.len());
                let pre_seg = Segment {
                    mkt_lo: pre_lo,
                    mkt_hi: pre_hi,
                    t0: pre5_t0,
                    t1: pre5_t1,
                };
                let pre_row = compute_segment_features(
                    &pre_seg, &market, &trade, side, params, &imb, vt_same, vt_opp, vs_same,
                    vs_opp, r10_same, r10_opp, &r20,
                );
                pre5_flat.extend_from_slice(&pre_row);
            } else {
                // pre5 窗口不足，填 NaN
                pre5_flat.extend_from_slice(&[f32::NAN; NCOLS]);
            }
        }
        results[ci] = (seg_flat, nrows, pre5_flat, nrows);
    }

    Ok(results)
}
// ============================================================================
// PyO3 接口
// ============================================================================

use pyo3::prelude::*;

/// Python 可调用：py_compute_observable_order_metrics(code, date, params_dict=None)
///
/// 返回 dict:
///   {
///     "bid_A": {"seg_flat": list[f64], "seg_rows": int,
///              "pre5_flat": list[f64], "pre5_rows": int, "ncols": int},
///     ...
///   }
/// seg_flat 为 row-major 展平的 (seg_rows × NCOLS)，Python 端 reshape 即可。
#[pyfunction]
#[pyo3(signature = (code, date, params=None))]
pub fn py_compute_observable_order_metrics(
    py: Python<'_>,
    code: &str,
    date: i64,
    params: Option<&pyo3::types::PyAny>,
) -> PyResult<PyObject> {
    let p = match params {
        Some(d) => parse_params(d)?,
        None => ObsOrderParams::default(),
    };
    let results = compute_observable_order_metrics(code, date, &p)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;

    let keys = ["bid_A", "bid_B", "ask_A", "ask_B"];
    let out = pyo3::types::PyDict::new(py);
    let keep = keep_cols();
    for (i, key) in keys.iter().enumerate() {
        let (seg_flat, seg_rows, pre5_flat, pre5_rows) = &results[i];
        // 过滤 DELETE_COLS，只保留 keep 中索引对应的列
        let seg_filt = filter_cols(seg_flat, *seg_rows, &keep);
        let pre5_filt = filter_cols(pre5_flat, *pre5_rows, &keep);
        let d = pyo3::types::PyDict::new(py);
        d.set_item("seg_flat", seg_filt)?;
        d.set_item("seg_rows", *seg_rows)?;
        d.set_item("pre5_flat", pre5_filt)?;
        d.set_item("pre5_rows", *pre5_rows)?;
        d.set_item("ncols", NCOLS_OUT)?;
        out.set_item(key, d)?;
    }
    Ok(out.to_object(py))
}

fn parse_params(d: &pyo3::types::PyAny) -> PyResult<ObsOrderParams> {
    let mut p = ObsOrderParams::default();
    if let Ok(v) = d.get_item("q").and_then(|v| v.extract::<f64>().map(|x| x as f32)) {
        p.q = v;
    }
    if let Ok(v) = d.get_item("drop_min").and_then(|v| v.extract::<usize>()) {
        p.drop_min = v;
    }
    if let Ok(v) = d.get_item("forward_sec").and_then(|v| v.extract::<f64>().map(|x| x as f32)) {
        p.forward_sec = v;
    }
    if let Ok(v) = d.get_item("pre_minutes").and_then(|v| v.extract::<f64>().map(|x| x as f32)) {
        p.pre_minutes = v;
    }
    if let Ok(v) = d.get_item("rolling_n").and_then(|v| v.extract::<usize>()) {
        p.rolling_n = v;
    }
    if let Ok(v) = d
        .get_item("autocorr_lag")
        .and_then(|v| v.extract::<usize>())
    {
        p.autocorr_lag = v;
    }
    Ok(p)
}
/// 把 (n_rows × NCOLS) 的展平数据，按 keep 索引过滤为 (n_rows × keep.len())。
fn filter_cols(data: &[f32], n_rows: usize, keep: &[usize]) -> Vec<f32> {
    if n_rows == 0 {
        return vec![];
    }
    let mut out = Vec::with_capacity(n_rows * keep.len());
    for r in 0..n_rows {
        for &c in keep {
            out.push(data[r * NCOLS + c]);
        }
    }
    out
}
