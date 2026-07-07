//! 极端点拟合因子（Extreme Point Fit）。
//!
//! 核心思想：从逐笔成交或盘口快照序列中，按某种标准挑出 N 个最极端的点，
//! 把这些点上的目标量（价格 / 主买占比 / 挂单失衡）用直线、二次曲线或
//! 分段线性插值拟合回原始长度，然后计算拟合曲线与原始曲线的相关性或偏差。
//!
//! ## 输出布局
//!
//! 26 行（N = 5, 6, ..., 30）× 192 列 = 4992 个因子值，row-major 展平。
///
/// ### 逐笔部分（列 0..143，共 144 列）
///
/// 3 目标 × 8 挑点标准 × 3 拟合 × 2 比较 = 144
///
/// | 维度 | 取值 |
/// |------|------|
/// | 目标 | P=价格, B=滚动主买占比, I=挂单失衡(最近盘口) |
/// | 标准 | V=单笔量, L=局部放量, D_b/s/c=买卖单号差, A_b/s/c=单边主动量 |
/// | 拟合 | lin=直线, bra=二次曲线, int=分段插值 |
/// | 比较 | cor=相关性, dev=偏差 |
///
/// ### 盘口部分（列 144..191，共 48 列）
///
/// 2 目标 × 4 挑点标准 × 3 拟合 × 2 比较 = 48
///
/// | 维度 | 取值 |
/// |------|------|
/// | 目标 | P=最新价, I=挂单失衡 |
/// | 标准 | Z=快照间隔量, D_b/s/c=区间买卖单号差 |
///
/// ## 金融含义
///
/// - **相关性 cor**：极端点拟合线与原线的 Pearson 相关。值越高说明极端点越能
///   代表整体趋势；值越低说明极端点偏离趋势，蕴含结构性信息。
/// - **偏差 dev**：极端点拟合线与原线的平均绝对偏差。值越大说明极端点
///   偏离整体越远，市场在这些极端时刻的信息含量越高。
/// - **买卖方向拆分**：D_b 挑买方最迫切的时刻 → 拟合出「看多者心中的曲线」；
///   D_s 挑卖方最迫切的时刻 → 「看空者心中的曲线」。
use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use pyo3::prelude::*;

// ============================================================================
// 常量
// ============================================================================

/// 取点数序列：5, 6, 7, ..., 30（共 26 个）。
pub const N_POINTS: &[usize] = &[
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30,
];

/// 行数 = N_POINTS 长度。
pub const N_ROWS: usize = 26;

/// 逐笔列数。
pub const N_TRADE_COLS: usize = 144;

/// 盘口列数。
pub const N_MARKET_COLS: usize = 48;

/// 总列数。
pub const N_COLS: usize = N_TRADE_COLS + N_MARKET_COLS; // 192

/// 输出总长度（原始展平）。
pub const OUT_LEN: usize = N_ROWS * N_COLS; // 4992

/// get_features_factors 降维后的因子数。
/// 19 个单列统计量 × 192 列 + C(192,2) 配对相关性 = 3648 + 18336 = 21984
pub const FEAT_LEN: usize = 19 * N_COLS + N_COLS * (N_COLS - 1) / 2;

/// 局部放量窗口半径（前后各 3 秒）。
const LOCAL_VOL_HALF_WINDOW: f32 = 3.0;

/// 滚动主买占比窗口（秒）。
const BUY_RATIO_WINDOW_SEC: f32 = 60.0;

/// 拟合方法数。
const N_FITS: usize = 3; // lin, bra, int

/// 比较方法数。
const N_CMPS: usize = 2; // cor, dev

// ============================================================================
// 因子名生成（与输出列序严格对齐，单一真相源）
// ============================================================================

/// 生成 192 个列名（不含行 N 前缀），用于 get_features_factors 降维时的 col_names。
/// 列序与 compute_extreme_fit_full 的 192 列严格对齐。
pub fn extreme_fit_col_names() -> Vec<String> {
    let trade_targets = ["P", "B", "I"];
    let trade_crits: &[(&str, &str)] = &[
        ("V", "c"),
        ("L", "c"),
        ("D", "b"),
        ("D", "s"),
        ("D", "c"),
        ("A", "b"),
        ("A", "s"),
        ("A", "c"),
    ];
    let market_targets = ["P", "I"];
    let market_crits: &[(&str, &str)] = &[("Z", "c"), ("D", "b"), ("D", "s"), ("D", "c")];
    let fits = ["lin", "bra", "int"];
    let cmps = ["cor", "dev"];

    let mut cols = Vec::with_capacity(N_COLS);
    // 逐笔：3 目标 × 8 标准 × 3 拟合 × 2 比较 = 144
    for tgt in trade_targets {
        for &(cr, dir) in trade_crits {
            for fit in fits {
                for cmp in cmps {
                    cols.push(format!("T-{tgt}-{cr}-{dir}-{fit}-{cmp}"));
                }
            }
        }
    }
    // 盘口：2 目标 × 4 标准 × 3 拟合 × 2 比较 = 48
    for tgt in market_targets {
        for &(cr, dir) in market_crits {
            for fit in fits {
                for cmp in cmps {
                    cols.push(format!("S-{tgt}-{cr}-{dir}-{fit}-{cmp}"));
                }
            }
        }
    }
    assert_eq!(cols.len(), N_COLS);
    cols
}

/// 生成全部 4992 个原始展平因子名（N5..N30 × 192 列）。
pub fn extreme_fit_names() -> Vec<String> {
    let cols = extreme_fit_col_names();
    let mut names = Vec::with_capacity(OUT_LEN);
    for &n in N_POINTS {
        for c in &cols {
            names.push(format!("N{n}_{c}"));
        }
    }
    assert_eq!(names.len(), OUT_LEN);
    names
}

/// 生成降维部分的 21984 个因子名（与 get_features_factors_rust_full 输出对齐）。
/// 19 个单列统计量 × 192 列 + C(192,2) 配对相关性。
pub fn extreme_fit_feat_names() -> Vec<String> {
    let cols = extreme_fit_col_names();
    let mut names = Vec::with_capacity(FEAT_LEN);
    // 19 个算子（与 features.rs push_group 顺序严格一致）
    let ops = [
        "mean",
        "median",
        "std",
        "skew",
        "kurt",
        "p5",
        "p25",
        "p75",
        "p95",
        "iqr",
        "cv",
        "autocorr1",
        "autocorr1_abs",
        "trend",
        "period_diff",
        "period_ratio",
        "lz_complexity",
        "entropy_1d",
        "max_range_product",
    ];
    for op in ops {
        for c in &cols {
            names.push(format!("{c}_{op}"));
        }
    }
    // C(192,2) 配对相关性
    let n = cols.len();
    for i in 0..n {
        for j in (i + 1)..n {
            names.push(format!("{}_corr_{}", cols[i], cols[j]));
        }
    }
    assert_eq!(names.len(), FEAT_LEN);
    names
}

/// 生成全部因子名：原始展平 4992 + 降维 21984 = 26976（pipeline 输出的单一真相源）。
pub fn extreme_fit_full_names() -> Vec<String> {
    let mut names = extreme_fit_names();
    names.extend(extreme_fit_feat_names());
    assert_eq!(names.len(), OUT_LEN + FEAT_LEN);
    names
}

// ============================================================================
// 拟合函数（写入预分配 buffer，避免堆分配）
// ============================================================================

/// 直线拟合系数 (a, b)，y = a*t + b。
#[inline]
fn linear_coeffs(pts: &[(f32, f32)]) -> (f32, f32) {
    let n = pts.len() as f64;
    let sum_t: f64 = pts.iter().map(|(t, _)| *t as f64).sum();
    let sum_y: f64 = pts.iter().map(|(_, y)| *y as f64).sum();
    let sum_tt: f64 = pts.iter().map(|(t, _)| (*t as f64) * (*t as f64)).sum();
    let sum_ty: f64 = pts.iter().map(|(t, y)| (*t as f64) * (*y as f64)).sum();
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-30 {
        (0.0, (sum_y / n) as f32) // 所有点时间相同 → 常数拟合
    } else {
        let a = (n * sum_ty - sum_t * sum_y) / denom;
        let b = (sum_y - a * sum_t) / n;
        (a as f32, b as f32)
    }
}

/// 直线 + 二次曲线拟合，一次遍历同时写入两个 buffer。
/// 二次点不足退化为直线（bra buffer = lin buffer）。
fn fit_lin_bra_into(pts: &[(f32, f32)], eval_t: &[f32], out_lin: &mut [f32], out_bra: &mut [f32]) {
    let (la, lb) = linear_coeffs(pts);
    let bra = quadratic_coeffs(pts);
    for (i, &t) in eval_t.iter().enumerate() {
        let lin_val = la * t + lb;
        out_lin[i] = lin_val;
        out_bra[i] = match bra {
            Some((a, b, c)) => a * t * t + b * t + c,
            None => lin_val,
        };
    }
}

/// 二次曲线拟合系数 (a, b, c)，y = a*t^2 + b*t + c。点不足或奇异返回 None。
fn quadratic_coeffs(pts: &[(f32, f32)]) -> Option<(f32, f32, f32)> {
    if pts.len() < 3 {
        return None;
    }
    let mut s = [0.0f64; 5];
    let mut sy = [0.0f64; 3];
    for &(t, y) in pts {
        let (t, y) = (t as f64, y as f64);
        let mut tk = 1.0;
        for k in 0..5 {
            s[k] += tk;
            tk *= t;
        }
        tk = 1.0;
        for k in 0..3 {
            sy[k] += tk * y;
            tk *= t;
        }
    }
    let m = [[s[4], s[3], s[2]], [s[3], s[2], s[1]], [s[2], s[1], s[0]]];
    let rhs = [sy[2], sy[1], sy[0]];
    let det = det3(&m);
    if det.abs() < 1e-30 {
        return None;
    }
    let a = det3(&[
        [rhs[0], m[0][1], m[0][2]],
        [rhs[1], m[1][1], m[1][2]],
        [rhs[2], m[2][1], m[2][2]],
    ]) / det;
    let b = det3(&[
        [m[0][0], rhs[0], m[0][2]],
        [m[1][0], rhs[1], m[1][2]],
        [m[2][0], rhs[2], m[2][2]],
    ]) / det;
    let c = det3(&[
        [m[0][0], m[0][1], rhs[0]],
        [m[1][0], m[1][1], rhs[1]],
        [m[2][0], m[2][1], rhs[2]],
    ]) / det;
    Some((a as f32, b as f32, c as f32))
}

/// 3×3 行列式。
#[inline]
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// 分段线性插值，写入 `out`。两端 clamp，eval_t 必须升序。
fn fit_interp_into(pts: &[(f32, f32)], eval_t: &[f32], out: &mut [f32]) {
    let mut sorted: Vec<(f32, f32)> = pts
        .iter()
        .filter(|(t, y)| t.is_finite() && y.is_finite())
        .copied()
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let np = sorted.len();
    if np == 0 {
        out.fill(f32::NAN);
        return;
    }
    if np == 1 {
        out.fill(sorted[0].1);
        return;
    }
    let mut seg = 0usize;
    for (i, &t) in eval_t.iter().enumerate() {
        if t <= sorted[0].0 {
            out[i] = sorted[0].1;
        } else if t >= sorted[np - 1].0 {
            out[i] = sorted[np - 1].1;
        } else {
            while seg < np - 2 && sorted[seg + 1].0 <= t {
                seg += 1;
            }
            let (t0, y0) = sorted[seg];
            let (t1, y1) = sorted[seg + 1];
            let dt = t1 - t0;
            out[i] = if dt > 0.0 {
                y0 + (t - t0) / dt * (y1 - y0)
            } else {
                y0
            };
        }
    }
}

// ============================================================================
// 比较函数（单次遍历同时算 cor + dev）
// ============================================================================

/// target 序列的有效值统计量（组级预计算，26 个 N 复用）。
struct TargetStats {
    sum: f64,
    sqsum: f64,
    count: u32,
}

fn target_stats(target: &[f32]) -> TargetStats {
    let mut sum = 0f64;
    let mut sqsum = 0f64;
    let mut count = 0u32;
    for &b in target {
        if b.is_finite() {
            let bf = b as f64;
            sum += bf;
            sqsum += bf * bf;
            count += 1;
        }
    }
    TargetStats { sum, sqsum, count }
}

/// 一次遍历同时算三种拟合的 (cor, dev)。
/// 三种拟合共用同一个 target，合并为单次遍历避免 3 倍 target 扫描。
/// 只检查 target 的 NaN（lin/bra 拟合保证有限；int 在极端 np=0 时可能全 NaN，
/// NaN 参与累加后该拟合的 cor/dev 自然为 NaN，不影响 lin/bra）。
fn compare_three_pass(
    lin: &[f32],
    bra: &[f32],
    int: &[f32],
    target: &[f32],
    ts: &TargetStats,
) -> [(f32, f32); 3] {
    let mut sa = [0f64; 3];
    let mut saa = [0f64; 3];
    let mut sab = [0f64; 3];
    let mut abs_diff = [0f64; 3];
    let mut cnt = 0u32;
    for i in 0..target.len() {
        let b = target[i];
        if b.is_finite() {
            let bf = b as f64;
            let triple = [lin[i], bra[i], int[i]];
            for j in 0..3 {
                let af = triple[j] as f64;
                sa[j] += af;
                saa[j] += af * af;
                sab[j] += af * bf;
                abs_diff[j] += (af - bf).abs();
            }
            cnt += 1;
        }
    }
    let mut result = [(f32::NAN, f32::NAN); 3];
    if cnt >= 3 {
        let nf = cnt as f64;
        for j in 0..3 {
            let cov = sab[j] - sa[j] * ts.sum / nf;
            let vara = saa[j] - sa[j] * sa[j] / nf;
            let varb = ts.sqsum - ts.sum * ts.sum / nf;
            let denom = (vara * varb).sqrt();
            let cor = if denom < 1e-30 {
                f32::NAN
            } else {
                (cov / denom) as f32
            };
            result[j] = (cor, (abs_diff[j] / nf) as f32);
        }
    }
    result
}

// ============================================================================
// Top-K 选取（带去相邻）
// ============================================================================

/// 预排序：把 (score, norm_time, idx) 按 score 降序排好，供贪心去邻复用。
/// 同时要求 score 和 target 都是有限值（NaN 候选直接剔除）。
fn presort_desc(scores: &[f32], target: &[f32], times: &[f32]) -> Vec<(f32, f32, usize)> {
    let mut pairs: Vec<(f32, f32, usize)> = scores
        .iter()
        .zip(target.iter())
        .zip(times.iter())
        .enumerate()
        .filter(|(_, ((s, y), _))| s.is_finite() && y.is_finite())
        .map(|(i, ((s, _), &t))| (*s, t, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

/// 贪心选取 K 个点：按 score 降序遍历，跳过与已选点归一化时间距离 < min_gap 的。
///
/// 金融含义：避免极端点在时间上聚集（例如几笔天量大单挤在同一秒、或买卖单号差
/// 极值集中在同一价位的不同笔成交）。若不去邻，拟合曲线会被局部簇主导，
/// 且当簇内点同价位时会令拟合方差为零、相关性退化成 NaN。
///
/// 举例：取成交量最大的 5 个点，若第 2、3 大紧密相邻，则舍弃第 3 大，
/// 改为取第 1、2、4、5、6 大。候选不足或去邻后剩余 < K 时返回实际数量。
fn greedy_select_dedup(
    sorted: &[(f32, f32, usize)], // (score, norm_time, idx) 按 score 降序
    k: usize,
    min_gap: f32,
) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut sel_times: Vec<f32> = Vec::with_capacity(k);
    for &(_, t, idx) in sorted {
        if selected.len() >= k {
            break;
        }
        let mut ok = true;
        for &st in &sel_times {
            if (t - st).abs() < min_gap {
                ok = false;
                break;
            }
        }
        if ok {
            selected.push(idx);
            sel_times.push(t);
        }
    }
    selected
}

// ============================================================================
// 数据预计算：目标序列
// ============================================================================

/// 滚动窗口主买占比。
///
/// 对每笔成交，看过去 `window_sec` 秒内的成交：
/// 主买量(flag=66) / (主买量 + 主卖量(flag=83))。
/// 窗口内无成交 → NaN。
fn compute_rolling_buy_ratio(trade: &[TradeRecord], window_sec: f32) -> Vec<f32> {
    let n = trade.len();
    let mut out = vec![f32::NAN; n];
    let mut buy_vol = 0.0f64;
    let mut tot_vol = 0.0f64;
    let mut left = 0usize;
    for right in 0..n {
        let vol = trade[right].volume as f64;
        match trade[right].flag {
            66 => {
                buy_vol += vol;
                tot_vol += vol;
            }
            83 => tot_vol += vol,
            _ => {}
        }
        while left < right && trade[right].time_sec - trade[left].time_sec > window_sec {
            let vol_l = trade[left].volume as f64;
            match trade[left].flag {
                66 => {
                    buy_vol -= vol_l;
                    tot_vol -= vol_l;
                }
                83 => tot_vol -= vol_l,
                _ => {}
            }
            left += 1;
        }
        if tot_vol > 0.0 {
            out[right] = (buy_vol / tot_vol) as f32;
        }
    }
    out
}

/// 逐笔的挂单失衡（从最近的前一个盘口快照继承）。
///
/// 失衡 = (Σbid_vol - Σask_vol) / (Σbid_vol + Σask_vol)，用 10 档量计算。
fn compute_trade_imbalance(trade: &[TradeRecord], market: &[MarketRecord]) -> Vec<f32> {
    let n = trade.len();
    let mut out = vec![f32::NAN; n];
    if market.is_empty() {
        return out;
    }
    // 预算盘口失衡
    let mkt_imb: Vec<f32> = market
        .iter()
        .map(|m| {
            let bid: f32 = m.bid_vols.iter().sum();
            let ask: f32 = m.ask_vols.iter().sum();
            let d = bid + ask;
            if d > 0.0 {
                (bid - ask) / d
            } else {
                f32::NAN
            }
        })
        .collect();
    // 对每笔成交，二分查找最近的前一个快照
    for i in 0..n {
        let t = trade[i].time_sec;
        let pos = market.partition_point(|m| m.time_sec <= t);
        if pos > 0 {
            out[i] = mkt_imb[pos - 1];
        }
    }
    out
}

/// 盘口的挂单失衡。
fn compute_market_imbalance(market: &[MarketRecord]) -> Vec<f32> {
    market
        .iter()
        .map(|m| {
            let bid: f32 = m.bid_vols.iter().sum();
            let ask: f32 = m.ask_vols.iter().sum();
            let d = bid + ask;
            if d > 0.0 {
                (bid - ask) / d
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// 局部放量：每笔成交前后 `half_win` 秒内的总成交量。
fn compute_local_volume(trade: &[TradeRecord], half_win: f32) -> Vec<f32> {
    let n = trade.len();
    let mut out = vec![0.0f32; n];
    let mut left = 0usize;
    let mut right = 0usize;
    let mut vol_sum = 0.0f64;
    for i in 0..n {
        let tc = trade[i].time_sec;
        let lo = tc - half_win;
        let hi = tc + half_win;
        while right < n && trade[right].time_sec <= hi {
            vol_sum += trade[right].volume as f64;
            right += 1;
        }
        while left < right && trade[left].time_sec < lo {
            vol_sum -= trade[left].volume as f64;
            left += 1;
        }
        out[i] = vol_sum as f32;
    }
    out
}

/// 盘口快照间隔成交量（累积量差分）。
fn compute_interval_volume(market: &[MarketRecord]) -> Vec<f32> {
    let n = market.len();
    let mut out = vec![0.0f32; n];
    for i in 1..n {
        let dv = market[i].volume - market[i - 1].volume;
        out[i] = if dv > 0.0 { dv } else { 0.0 };
    }
    out
}

/// 盘口每个快照区间内的买卖单号差（从该区间内的逐笔成交聚合）。
///
/// 返回 (D_b, D_s, D_c)，分别为区间内 max(ask_order-bid_order),
/// max(bid_order-ask_order), max(|ask_order-bid_order|)。
/// 区间内无成交 → NaN。
fn compute_snapshot_order_diff(
    trade: &[TradeRecord],
    market: &[MarketRecord],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let nm = market.len();
    let mut d_b = vec![f32::NAN; nm];
    let mut d_s = vec![f32::NAN; nm];
    let mut d_c = vec![f32::NAN; nm];
    if nm < 2 || trade.is_empty() {
        return (d_b, d_s, d_c);
    }
    let mut ti = 0usize;
    for i in 1..nm {
        let t0 = market[i - 1].time_sec;
        let t1 = market[i].time_sec;
        while ti < trade.len() && trade[ti].time_sec < t0 {
            ti += 1;
        }
        let mut mb = f32::MIN;
        let mut ms = f32::MIN;
        let mut mc = f32::MIN;
        let mut tj = ti;
        while tj < trade.len() && trade[tj].time_sec < t1 {
            let diff = (trade[tj].ask_order - trade[tj].bid_order) as f32;
            mb = mb.max(diff);
            ms = ms.max(-diff);
            mc = mc.max(diff.abs());
            tj += 1;
        }
        if mb > f32::MIN {
            d_b[i] = mb;
            d_s[i] = ms;
            d_c[i] = mc;
        }
    }
    (d_b, d_s, d_c)
}

// ============================================================================
// 列组计算（一个目标序列 × 一个挑点标准 → 6 列 × 26 行）
// ============================================================================

/// 对一组（目标序列, 挑点标准）计算全部 N×拟合×比较 的因子值。
///
/// `tn` = 归一化时间戳；`target` = 目标值序列；`scores` = 挑点标准序列。
/// 每个 N 做贪心去邻：两候选点归一化时间距离 < 0.5/N 时舍弃较弱者，
/// 避免极端点在时间上聚集导致拟合被局部簇主导。
/// 结果写入 `out` 的 `[row * N_COLS + col_base .. ]` 区域。
fn compute_column_group(
    tn: &[f32],
    target: &[f32],
    scores: &[f32],
    col_base: usize,
    out: &mut [f32],
) {
    // 预排序一次（按 score 降序），后续每个 K 复用
    let sorted = presort_desc(scores, target, tn);
    if sorted.len() < 2 {
        return;
    }

    // 预分配 buffer（组级复用，26 个 N 共享，避免 26×3=78 次堆分配）
    let n = tn.len();
    let mut buf_lin = vec![0f32; n];
    let mut buf_bra = vec![0f32; n];
    let mut buf_int = vec![0f32; n];
    // 预计算 target 有效统计量（26 个 N 的 cor 计算复用）
    let ts = target_stats(target);

    for (n_idx, &n_pts) in N_POINTS.iter().enumerate() {
        // 最小归一化时间间距 = 理想均匀间距的一半。
        // K 个点理想均匀分布在 [0,1] 上间距 1/(K-1)，取一半作为去邻阈值：
        // 既允许两个次大的点适度靠近（不至于误杀），又能打散秒级聚集的簇。
        let min_gap = 0.5 / n_pts as f32;
        let selected = greedy_select_dedup(&sorted, n_pts, min_gap);
        if selected.len() < 2 {
            continue;
        }
        let pts: Vec<(f32, f32)> = selected.iter().map(|&idx| (tn[idx], target[idx])).collect();

        // 拟合写入预分配 buffer：lin+bra 合并遍历 + interp 单独遍历
        fit_lin_bra_into(&pts, tn, &mut buf_lin, &mut buf_bra);
        fit_interp_into(&pts, tn, &mut buf_int);

        // 一次遍历同时算三种拟合的 cor + dev
        let results = compare_three_pass(&buf_lin, &buf_bra, &buf_int, target, &ts);
        for (fit_idx, (cor, dev)) in results.iter().enumerate() {
            let col = col_base + fit_idx * N_CMPS;
            out[n_idx * N_COLS + col] = *cor;
            out[n_idx * N_COLS + col + 1] = *dev;
        }
    }
}

// ============================================================================
// 主计算入口
// ============================================================================

/// 极端点拟合因子的核心计算（pipeline 和 Python 入口的唯一共同调用点）。
///
/// 输出长度恒为 `OUT_LEN`（4992），布局 26 行 × 192 列 row-major。
pub fn compute_extreme_fit_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let trade = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    let market = read_market_fast_inner(code, date, false, true, usize::MAX)?;

    let mut out = vec![f32::NAN; OUT_LEN];

    // ===== 逐笔部分（列 0..143）=====
    if trade.len() >= 2 {
        let n = trade.len();

        // 归一化时间戳到 [0, 1]
        let t_min = trade.first().unwrap().time_sec;
        let t_max = trade.last().unwrap().time_sec;
        let t_range = (t_max - t_min).max(1e-6);
        let tn: Vec<f32> = trade
            .iter()
            .map(|t| (t.time_sec - t_min) / t_range)
            .collect();

        // 目标序列
        let tgt_p: Vec<f32> = trade.iter().map(|t| t.price).collect();
        let tgt_b = compute_rolling_buy_ratio(&trade, BUY_RATIO_WINDOW_SEC);
        let tgt_i = compute_trade_imbalance(&trade, &market);

        // 挑点标准
        let crit_v: Vec<f32> = trade.iter().map(|t| t.volume).collect();
        let crit_l = compute_local_volume(&trade, LOCAL_VOL_HALF_WINDOW);
        let odiff: Vec<f32> = trade
            .iter()
            .map(|t| (t.ask_order - t.bid_order) as f32)
            .collect();
        let crit_db: Vec<f32> = odiff.clone();
        let crit_ds: Vec<f32> = odiff.iter().map(|d| -d).collect();
        let crit_dc: Vec<f32> = odiff.iter().map(|d| d.abs()).collect();
        let crit_ab: Vec<f32> = trade
            .iter()
            .map(|t| if t.flag == 66 { t.volume } else { f32::NAN })
            .collect();
        let crit_as: Vec<f32> = trade
            .iter()
            .map(|t| if t.flag == 83 { t.volume } else { f32::NAN })
            .collect();
        let crit_ac = crit_v.clone(); // A_c = V_c（全量主动量 = 单笔量）

        // 列序：tgt(3) × crit(8) × fit(3) × cmp(2)
        // 每个 crit 占 6 列（3 fits × 2 cmps）
        const PER_CRIT: usize = N_FITS * N_CMPS; // 6
        const PER_TGT: usize = 8 * PER_CRIT; // 48

        let targets = [&tgt_p, &tgt_b, &tgt_i];
        let criteria = [
            &crit_v[..],
            &crit_l[..],
            &crit_db[..],
            &crit_ds[..],
            &crit_dc[..],
            &crit_ab[..],
            &crit_as[..],
            &crit_ac[..],
        ];

        for (ti, target) in targets.iter().enumerate() {
            for (ci, crit) in criteria.iter().enumerate() {
                let col_base = ti * PER_TGT + ci * PER_CRIT;
                compute_column_group(&tn, target, crit, col_base, &mut out);
            }
        }
    }

    // ===== 盘口部分（列 144..191）=====
    if market.len() >= 2 {
        let n = market.len();

        // 归一化时间戳
        let t_min = market.first().unwrap().time_sec;
        let t_max = market.last().unwrap().time_sec;
        let t_range = (t_max - t_min).max(1e-6);
        let tn: Vec<f32> = market
            .iter()
            .map(|m| (m.time_sec - t_min) / t_range)
            .collect();

        // 目标序列
        let tgt_p: Vec<f32> = market.iter().map(|m| m.last_prc).collect();
        let tgt_i = compute_market_imbalance(&market);

        // 挑点标准
        let crit_z = compute_interval_volume(&market);
        let (crit_db, crit_ds, crit_dc) = compute_snapshot_order_diff(&trade, &market);

        // 列序：tgt(2) × crit(4) × fit(3) × cmp(2)
        const PER_CRIT: usize = N_FITS * N_CMPS; // 6
        const PER_TGT: usize = 4 * PER_CRIT; // 24

        let targets = [&tgt_p, &tgt_i];
        let criteria = [&crit_z[..], &crit_db[..], &crit_ds[..], &crit_dc[..]];

        for (ti, target) in targets.iter().enumerate() {
            for (ci, crit) in criteria.iter().enumerate() {
                let col_base = N_TRADE_COLS + ti * PER_TGT + ci * PER_CRIT;
                compute_column_group(&tn, target, crit, col_base, &mut out);
            }
        }
    }

    Ok(out)
}

// ============================================================================
// 完整输出（原始展平 + get_features_factors 降维）—— 双入口共同真相源
// ============================================================================

/// 拼接原始展平 4992 + get_features_factors 降维 21984 = 26976。
/// pipeline 和 py_ 入口的唯一共同调用点，保证两入口逐字节一致。
pub fn build_full_output(raw: &[f32]) -> Vec<f32> {
    let mut all = raw.to_vec();
    // 重塑为 26×192 的 Array2，过 get_features_factors_rust_full 降维
    let arr = ndarray::Array2::from_shape_vec((N_ROWS, N_COLS), raw.to_vec())
        .unwrap_or_else(|_| ndarray::Array2::zeros((0, N_COLS)));
    let col_names = extreme_fit_col_names();
    let (feat_vals, _) = if arr.nrows() == 0 {
        (vec![f32::NAN; FEAT_LEN], vec![])
    } else {
        crate::features::get_features_factors_rust_full(&arr.view(), &col_names, false)
    };
    all.extend_from_slice(&feat_vals);
    all
}

/// 完整核心：读数据 → 算 4992 → 降维拼接到 26976。py_ 和 pipeline 共用。
pub fn compute_extreme_fit_with_features(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let raw = compute_extreme_fit_full(code, date)?;
    Ok(build_full_output(&raw))
}

// ============================================================================
// PyO3 接口
// ============================================================================

/// Python 可调用：单股单日计算，返回完整 26976 个因子值（原始展平 + 降维）。
/// 与 pipeline 走同一份 build_full_output，输出逐字节相同。
#[pyfunction]
pub fn py_extreme_point_fit(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_extreme_fit_with_features(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

/// Python 拿完整因子名（原始展平 4992 + 降维 21984 = 26976）。
#[pyfunction]
pub fn py_extreme_point_fit_names() -> Vec<String> {
    extreme_fit_full_names()
}
