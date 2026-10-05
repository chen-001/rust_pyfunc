//! hm54 创纪录熵变 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧代码要点（逐条对齐，见 成果hm54_创纪录熵变/hm54_okay.py）：
//!
//!   参数：get_many=0 / read_full=0 / change_time=1 / cross_section=False
//!        get_many=0 且 read_full=0 时 read_trade_data_single 直接返回整张表，
//!        旧代码自己再切窗口，**不做** adjust_afternoon 的下午平移，
//!        所以 Rust 侧用 with_afternoon_adjust=false 读。
//!
//!   撤单过滤：read_trade_data_single(with_retreat=0) 丢掉 flag==32 的行。
//!        旧框架按 pandas 数值比较（2025 年文件里写的是 "32.0"，一样过滤得掉），
//!        而 Rust 侧 with_retreat 那条判断是按字节比 "32"，对 "32.0" 不生效。
//!        所以这里用 read_trade_fast_inner(code, date, true, false, MAX) 全读进来，
//!        再自己 .filter(|r| r.flag != 32)。（flag 解析器对 "32.0" 也给出 32。）
//!
//!   窗口：df.set_index("exchtime").between_time("09:30", "14:57")，
//!        两端都含、精确到纳秒。exchtime 是「1970-01-01 08:00 + exchtime_us/1e6」，
//!        所以盘中判定就是 time_us 的当日微秒落在 [09:30:00, 14:57:00] 闭区间。
//!        （原始 exchtime 是整数微秒，纳秒精度等价于微秒精度比较。）
//!
//!   12 个基准列 = 2 方向 × 2 侧 × 3 窗口 {3,30,300}：
//!        up_*   = rp.calculate_shannon_entropy_change(...)        价格创新高时算熵变
//!        down_* = rp.calculate_shannon_entropy_change_at_low(...) 价格创新低时算熵变
//!        两个函数把「基准分布」（开盘到当前时刻按委托号累计成交量）与
//!        「窗口分布」（基准 + 事件后 W 秒内的成交）各算一次香农熵，输出差。
//!        非事件点留 NaN。
//!
//!   逐值复刻的两个细节（照 src/sequence/mod.rs 抄，不照 etp_event_metrics.rs 抄）：
//!     1. 极值初值是 ±inf，所以第 0 行也算事件；但收尾时 result[0] 被强制设成 NaN。
//!     2. 窗口右端指针 window_end 初值 0、跨事件单调持久、不重置；
//!        条件是 t_sec[window_end+1] <= t_sec[i] + W（t_sec 为整数秒）。
//!        旧函数里 t_sec = (exchtime_ns/1e9) 截断 = time_us/1e6（已核对一致）。
//!     另外旧函数用 HashMap 累加熵，浮点求和顺序随机；本文件用 BTreeMap 按键升序，
//!        两者只差 ~1e-13（实测），远小于对账阈值 1e-4。
//!
//!   统计量（每个基准列 8 个 × 12 列 = 96，再加 18 个 DTW = 114）：
//!      mean / skew / max / min / sum / autocorr1 / autocorr2 / autocorr3
//!      全部按 pandas 语义：跳过 NaN；mean/skew/max/min 全 NaN 给 NaN；
//!      sum 全 NaN 给 0.0；skew 用 Fisher-Pearson 有偏校正（pandas nanskew）；
//!      autocorr(k) = corr(x[i], x[i-k])，先 dropna 再成对完整观测。
//!   18 个 DTW：a/b 两侧、up/down 之间两两组合，输入是 dropna 后的熵变序列。
//!      纯 Rust 实现（worker 链路不能碰 #[pyfunction] 的 crate::time_series::dtw_distance）。
//!
//!   旧文件里没有重名的 names_single，114 个名字唯一，不需要 _dup 后缀。
//!   `_fold`（= |x - 当日截面均值|）由因子库的 write_selected_factor_base 负责，
//!   本文件只输出 114 个原值。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};

const US_PER_SEC: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
/// 09:30:00（当天零点起微秒）
const SESSION_START_US: i64 = 34_200_000_000;
/// 14:57:00（当天零点起微秒）
const SESSION_END_US: i64 = 53_820_000_000;

/// 12 个基准列的名字，顺序 = 旧代码 names 前 96 项的分组顺序。
const COLS: [&str; 12] = [
    "up_ask_etp3",
    "up_bid_etp3",
    "up_ask_etp30",
    "up_bid_etp30",
    "up_ask_etp300",
    "up_bid_etp300",
    "down_ask_etp3",
    "down_bid_etp3",
    "down_ask_etp30",
    "down_bid_etp30",
    "down_ask_etp300",
    "down_bid_etp300",
];

/// 每个基准列的构造参数：(up?, side_ask?, 窗口秒)。与 COLS 一一对应。
const COL_SPEC: [(bool, bool, i64); 12] = [
    (true, true, 3),
    (true, false, 3),
    (true, true, 30),
    (true, false, 30),
    (true, true, 300),
    (true, false, 300),
    (false, true, 3),
    (false, false, 3),
    (false, true, 30),
    (false, false, 30),
    (false, true, 300),
    (false, false, 300),
];

/// 5 个基本统计量的后缀顺序（旧 names 里 mean/skew/max/min/sum 依次排列）。
const STATS: [&str; 5] = ["mean", "skew", "max", "min", "sum"];

/// 18 个 DTW 的名字（顺序即旧 names 末段顺序）。
const DTW_NAMES: [&str; 18] = [
    "dtw_up_ab_etp3",
    "dtw_up_ab_etp30",
    "dtw_up_ab_etp300",
    "dtw_down_ab_etp3",
    "dtw_down_ab_etp30",
    "dtw_down_ab_etp300",
    "dtw_ud_ask_etp3",
    "dtw_ud_bid_etp3",
    "dtw_ud_ask_etp30",
    "dtw_ud_bid_etp30",
    "dtw_ud_ask_etp300",
    "dtw_ud_bid_etp300",
    "dtw_ud_ab_1_etp3",
    "dtw_ud_ab_1_etp30",
    "dtw_ud_ab_1_etp300",
    "dtw_ud_ab_2_etp3",
    "dtw_ud_ab_2_etp30",
    "dtw_ud_ab_2_etp300",
];

/// 12 列 × 8 个统计量 + 18 个 DTW。
pub const N_OUT: usize = 12 * 8 + 18;

/// DTW 的取列对：下标按 COLS 的顺序数（0..11），与旧代码里的调用一一对应。
const DTW_PAIRS: [(usize, usize); 18] = [
    (0, 1),   // dtw_up_ab_etp3      = up_ask_etp3  vs up_bid_etp3
    (2, 3),   // dtw_up_ab_etp30
    (4, 5),   // dtw_up_ab_etp300
    (6, 7),   // dtw_down_ab_etp3
    (8, 9),   // dtw_down_ab_etp30
    (10, 11), // dtw_down_ab_etp300
    (0, 6),   // dtw_ud_ask_etp3     = up_ask_etp3   vs down_ask_etp3
    (1, 7),   // dtw_ud_bid_etp3     = up_bid_etp3   vs down_bid_etp3
    (2, 8),   // dtw_ud_ask_etp30
    (3, 9),   // dtw_ud_bid_etp30
    (4, 10),  // dtw_ud_ask_etp300
    (5, 11),  // dtw_ud_bid_etp300
    (0, 7),   // dtw_ud_ab_1_etp3    = up_ask_etp3   vs down_bid_etp3
    (2, 9),   // dtw_ud_ab_1_etp30
    (4, 11),  // dtw_ud_ab_1_etp300
    (1, 6),   // dtw_ud_ab_2_etp3    = up_bid_etp3   vs down_ask_etp3
    (3, 8),   // dtw_ud_ab_2_etp30
    (5, 10),  // dtw_ud_ab_2_etp300
];

/// 因子名清单（与 compute_stock 的输出严格逐个对齐）。
pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for s in STATS.iter() {
        for c in COLS.iter() {
            out.push(format!("{}_{}", c, s));
        }
    }
    for lag in 1..=3 {
        for c in COLS.iter() {
            out.push(format!("{}_autocorr{}", c, lag));
        }
    }
    for n in DTW_NAMES.iter() {
        out.push((*n).to_string());
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// pandas 统计量
// ============================================================================

#[inline]
fn finite(v: &[f64]) -> Vec<f64> {
    v.iter().cloned().filter(|x| x.is_finite()).collect()
}

/// Series.mean()：跳过 NaN，全 NaN 返回 NaN。
fn pd_mean(v: &[f64]) -> f64 {
    let f = finite(v);
    if f.is_empty() {
        return f64::NAN;
    }
    let mut s = 0.0f64;
    for &x in f.iter() {
        s += x;
    }
    s / f.len() as f64
}

/// Series.skew()：照 pandas.core.nanops.nanskew 逐行搬（Fisher-Pearson 有偏校正）。
///   result = (n·sqrt(n-1)/(n-2)) · (m3 / m2^1.5)，m2/m3 是**离差平方/立方的和**；
///   |m2|<1e-14 归零、|m3|<1e-14 归零；m2==0 时结果为 0；n<3 一律 NaN（这条最后生效）。
fn pd_skew(v: &[f64]) -> f64 {
    let f = finite(v);
    let n = f.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mut s = 0.0f64;
    for &x in f.iter() {
        s += x;
    }
    let mean = s / nf;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &x in f.iter() {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
    }
    if m2.abs() < 1e-14 {
        m2 = 0.0;
    }
    if m3.abs() < 1e-14 {
        m3 = 0.0;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    (nf * (nf - 1.0).sqrt() / (nf - 2.0)) * (m3 / m2.powf(1.5))
}

/// Series.max()：跳过 NaN，全 NaN 返回 NaN。
fn pd_max(v: &[f64]) -> f64 {
    let f = finite(v);
    if f.is_empty() {
        return f64::NAN;
    }
    let mut m = f[0];
    for &x in f.iter() {
        if x > m {
            m = x;
        }
    }
    m
}

/// Series.min()：跳过 NaN，全 NaN 返回 NaN。
fn pd_min(v: &[f64]) -> f64 {
    let f = finite(v);
    if f.is_empty() {
        return f64::NAN;
    }
    let mut m = f[0];
    for &x in f.iter() {
        if x < m {
            m = x;
        }
    }
    m
}

/// Series.sum()：跳过 NaN，全 NaN 返回 0.0。
fn pd_sum(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v.iter() {
        if x.is_finite() {
            s += x;
        }
    }
    s
}

/// Pearson 相关系数，成对完整观测（输入已无 NaN）。
/// 与 np.corrcoef 同构（先减均值再求协方差）；样本 < 2 或任一侧方差为 0 返回 NaN。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..n {
        sx += x[i];
        sy += y[i];
    }
    let mx = sx / nf;
    let my = sy / nf;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Series.dropna().autocorr(lag)：先去掉 NaN，再 corr(x[i], x[i-lag])。
fn autocorr(v: &[f64], lag: usize) -> f64 {
    let f = finite(v);
    let n = f.len();
    if n <= lag {
        return f64::NAN;
    }
    pearson(&f[lag..], &f[..n - lag])
}

/// rp.dtw_distance(a, b) 的纯 Rust 版（radius=None、不限时那条路径）。
///
/// ！！不能直接调 crate::time_series::dtw_distance：那是 #[pyfunction] 返回 PyResult，
/// worker 二进制不带 libpython，链路上碰到会让 worker 链接失败而构建表面仍「成功」。
/// 算法逐行照 src/time_series/mod.rs 抄：warp[0][0]=0、其余 inf，
/// 代价 |a_i-b_j|，累积取上/左/左上最小值，返回 warp[n][m]。
/// 两边都空返回 0（warp[0][0]=0），只有一边空返回 +inf。
fn dtw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return if n == 0 && m == 0 { 0.0 } else { f64::INFINITY };
    }
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut cur = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = f64::INFINITY;
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let best = prev[j].min(cur[j - 1]).min(prev[j - 1]);
            cur[j] = cost + best;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

// ============================================================================
// 熵变序列（照 src/sequence/mod.rs 的 calculate_shannon_entropy_change / _at_low）
// ============================================================================

/// 香农熵 H = -Σ p·ln(p)，只统计严格大于 0 的桶。
fn entropy(map: &std::collections::BTreeMap<i64, f64>, total: f64) -> f64 {
    let mut e = 0.0f64;
    for &vol in map.values() {
        if vol > 0.0 {
            let p = vol / total;
            e -= p * p.ln();
        }
    }
    e
}

/// 一条熵变序列。t_sec 是整数秒；up=true 用创新高、false 用创新低。
///
/// 与旧函数的对齐点：极值初值 ±inf（第 0 行也算事件）、窗口右端指针跨事件持久、
/// 收尾把 result[0] 强制设成 NaN、只在 window_end > i 且窗口总量 > 0 时写值。
fn etp_series(
    t_sec: &[i64],
    order: &[i64],
    volume: &[f64],
    price: &[f64],
    w: i64,
    up: bool,
) -> Vec<f64> {
    let n = t_sec.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }
    let mut base: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
    let mut cum = 0.0f64;
    let mut window_end = 0usize;
    let mut extreme = if up { f64::NEG_INFINITY } else { f64::INFINITY };

    for i in 0..n {
        *base.entry(order[i]).or_insert(0.0) += volume[i];
        cum += volume[i];

        let p = price[i];
        let is_event = if up { p > extreme } else { p < extreme };
        if !is_event {
            continue;
        }
        extreme = p;

        let base_ent = if cum > 0.0 { entropy(&base, cum) } else { 0.0 };

        let end_time = t_sec[i] + w;
        // 指针跨事件只前进、不重置。
        while window_end + 1 < n && t_sec[window_end + 1] <= end_time {
            window_end += 1;
        }
        if window_end > i {
            let mut win = base.clone();
            let mut win_total = cum;
            for j in (i + 1)..=window_end {
                *win.entry(order[j]).or_insert(0.0) += volume[j];
                win_total += volume[j];
            }
            if win_total > 0.0 {
                result[i] = entropy(&win, win_total) - base_ent;
            }
        }
    }
    if n > 0 {
        result[0] = f64::NAN;
    }
    result
}

// ============================================================================
// 入口
// ============================================================================

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // with_afternoon_adjust=false：旧代码用真实钟点窗口，不做下午平移。
    // with_retreat=true 只是先把行全读进来，撤单过滤下面自己做（理由见文件头）。
    let trades = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;

    let mut t_sec: Vec<i64> = Vec::new();
    let mut ask_order: Vec<i64> = Vec::new();
    let mut bid_order: Vec<i64> = Vec::new();
    let mut volume: Vec<f64> = Vec::new();
    let mut price: Vec<f64> = Vec::new();
    for r in trades.iter() {
        if r.flag == 32 {
            continue;
        }
        let day_us = r.time_us.rem_euclid(DAY_US);
        if day_us < SESSION_START_US || day_us > SESSION_END_US {
            continue;
        }
        t_sec.push(r.time_us / US_PER_SEC);
        ask_order.push(r.ask_order);
        bid_order.push(r.bid_order);
        volume.push(r.volume as f64);
        price.push(r.price as f64);
    }
    // 窗口内一行都没有时旧代码 return None（res 未定义），照做。
    if t_sec.is_empty() {
        return None;
    }

    // 12 个基准列：2 方向 × 2 侧 × 3 窗口
    let mut series: Vec<Vec<f64>> = Vec::with_capacity(12);
    for &(up, side_ask, w) in COL_SPEC.iter() {
        let order = if side_ask { &ask_order } else { &bid_order };
        series.push(etp_series(&t_sec, order, &volume, &price, w, up));
    }

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for s in STATS.iter() {
        for col in series.iter() {
            let v = match *s {
                "mean" => pd_mean(col),
                "skew" => pd_skew(col),
                "max" => pd_max(col),
                "min" => pd_min(col),
                _ => pd_sum(col),
            };
            out.push(v as f32);
        }
    }
    for lag in 1..=3usize {
        for col in series.iter() {
            out.push(autocorr(col, lag) as f32);
        }
    }
    // DTW：输入是 dropna 后的序列
    let dropped: Vec<Vec<f64>> = series.iter().map(|c| finite(c)).collect();
    for &(a, b) in DTW_PAIRS.iter() {
        out.push(dtw(&dropped[a], &dropped[b]) as f32);
    }

    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm54 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
