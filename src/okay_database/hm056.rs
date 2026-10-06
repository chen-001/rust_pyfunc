//! hm056「瞬时协同后的死亡半衰期」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm56_瞬时协同后的死亡半衰期/hm56_okay.py
//!   run_base 的 Go.get_factor 参数 get_many=0 / read_full=0 / change_time=1，逐股系列。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)   # 数值比较丢掉 flag==32
//!    change_time=1 → exchtime 换成 Timestamp（epoch 微秒 → ns，走 pandas 的
//!    cast_from_unit_vectorized，末位有 ns 噪声）。
//! 2) df = df.set_index("exchtime")
//!      df_open15  = between_time("09:30", "09:45")   # 闭区间，ns 精度
//!      df_close15 = between_time("14:45", "14:57")
//!    （没有调 adjust_afternoon，用真实钟点）
//! 3) res = sing(open15,0) + sing(close15,0) + sing(open15,1) + sing(close15,1)
//!    extreme=0 用 find_half_energy_time，extreme=1 用 find_half_extreme_time；
//!    四个 130 项拼成 520，顺序 = names 的四个后缀
//!    ['_open15_full','_close15_full','_open15_extreme','_close15_extreme']。
//!
//! 二、sing(df, extreme) 的 130 个值
//!    · df.shape[0] == 0 或 >= 100000 → 直接返回 [nan]*130。
//!    · 三个能量列（time_window = M1=1 / N1=5 / N2=30）：
//!        energyM1 = rp.find_half_*_time(exchtime_ns_f64, price, 1)   等三列
//!      注意传进去的是 datetime64[ns].to_numpy(dtype=float)，即**纳秒**的 f64；
//!      函数内部再 /1e9 变秒。timeout_seconds 走默认 None，所以没有超时机制。
//!    · 三列里等于 0 的换成 NaN（replace(0, np.nan)）。
//!    · fivetoone = energyN1/energyM1，thirtytofive = energyN2/energyN1。
//!    · follow_price = find_follow_volume_sum_same_price(t, price, volume)
//!      （默认 time_window=0.1、check_price=True、filter_ratio=0）
//!      只保留 round(follow_price - volume, 2) > 0 的行（numpy 的 round：rint(x*100)/100）。
//!    · follow_group = mark_follow_groups(t, price, volume, 0.1)，
//!      drop_duplicates(subset=['follow_group'], keep='first')：每个组只留第一次出现的那行。
//!    · 之后所有统计都在这张去重后的表上做。顺序不能换：
//!      df_66 / df_83 = flag==66 / flag==83；
//!      df_high_price = price >= price.quantile(0.9)；
//!      df_low_price  = price <= price.quantile(0.1)；
//!      df_mid_price  = (price >= q10) & (price <= q90)。
//!    · 130 项的顺序见 METRIC_NAMES。
//!
//! 三、几个必须对齐的 pandas 细节
//! ════════════════════════════════════════════════════════════════════
//! · DataFrame.corr() → libalgos.nancorr（Welford 在线算法、掩码 isfinite、min_periods=1）。
//!   它只算 yi <= xi 那一半再对称复制：covxy += (vx - meanx_new) * dy_old，
//!   两列换序结果不一样（1e-3 量级）。这里的 frame 恒为 [其它列, energy列]，
//!   所以必须拿**第二列当 x**、第一列当 y。本文件统一写成 welford_corr(energy, other)。
//! · sum / mean / std / max / min / median 都跳过 NaN；空组或全 NaN 组的 sum 是 0.0
//!   （这是 bse_* 那 5 项在 66/83 组为空时的行为），mean/std 是 NaN。
//! · std 是样本标准差 ddof=1，非 NaN 个数 < 2 时 NaN。
//! · quantile(q) 是 numpy 的 linear 法：virtual = (n-1)*q，t < 0.5 走 a+(b-a)t，
//!   否则走 b-(b-a)(1-t)。
//! · Series.autocorr(k) = Series.corr(shift(k)) → np.corrcoef（两遍法，notna 成对剔除，
//!   最后 clip 到 [-1,1]）；k >= len 会抛 ValueError，被 get_factor 的 try/except 吞掉。
//!
//! ！！异常语义（照抄）：整段包在 try/except 里，`autocorr(5)` 要求 len > 5，
//!   所以「去重后的行数 < 6」→ 整只股票日返回 None（上游补 NaN）。行数为 0 同理
//!   （len=0 时 autocorr(1) 就抛了），不是返回 130 个 NaN。
//!
//! 四、三个 #[pyfunction] 的纯 Rust 版本
//!   find_half_energy_time / find_half_extreme_time（src/time_series/mod.rs:1586 / 1379）、
//   find_follow_volume_sum_same_price（同文件 842）、mark_follow_groups（同文件 1089）
//!   都是 #[pyfunction]，worker 不带 libpython 不能在链路里调，
//!   所以把纯 Rust 主体抄进来（PyReadonlyArray1 → &[f64]、去掉 PyResult 与 timeout，
//!   因为本系列这三个调用都用默认 timeout=None、filter_ratio=0）。src/time_series 一行没动。
//!   抄写来源：half_extreme_time / half_energy_time 与 hm047.rs 的同名函数逐行一致，
//!   ffvsp 与 hm048.rs 的同名函数逐行一致。

use crate::data_paths;
use std::collections::HashSet;

/// 一天的纳秒数。
const DAY_NS: i64 = 86_400_000_000_000;
/// pandas Timestamp("1970-01-01 08:00:00").value
const H8_NS: i64 = 8 * 3600 * 1_000_000_000;

/// 两个时段（真实钟点的闭区间，单位纳秒）
const OPEN15: (i64, i64) = (hms(9, 30, 0), hms(9, 45, 0));
const CLOSE15: (i64, i64) = (hms(14, 45, 0), hms(14, 57, 0));

const fn hms(h: i64, m: i64, s: i64) -> i64 {
    (h * 3600 + m * 60 + s) * 1_000_000_000
}

/// 三个能量窗口（秒）
const M1: f64 = 1.0;
const N1: f64 = 5.0;
const N2: f64 = 30.0;

/// sing() 的输出长度
const PER_SING: usize = 130;
/// 每只股票的总长度：2 个时段 × 2 个 extreme 版本
pub const N_OUT: usize = PER_SING * 4;
const METRIC_NAMES: [&str; 130] = [
    "buy_sell_energy_N1",
    "buy_sell_energy_N2",
    "buy_sell_energy_M1",
    "buy_sell_energy_5",
    "buy_sell_energy_30",
    "high_price_energy_N1",
    "high_price_energy_N2",
    "high_price_energy_M1",
    "high_price_energy_5",
    "high_price_energy_30",
    "low_price_energy_N1",
    "low_price_energy_N2",
    "low_price_energy_M1",
    "low_price_energy_5",
    "low_price_energy_30",
    "corr_price_N1",
    "corr_price_N2",
    "corr_price_M1",
    "corr_price_5",
    "corr_price_30",
    "corr_volume_N1",
    "corr_volume_N2",
    "corr_volume_M1",
    "corr_volume_5",
    "corr_volume_30",
    "corr_price_N1_66",
    "corr_price_N1_83",
    "corr_price_N2_66",
    "corr_price_N2_83",
    "corr_price_M1_66",
    "corr_price_M1_83",
    "corr_price_5_66",
    "corr_price_5_83",
    "corr_price_30_66",
    "corr_price_30_83",
    "corr_volume_N1_66",
    "corr_volume_N1_83",
    "corr_volume_N2_66",
    "corr_volume_N2_83",
    "corr_volume_M1_66",
    "corr_volume_M1_83",
    "corr_volume_5_66",
    "corr_volume_5_83",
    "corr_volume_30_66",
    "corr_volume_30_83",
    "corr_price_N1_high",
    "corr_price_N1_low",
    "corr_price_N1_mid",
    "corr_price_N2_high",
    "corr_price_N2_low",
    "corr_price_N2_mid",
    "corr_price_M1_high",
    "corr_price_M1_low",
    "corr_price_M1_mid",
    "corr_price_5_high",
    "corr_price_5_low",
    "corr_price_5_mid",
    "corr_price_30_high",
    "corr_price_30_low",
    "corr_price_30_mid",
    "corr_volume_N1_high",
    "corr_volume_N1_low",
    "corr_volume_N1_mid",
    "corr_volume_N2_high",
    "corr_volume_N2_low",
    "corr_volume_N2_mid",
    "corr_volume_M1_high",
    "corr_volume_M1_low",
    "corr_volume_M1_mid",
    "corr_volume_5_high",
    "corr_volume_5_low",
    "corr_volume_5_mid",
    "corr_volume_30_high",
    "corr_volume_30_low",
    "corr_volume_30_mid",
    "half_mean_N1",
    "half_std_N1",
    "half_max_N1",
    "half_min_N1",
    "half_median_N1",
    "half_sum_N1",
    "half_autocorr1_N1",
    "half_autocorr2_N1",
    "half_autocorr3_N1",
    "half_autocorr4_N1",
    "half_autocorr5_N1",
    "half_mean_N2",
    "half_std_N2",
    "half_max_N2",
    "half_min_N2",
    "half_median_N2",
    "half_sum_N2",
    "half_autocorr1_N2",
    "half_autocorr2_N2",
    "half_autocorr3_N2",
    "half_autocorr4_N2",
    "half_autocorr5_N2",
    "half_mean_M1",
    "half_std_M1",
    "half_max_M1",
    "half_min_M1",
    "half_median_M1",
    "half_sum_M1",
    "half_autocorr1_M1",
    "half_autocorr2_M1",
    "half_autocorr3_M1",
    "half_autocorr4_M1",
    "half_autocorr5_M1",
    "rate5_mean",
    "rate5_std",
    "rate5_max",
    "rate5_min",
    "rate5_median",
    "rate5_sum",
    "rate5_autocorr1",
    "rate5_autocorr2",
    "rate5_autocorr3",
    "rate5_autocorr4",
    "rate5_autocorr5",
    "rate30_mean",
    "rate30_std",
    "rate30_max",
    "rate30_min",
    "rate30_median",
    "rate30_sum",
    "rate30_autocorr1",
    "rate30_autocorr2",
    "rate30_autocorr3",
    "rate30_autocorr4",
    "rate30_autocorr5",
];
pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for suf in ["_open15_full", "_close15_full", "_open15_extreme", "_close15_extreme"] {
        for m in METRIC_NAMES.iter() {
            out.push(format!("{}{}", m, suf));
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// pandas / numpy 语义的小工具
// ============================================================================

/// numpy 的 rint：四舍六入五成双。
#[inline]
fn rint(x: f64) -> f64 {
    x.round_ties_even()
}

/// pandas 的 cast_from_unit_vectorized(values, "s", "ns") + 8 小时偏移。
#[inline]
fn to_pandas_ns(exchtime_us: f64) -> i64 {
    let val = exchtime_us / 1e6;
    let base = val as i64;
    let frac = val - base as f64;
    let f9 = rint(frac * 1e9);
    let frac2 = f9 / 1e9;
    base * 1_000_000_000 + (frac2 * 1e9) as i64 + H8_NS
}

/// numpy 的 `np.round(x, 2)`：rint(x*100)/100。
#[inline]
fn round2(x: f64) -> f64 {
    rint(x * 100.0) / 100.0
}

/// numpy 的成对求和（PW_BLOCKSIZE=128，8 路展开）。
fn pairwise_sum(v: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    if v.len() < 8 {
        let mut r = 0.0;
        for &x in v {
            r += x;
        }
        return r;
    }
    if v.len() <= PW_BLOCKSIZE {
        let n = v.len();
        let mut r = [0.0f64; 8];
        let mut i = 0;
        while i + 8 <= n {
            for k in 0..8 {
                r[k] += v[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += v[i];
            i += 1;
        }
        return res;
    }
    let n2 = (v.len() / 2) & !7;
    pairwise_sum(&v[..n2]) + pairwise_sum(&v[n2..])
}

/// pandas 的 sum(skipna=True)：非 NaN 求和；空组或全 NaN 给 0.0。
fn p_sum(v: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut c = 0usize;
    for &x in v {
        if x == x {
            s += x;
            c += 1;
        }
    }
    if c == 0 {
        0.0
    } else {
        s
    }
}

/// pandas 的 mean(skipna=True)。
fn p_mean(v: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut c = 0usize;
    for &x in v {
        if x == x {
            s += x;
            c += 1;
        }
    }
    if c == 0 {
        f64::NAN
    } else {
        s / c as f64
    }
}

/// pandas 的 std(skipna=True, ddof=1)。
fn p_std(v: &[f64]) -> f64 {
    let x: Vec<f64> = v.iter().cloned().filter(|y| y == y).collect();
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = pairwise_sum(&x) / n as f64;
    let sq: Vec<f64> = x.iter().map(|&y| (y - m) * (y - m)).collect();
    (pairwise_sum(&sq) / (n - 1) as f64).sqrt()
}

fn p_max(v: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &x in v {
        if x == x && (m != m || x > m) {
            m = x;
        }
    }
    m
}

fn p_min(v: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &x in v {
        if x == x && (m != m || x < m) {
            m = x;
        }
    }
    m
}

/// pandas 的 median(skipna=True)：偶数个取中间两个的平均。
fn p_median(v: &[f64]) -> f64 {
    let mut a: Vec<f64> = v.iter().cloned().filter(|y| y == y).collect();
    if a.is_empty() {
        return f64::NAN;
    }
    a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.0
    }
}

/// np.percentile(sorted, q, method="linear")，q 取 0..1。
fn percentile_linear_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let vi = (n as f64 - 1.0) * q;
    let mut lo = vi.floor();
    if lo < 0.0 {
        lo = 0.0;
    }
    let mut hi = lo + 1.0;
    if hi > (n - 1) as f64 {
        hi = (n - 1) as f64;
    }
    let (l, h) = (lo as usize, hi as usize);
    let d = sorted[h] - sorted[l];
    let t = vi - lo;
    if t >= 0.5 {
        sorted[h] - d * (1.0 - t)
    } else {
        sorted[l] + d * t
    }
}

/// libalgos.nancorr 的 Welford 版本（DataFrame.corr 走这条）。
/// x 必须是**下标大的那一列**（见文件头）。
fn welford_corr(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs: i64 = 0;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let vx = x[i];
        let vy = y[i];
        nobs += 1;
        let dx = vx - meanx;
        let dy = vy - meany;
        meanx += 1.0 / nobs as f64 * dx;
        meany += 1.0 / nobs as f64 * dy;
        ssqdmx += (vx - meanx) * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += (vx - meanx) * dy;
    }
    if nobs < 1 {
        return f64::NAN;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// Series.corr → np.corrcoef（两遍法）。
fn pearson_corrcoef(x: &[f64], y: &[f64]) -> f64 {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for i in 0..x.len() {
        if x[i] == x[i] && y[i] == y[i] {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mx = pairwise_sum(&xs) / n as f64;
    let my = pairwise_sum(&ys) / n as f64;
    let dxv: Vec<f64> = xs.iter().map(|&v| v - mx).collect();
    let dyv: Vec<f64> = ys.iter().map(|&v| v - my).collect();
    let sxy: Vec<f64> = (0..n).map(|i| dxv[i] * dyv[i]).collect();
    let sxx: Vec<f64> = dxv.iter().map(|&v| v * v).collect();
    let syy: Vec<f64> = dyv.iter().map(|&v| v * v).collect();
    let fact = (n - 1) as f64;
    let c00 = pairwise_sum(&sxx) / fact;
    let c11 = pairwise_sum(&syy) / fact;
    let c01 = pairwise_sum(&sxy) / fact;
    let mut r = c01 / c00.sqrt() / c11.sqrt();
    if r != r {
        return f64::NAN;
    }
    if r > 1.0 {
        r = 1.0;
    }
    if r < -1.0 {
        r = -1.0;
    }
    r
}

/// Series.autocorr(lag)：corr(self, self.shift(lag))。
fn autocorr(v: &[f64], lag: usize) -> f64 {
    let n = v.len();
    if lag >= n {
        return f64::NAN;
    }
    let mut shifted = vec![f64::NAN; n];
    shifted[lag..].copy_from_slice(&v[..n - lag]);
    pearson_corrcoef(v, &shifted)
}

// ============================================================================
// src/time_series/mod.rs 里几个 #[pyfunction] 的纯 Rust 主体
// ============================================================================

/// find_half_extreme_time 的纯 Rust 主体（本系列 direction 走默认 "ignore"）。
fn half_extreme_time(times_ns: &[f64], prices: &[f64], time_window: f64) -> Vec<f64> {
    let times: Vec<f64> = times_ns.iter().map(|&x| x / 1.0e9).collect();
    let n = times.len();
    let mut result = vec![time_window; n];
    for i in 0..n {
        let current_time = times[i];
        let current_price = prices[i];
        if !current_price.is_finite() {
            result[i] = f64::NAN;
            continue;
        }
        let mut max_up = 0.0f64;
        let mut max_down = 0.0f64;
        for j in i..n {
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_ratio = (prices[j] - current_price) / current_price;
            if price_ratio > max_up {
                max_up = price_ratio;
            } else if price_ratio < -max_down {
                max_down = -price_ratio;
            }
        }
        let (target_ratio, dir_value) = if max_up > max_down {
            (max_up, 1.0)
        } else {
            (max_down, -1.0)
        };
        if target_ratio <= 0.0 {
            continue;
        }
        let half_ratio = target_ratio / 2.0 * dir_value;
        for j in i..n {
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_ratio = (prices[j] - current_price) / current_price;
            if (dir_value > 0.0 && price_ratio >= half_ratio)
                || (dir_value < 0.0 && price_ratio <= half_ratio)
            {
                result[i] = time_diff;
                break;
            }
        }
    }
    result
}

/// find_half_energy_time 的纯 Rust 主体（direction 走默认 "ignore"）。
fn half_energy_time(times_ns: &[f64], prices: &[f64], time_window: f64) -> Vec<f64> {
    let times: Vec<f64> = times_ns.iter().map(|&x| x / 1.0e9).collect();
    let n = times.len();
    let mut result = vec![time_window; n];
    for i in 0..n {
        let current_time = times[i];
        let current_price = prices[i];
        if !current_price.is_finite() {
            result[i] = f64::NAN;
            continue;
        }
        let mut final_energy = 0.0f64;
        for j in i..n {
            if j == i {
                continue;
            }
            let time_diff = times[j] - current_time;
            if time_diff < time_window {
                continue;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_change = prices[j] - current_price;
            final_energy = price_change.abs() / current_price;
            break;
        }
        if final_energy <= 0.0 {
            result[i] = 0.0;
            continue;
        }
        let half_energy = final_energy / 2.0;
        for j in i..n {
            if j == i {
                continue;
            }
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_ratio = (prices[j] - current_price).abs() / current_price;
            if price_ratio >= half_energy {
                result[i] = time_diff;
                break;
            }
        }
    }
    result
}

/// find_follow_volume_sum_same_price 的纯 Rust 版（check_price=true、filter_ratio=0）。
fn ffvsp(times_ns: &[f64], prices: &[f64], volumes: &[f64], time_window: f64) -> Vec<f64> {
    let times: Vec<f64> = times_ns.iter().map(|&x| x / 1.0e9).collect();
    let n = times.len();
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let current_time = times[i];
        let current_price = prices[i];
        let current_volume = volumes[i];
        let mut sum = current_volume;
        let mut has_match = false;
        for j in (i + 1)..n {
            if times[j] - current_time > time_window {
                break;
            }
            if (prices[j] - current_price).abs() < 1e-10
                && (volumes[j] - current_volume).abs() < 1e-10
            {
                has_match = true;
                sum += volumes[j];
            }
        }
        result[i] = if !has_match { f64::NAN } else { sum };
    }
    result
}

/// mark_follow_groups 的纯 Rust 版：组号从 1 开始，0 表示没归组。
fn mark_follow_groups(times_ns: &[f64], prices: &[f64], volumes: &[f64], time_window: f64) -> Vec<i32> {
    let times: Vec<f64> = times_ns.iter().map(|&x| x / 1.0e9).collect();
    let n = times.len();
    let mut result = vec![0i32; n];
    let mut current_group = 0i32;
    for i in 0..n {
        if result[i] != 0 {
            continue;
        }
        let current_time = times[i];
        let current_price = prices[i];
        let current_volume = volumes[i];
        let mut has_group = false;
        for j in i..n {
            if j > i && times[j] - current_time > time_window {
                break;
            }
            if (prices[j] - current_price).abs() < 1e-10
                && (volumes[j] - current_volume).abs() < 1e-10
            {
                if !has_group {
                    current_group += 1;
                    has_group = true;
                }
                result[j] = current_group;
            }
        }
    }
    result
}

// ============================================================================
// sing()：一个时段 × 一个 extreme 版本的 130 个值
// ============================================================================

/// 五个「能量类」列：energyN1 / energyN2 / energyM1 / fivetoone / thirtytofive。
struct Cols {
    n1: Vec<f64>,
    n2: Vec<f64>,
    m1: Vec<f64>,
    f5: Vec<f64>,
    t30: Vec<f64>,
}

fn sing(
    ns: &[i64],
    price: &[f64],
    volume: &[f64],
    flag: &[i32],
    sel: &[usize],
    extreme: bool,
) -> Option<Vec<f64>> {
    if sel.is_empty() || sel.len() >= 100_000 {
        return Some(vec![f64::NAN; PER_SING]);
    }
    let t: Vec<f64> = sel.iter().map(|&i| ns[i] as f64).collect();
    let mut p: Vec<f64> = sel.iter().map(|&i| price[i]).collect();
    let mut v: Vec<f64> = sel.iter().map(|&i| volume[i]).collect();
    let fl: Vec<i32> = sel.iter().map(|&i| flag[i]).collect();

    let mut m1 = if extreme { half_extreme_time(&t, &p, M1) } else { half_energy_time(&t, &p, M1) };
    let mut n1 = if extreme { half_extreme_time(&t, &p, N1) } else { half_energy_time(&t, &p, N1) };
    let mut n2 = if extreme { half_extreme_time(&t, &p, N2) } else { half_energy_time(&t, &p, N2) };
    for a in [&mut m1, &mut n1, &mut n2] {
        for x in a.iter_mut() {
            if *x == 0.0 {
                *x = f64::NAN;
            }
        }
    }
    let mut f5: Vec<f64> = (0..p.len()).map(|i| n1[i] / m1[i]).collect();
    let mut t30: Vec<f64> = (0..p.len()).map(|i| n2[i] / n1[i]).collect();

    // follow_price → 只留 round(follow_price - volume, 2) > 0 的行
    let fp = ffvsp(&t, &p, &v, 0.1);
    let mut keep: Vec<usize> = Vec::with_capacity(p.len());
    for i in 0..p.len() {
        if round2(fp[i] - v[i]) > 0.0 {
            keep.push(i);
        }
    }
    if keep.is_empty() {
        // autocorr(1) 在 len=0 时抛 ValueError，整只股票日作废
        return None;
    }
    let take = |a: &Vec<f64>| -> Vec<f64> { keep.iter().map(|&i| a[i]).collect() };
    p = take(&p);
    v = take(&v);
    m1 = take(&m1);
    n1 = take(&n1);
    n2 = take(&n2);
    f5 = take(&f5);
    t30 = take(&t30);
    let t2: Vec<f64> = keep.iter().map(|&i| t[i]).collect();
    let fl: Vec<i32> = keep.iter().map(|&i| fl[i]).collect();

    // 同一 follow_group 只留第一次出现
    let groups = mark_follow_groups(&t2, &p, &v, 0.1);
    let mut seen: HashSet<i32> = HashSet::new();
    let mut keep2: Vec<usize> = Vec::new();
    for (i, &g) in groups.iter().enumerate() {
        if seen.insert(g) {
            keep2.push(i);
        }
    }
    let take2 = |a: &Vec<f64>| -> Vec<f64> { keep2.iter().map(|&i| a[i]).collect() };
    p = take2(&p);
    v = take2(&v);
    m1 = take2(&m1);
    n1 = take2(&n1);
    n2 = take2(&n2);
    f5 = take2(&f5);
    t30 = take2(&t30);
    let fl: Vec<i32> = keep2.iter().map(|&i| fl[i]).collect();

    let nrow = p.len();
    if nrow < 6 {
        // autocorr(5) 要求 lag < len，否则 ValueError → 整只股票日 None
        return None;
    }

    let mut sp = p.clone();
    sp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q10 = percentile_linear_sorted(&sp, 0.1);
    let q90 = percentile_linear_sorted(&sp, 0.9);

    let m66: Vec<usize> = (0..nrow).filter(|&i| fl[i] == 66).collect();
    let m83: Vec<usize> = (0..nrow).filter(|&i| fl[i] == 83).collect();
    let mhi: Vec<usize> = (0..nrow).filter(|&i| p[i] >= q90).collect();
    let mlo: Vec<usize> = (0..nrow).filter(|&i| p[i] <= q10).collect();
    let mmid: Vec<usize> = (0..nrow)
        .filter(|&i| p[i] >= q10 && p[i] <= q90)
        .collect();

    let pick = |a: &Vec<f64>, idx: &Vec<usize>| -> Vec<f64> { idx.iter().map(|&i| a[i]).collect() };

    let bse = |a: &Vec<f64>| -> f64 {
        let s66 = p_sum(&pick(a, &m66));
        let s83 = p_sum(&pick(a, &m83));
        (s66 - s83) / (s66 + s83)
    };
    let pe = |a: &Vec<f64>, idx: &Vec<usize>| -> f64 { p_mean(&pick(a, idx)) };

    let cols = Cols {
        n1,
        n2,
        m1,
        f5,
        t30,
    };
    let energy: [&Vec<f64>; 5] = [&cols.n1, &cols.n2, &cols.m1, &cols.f5, &cols.t30];

    let mut res: Vec<f64> = Vec::with_capacity(PER_SING);
    // 0-4 buy_sell_energy_{N1,N2,M1,5,30}
    for a in energy.iter() {
        res.push(bse(a));
    }
    // 5-9 high_price_energy_{N1,N2,M1,5,30}
    for a in energy.iter() {
        res.push(pe(a, &mhi) / pe(a, &mmid));
    }
    // 10-14 low_price_energy_{N1,N2,M1,5,30}
    for a in energy.iter() {
        res.push(pe(a, &mlo) / pe(a, &mmid));
    }
    // 15-24 corr_price_{N1,N2,M1,5,30} 然后 corr_volume_同序
    for a in energy.iter() {
        res.push(welford_corr(a, &p));
    }
    for a in energy.iter() {
        res.push(welford_corr(a, &v));
    }
    // 25-44 每组 {66,83}，先 price 后 volume
    for base in [&p, &v] {
        for a in energy.iter() {
            let a66 = pick(a, &m66);
            let b66 = pick(base, &m66);
            res.push(welford_corr(&a66, &b66));
            let a83 = pick(a, &m83);
            let b83 = pick(base, &m83);
            res.push(welford_corr(&a83, &b83));
        }
    }
    // 45-74 每组 {high, low, mid}，先 price 后 volume
    for base in [&p, &v] {
        for a in energy.iter() {
            for idx in [&mhi, &mlo, &mmid] {
                let aa = pick(a, idx);
                let bb = pick(base, idx);
                res.push(welford_corr(&aa, &bb));
            }
        }
    }
    // 75-129 五列 × {mean, std, max, min, median, sum, autocorr1..5}
    for a in energy.iter() {
        res.push(p_mean(a));
        res.push(p_std(a));
        res.push(p_max(a));
        res.push(p_min(a));
        res.push(p_median(a));
        res.push(p_sum(a));
        for lag in 1..=5 {
            res.push(autocorr(a, lag));
        }
    }
    debug_assert_eq!(res.len(), PER_SING);
    Some(res)
}

// ============================================================================
// CSV 读取（自己按 f64 解析，复刻 pandas 的 precise_xstrtod）
// ============================================================================

fn trim_ascii(raw: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = raw.len();
    while s < e && raw[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && raw[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    &raw[s..e]
}

/// 取字段的整数前缀：`"32.0"` 给 32。
fn int_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    let mut i = 0usize;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// exchtime 列（形如 `1740964500130000.0`）：整数前缀就是微秒数。
fn us_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    if !b.contains(&b'e') && !b.contains(&b'E') {
        return int_prefix(b);
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.round() as i64)
        .unwrap_or(0)
}

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（同 hm033.rs / hm090.rs / hm047.rs）。
fn parse_f64_pandas(b: &[u8]) -> f64 {
    const MAX_DIGITS: usize = 17;
    let n = b.len();
    let mut p = 0usize;
    while p < n && matches!(b[p], b' ' | b'\t' | b'\r' | b'\n') {
        p += 1;
    }
    let mut negative = false;
    if p < n && (b[p] == b'-' || b[p] == b'+') {
        negative = b[p] == b'-';
        p += 1;
    }
    let mut number = 0.0f64;
    let mut exponent: i32 = 0;
    let mut num_digits = 0usize;
    let mut num_decimals = 0usize;
    while p < n && b[p].is_ascii_digit() {
        if num_digits < MAX_DIGITS {
            number = number * 10.0 + (b[p] - b'0') as f64;
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    if p < n && b[p] == b'.' {
        p += 1;
        while num_digits < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            number = number * 10.0 + (b[p] - b'0') as f64;
            p += 1;
            num_digits += 1;
            num_decimals += 1;
        }
        if num_digits >= MAX_DIGITS {
            while p < n && b[p].is_ascii_digit() {
                p += 1;
            }
        }
        exponent -= num_decimals as i32;
    }
    if num_digits == 0 {
        return 0.0;
    }
    if negative {
        number = -number;
    }
    if p < n && (b[p] | 0x20) == b'e' {
        p += 1;
        let mut eneg = false;
        if p < n && (b[p] == b'-' || b[p] == b'+') {
            eneg = b[p] == b'-';
            p += 1;
        }
        let mut nn: i32 = 0;
        let mut nd = 0usize;
        while nd < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            nn = nn * 10 + (b[p] - b'0') as i32;
            nd += 1;
            p += 1;
        }
        if eneg {
            exponent -= nn;
        } else {
            exponent += nn;
        }
    }
    if exponent > 308 {
        f64::INFINITY
    } else if exponent > 0 {
        number * pow10_table()[exponent as usize]
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            let t = pow10_table();
            (number / t[(-308 - exponent) as usize]) / t[308]
        }
    } else {
        number / pow10_table()[(-exponent) as usize]
    }
}

fn pow10_table() -> &'static [f64; 309] {
    static T: std::sync::OnceLock<[f64; 309]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [1.0f64; 309];
        for (k, v) in t.iter_mut().enumerate() {
            *v = format!("1e{}", k).parse::<f64>().unwrap();
        }
        t
    })
}

fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
}

/// 一天的逐笔（已丢 flag==32），返回 (纳秒, price, volume, flag)。
/// 文件不存在返回 None（旧框架 get_stock_data_by_dates 也返回 None）。
fn read_day(code: &str, date: i64) -> Option<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i32>)> {
    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_FLAG: usize = 10;
    const NCOL: usize = 15;

    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in data_paths::level2_roots() {
        let path = root
            .join(date.to_string())
            .join("transaction")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    let mut ns_out: Vec<i64> = Vec::new();
    let mut price_out: Vec<f64> = Vec::new();
    let mut vol_out: Vec<f64> = Vec::new();
    let mut flag_out: Vec<i32> = Vec::new();

    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; NCOL] = [&[][..]; NCOL];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < NCOL {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < NCOL {
            fields[commas] = &line[start..];
        }
        if commas + 1 < NCOL {
            continue;
        }
        let exchtime_us = us_prefix(fields[COL_EXCHTIME]);
        if exchtime_us == 0 {
            continue;
        }
        let flag = int_prefix(fields[COL_FLAG]) as i32;
        if flag == 32 {
            continue;
        }
        ns_out.push(to_pandas_ns(exchtime_us as f64));
        price_out.push(num(fields[COL_PRICE]));
        vol_out.push(num(fields[COL_VOLUME]));
        flag_out.push(flag);
    }
    Some((ns_out, price_out, vol_out, flag_out))
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 逐股系列：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let (ns, price, volume, flag) = read_day(code, date)?;
    let tod: Vec<i64> = ns.iter().map(|&x| x.rem_euclid(DAY_NS)).collect();
    // 顺序必须是 open15_full / close15_full / open15_extreme / close15_extreme
    let plan: [((i64, i64), bool); 4] = [
        (OPEN15, false),
        (CLOSE15, false),
        (OPEN15, true),
        (CLOSE15, true),
    ];
    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for ((lo, hi), extreme) in plan.iter() {
        let sel: Vec<usize> = (0..ns.len())
            .filter(|&i| tod[i] >= *lo && tod[i] <= *hi)
            .collect();
        let r = sing(&ns, &price, &volume, &flag, &sel, *extreme)?;
        out.extend_from_slice(&r);
    }
    Some(crate::okay_database::fit(
        out.into_iter().map(|x| x as f32).collect(),
        N_OUT,
    ))
}

/// hm056 是逐股系列，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
