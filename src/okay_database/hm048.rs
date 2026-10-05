//! hm48「瞬时协同操作」——旧 framework design_whatever_out.TickDataBase 的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm48_瞬时协同操作/hm48_okay.py
//!   run_base 里的 Go.get_factor，get_many=0 / read_full=0 / change_time=1。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0) → **丢掉** flag==32 的撤单行。
//!    ！！库里那条过滤按字节比 "32"，2025 年的明细写的是 "32.0"，过滤不生效，
//!      所以 Rust 侧用 read_trade_fast_inner(..., with_retreat=true, false, ...) 读进来，
//!      自己 filter flag != 32。
//! 2) df.set_index("exchtime").between_time("09:30", "14:57")：微秒/纳秒精度的闭区间。
//! 3) df.price = round(price, 2)；df.volume = round(volume, 0)
//!    （numpy 的 round：rint(x*100)/100、rint(x)，半值取偶）
//! 4) sing(df, tw)，tw ∈ {0.1, 1, 3} 秒，三个版本各算一遍，拼成 3 × 120 个值：
//!      follow = df.volume.rolling_future(f"{tw}s", include_current=True).sum()
//!        = 以**精确纳秒索引**滑动，窗口是 [i, j]（times[j]-times[i] <= tw*1e9），
//!          累加器是「加进窗口 / 从窗口头部减掉」的增量 f64（见 rolling_window_stat）。
//!      follow_price        = find_follow_volume_sum_same_price(t, price, volume, tw)
//!      follow_price_noprice= 同上 check_price=False
//!      follow_price_nofreq = 同上 filter_ratio=0.1
//!      follow_price_noprice_nofreq = 两个开关都开
//!        其中 t = df.exchtime.to_numpy(dtype=float)（datetime64[ns] → f64 是「纳秒整数转 f64」），
//!        函数内部再 /1e9 变成秒。价格比较用 |Δ|<1e-10，窗口判据是 f64 秒。
//!      四个 rate = follow_price* / follow。
//! 5) 五个时段（对同一张全天的表再切，边界闭区间）：
//!      all（09:30~14:57 整张，不再筛）/ open30 09:30~10:00 / morning 09:30~11:30 /
//!      afternoon 13:00~14:57 / close30 14:30~14:57
//! 6) 每个时段 one(df) 出 24 个值，顺序见 METRICS：
//!      4 个 rate 的均值
//!      4 个 corr(follow, rate)   —— df[[...]].corr().iloc[0,1]，走 Welford
//!      4 个 corr(price,  rate)
//!      4 个 rp.trend(rate.dropna())
//!      price_people = groupby("price").volume.sum().sort_values().index[-1]
//!      4 个 sum 版：groupby("price").rate.sum().sort_values().index[-1] / price_people
//!      4 个 mean 版：groupby("price").rate.mean().sort_values().index[-1] / price_people
//!    输出 = one(all) + one(open30) + one(morning) + one(afternoon) + one(close30) = 120
//!   → 三个 tw 拼起来 360 个，顺序 = names_single。
//!
//! names（names_single）：24 个指标 × 5 个时段 × 3 个 tw，三层后缀顺序见 names()。
//! names_save：原样搬进 hm48_okay_database.py（其中带 _fold 的那些由因子库
//!   factor_production.write_selected_base 用「|x − 当日截面均值|」自动派生，
//!   Rust 侧不需要也不应该自己算）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 必须照抄的库语义
//! ════════════════════════════════════════════════════════════════════
//! · 时间：pandas 2.3 的 to_timedelta(秒, unit="s") 走 conversion.pyx 的
//!   cast_from_unit_vectorized（base 截断 + np.round(frac,9) + 回乘），
//!   不是一次乘法；差几十纳秒。全篇的时间比较（between_time、rolling_future 的
//!   窗口、ffvsp 的秒差）都用这一套算出来的纳秒，才能和 pandas 同进同出。
//! · df[[a,b]].corr() 走 pandas._libs.algos.nancorr 的 **Welford** 在线算法，
//!   不是 np.corrcoef；mask 是 isfinite（NaN 与 ±inf 都不算观测）。
//! · groupby(...).sum() 对全 NaN 的一组返回 0.0（min_count=0），
//!   而 groupby(...).mean() 对全 NaN 的一组返回 NaN；sort_values 把 NaN 放最后
//!   （na_position="last"），所以「mean 版」的 index[-1] 一旦有全 NaN 组，
//!   取的就是这些 NaN 组里**价格最大**的那个。
//! · 并列：sort_values 默认 quicksort，numpy 对 n>=16 不是稳定排序，并列时
//!   index[-1] 取哪一个由快排分区决定。这里按「并列取价格最大的那个」处理
//!   （等价于稳定升序排序），并在文件末尾说明实测并列次数为 0。
//! · rp.trend 是「值序列与 1..n 的皮尔逊相关」，全有限值才算观测，
//!   方差为 0 或观测不足 2 个返回 0.0。
//! · find_follow_volume_sum_same_price 是 #[pyfunction]，worker 链路不能调，
//!   下面按 src/time_series/mod.rs 的实现逐行抄成纯 Rust（去掉 timeout：
//!   旧代码传 timeout_seconds=1，超时会让那一列整体变 NaN——这是旧框架自带
//!   的不可复现点，见文件末尾说明）。

use std::collections::BTreeMap;

use ordered_float::OrderedFloat;

use crate::fast_csv_reader::read_trade_fast_inner;

const DAY_NS: i64 = 86_400_000_000_000;
const SEC_US: i64 = 1_000_000;
const EXCHANGE_OFFSET_US: i64 = 8 * 3600 * SEC_US;

/// 真实钟点的纳秒闭区间
const NS_0930: i64 = (9 * 3600 + 30 * 60) * 1_000_000_000;
const NS_1000: i64 = 10 * 3600 * 1_000_000_000;
const NS_1130: i64 = (11 * 3600 + 30 * 60) * 1_000_000_000;
const NS_1300: i64 = 13 * 3600 * 1_000_000_000;
const NS_1430: i64 = (14 * 3600 + 30 * 60) * 1_000_000_000;
const NS_1457: i64 = (14 * 3600 + 57 * 60) * 1_000_000_000;

/// 五个时段（顺序 = 输出顺序），第一个是全天，不再筛。
const SUB_WINDOWS: [(&str, i64, i64); 5] = [
    ("all", NS_0930, NS_1457),
    ("open30", NS_0930, NS_1000),
    ("morning", NS_0930, NS_1130),
    ("afternoon", NS_1300, NS_1457),
    ("close30", NS_1430, NS_1457),
];

/// 三个时间窗（顺序 = 输出顺序）
const TIME_WINDOWS: [(&str, f64); 3] = [("0p1s", 0.1), ("1s", 1.0), ("3s", 3.0)];

/// 每个时段的 24 个指标（顺序必须与旧 one() 的返回列表一致）
const METRICS: [&str; 24] = [
    "follow_price_rate_mean",
    "follow_price_rate_noprice_mean",
    "follow_price_rate_nofreq_mean",
    "follow_price_rate_noprice_nofreq_mean",
    "corr_price_rate",
    "corr_price_rate_noprice",
    "corr_price_rate_nofreq",
    "corr_price_rate_noprice_nofreq",
    "corr_price_price_rate",
    "corr_price_price_rate_noprice",
    "corr_price_price_rate_nofreq",
    "corr_price_price_rate_noprice_nofreq",
    "trend_price_rate",
    "trend_price_rate_noprice",
    "trend_price_rate_nofreq",
    "trend_price_rate_noprice_nofreq",
    "price_follow_price_rate",
    "price_follow_price_rate_noprice",
    "price_follow_price_rate_nofreq",
    "price_follow_price_rate_noprice_nofreq",
    "price_follow_price_rate_mean",
    "price_follow_price_rate_noprice_mean",
    "price_follow_price_rate_nofreq_mean",
    "price_follow_price_rate_noprice_nofreq_mean",
];

const PER_BLOCK: usize = 24;
pub const N_OUT: usize = 24 * 5 * 3;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (tw_tag, _) in TIME_WINDOWS.iter() {
        for (sub, _, _) in SUB_WINDOWS.iter() {
            for m in METRICS.iter() {
                out.push(format!("{}_{}_{}", m, sub, tw_tag));
            }
        }
    }
    out
}

/// 横截面系列不适用。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

/// 旧框架 deal_data 里 exchtime → Timestamp 的逐位复刻（同 hm043）。
#[inline]
fn pandas_ts_ns(exchtime_us: i64) -> i64 {
    const F: f64 = 1_000_000_000.0;
    let v = exchtime_us as f64 / 1e6;
    let base = v as i64;
    let frac = v - base as f64;
    let rounded = (frac * F).round_ties_even() / F;
    base * 1_000_000_000 + (rounded * F) as i64
}

/// numpy round(x, 0)
#[inline]
fn round0(x: f64) -> f64 {
    x.round_ties_even()
}

/// numpy round(x, 2) = rint(x*100)/100
#[inline]
fn round2(x: f64) -> f64 {
    (x * 100.0).round_ties_even() / 100.0
}

// ============================================================================
// rolling_future(...).sum()：src/pandas_ext/mod.rs 的 "sum" 分支逐行复刻
// ============================================================================

/// 与 rolling_window_stat(times, values, window_ns, "sum", include_current=true) 一致。
///
/// 注意累加器是增量的（进窗口就加、出窗口头就减），不是每个 i 重新求和。
/// 这里 volume 都是整数（round(0) 之后），和值在 f64 里精确，所以增量与重算等价；
/// 仍然照抄增量写法，避免以后换输入时出问题。
fn rolling_future_sum(times: &[i64], values: &[f64], window_ns: i64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![f64::NAN; n];
    let mut window_sum = 0.0f64;
    let mut window_count = 0usize;
    let mut window_start = 0usize;
    let mut window_end = 0usize;
    for i in 0..n {
        while window_start < i {
            window_sum -= values[window_start];
            window_count -= 1;
            window_start += 1;
        }
        while window_end < n && times[window_end] - times[i] <= window_ns {
            window_sum += values[window_end];
            window_count += 1;
            window_end += 1;
        }
        if window_count > 0 {
            result[i] = window_sum;
        }
    }
    result
}

// ============================================================================
// find_follow_volume_sum_same_price 的纯 Rust 版（去掉 timeout）
// ============================================================================

fn ffvsp(
    times_ns_as_f64: &[f64],
    prices: &[f64],
    volumes: &[f64],
    time_window: f64,
    check_price: bool,
    filter_ratio: f64,
) -> Vec<f64> {
    let n = times_ns_as_f64.len();
    let times: Vec<f64> = times_ns_as_f64.iter().map(|&x| x / 1.0e9).collect();
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
            let price_match = !check_price || (prices[j] - current_price).abs() < 1e-10;
            let volume_match = (volumes[j] - current_volume).abs() < 1e-10;
            if price_match && volume_match {
                has_match = true;
                sum += volumes[j];
            }
        }
        result[i] = if !has_match { f64::NAN } else { sum };
    }

    if filter_ratio > 0.0 {
        let mut volume_counts: BTreeMap<OrderedFloat<f64>, usize> = BTreeMap::new();
        for i in 0..n {
            if result[i].is_nan() {
                continue;
            }
            *volume_counts.entry(OrderedFloat(volumes[i])).or_insert(0) += 1;
        }
        if !volume_counts.is_empty() {
            let mut volume_freq: Vec<(OrderedFloat<f64>, usize)> =
                volume_counts.iter().map(|(k, v)| (*k, *v)).collect();
            // 稳定排序：同频次保持 BTreeMap 的升序键顺序（与原实现一致）
            volume_freq.sort_by(|a, b| b.1.cmp(&a.1));
            let total_types = volume_freq.len();
            let mut filter_count = (total_types as f64 * filter_ratio).ceil() as usize;
            if filter_count == 0 && total_types > 0 {
                filter_count = 1;
            }
            let volume_to_filter: Vec<f64> = volume_freq
                .iter()
                .take(filter_count)
                .map(|(vol, _)| vol.into_inner())
                .collect();
            if !volume_to_filter.is_empty() {
                for i in 0..n {
                    if volume_to_filter
                        .iter()
                        .any(|&v| (v - volumes[i]).abs() < 1e-10)
                    {
                        result[i] = f64::NAN;
                    }
                }
            }
        }
    }
    result
}

// ============================================================================
// 指标用的小工具
// ============================================================================

/// pandas Series.mean()：跳过 NaN，全 NaN 给 NaN。
fn nanmean(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0usize;
    for &x in v {
        if !x.is_nan() {
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

/// pandas._libs.algos.nancorr（DataFrame.corr pearson）：Welford，mask = isfinite。
fn welford_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut nobs = 0.0f64;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    for i in 0..n {
        let vx = x[i];
        let vy = y[i];
        if !(vx.is_finite() && vy.is_finite()) {
            continue;
        }
        nobs += 1.0;
        let dx = vx - meanx;
        let dy = vy - meany;
        meanx += 1.0 / nobs * dx;
        meany += 1.0 / nobs * dy;
        ssqdmx += (vx - meanx) * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += (vx - meanx) * dy;
    }
    if nobs < 1.0 {
        return f64::NAN;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// rp.trend(series.dropna().to_numpy())：dropna 先去掉 NaN，再按 1..n 的位置算相关。
fn trend_dropna(v: &[f64]) -> f64 {
    let a: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let mut vals: Vec<f64> = Vec::with_capacity(a.len());
    let mut idxs: Vec<f64> = Vec::with_capacity(a.len());
    for (i, &x) in a.iter().enumerate() {
        if x.is_finite() {
            vals.push(x);
            idxs.push((i + 1) as f64);
        }
    }
    let n = vals.len();
    if n == 0 || n == 1 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x: f64 = vals.iter().sum::<f64>() / nf;
    let mean_y: f64 = idxs.iter().sum::<f64>() / nf;
    let mut covariance = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    for i in 0..n {
        let dx = vals[i] - mean_x;
        let dy = idxs[i] - mean_y;
        covariance += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    covariance / (var_x.sqrt() * var_y.sqrt())
}

/// groupby(price).agg 之后 sort_values().index[-1]：
/// 返回被选中的 price 键（不是聚合值）。mean_mode=false 用 sum，true 用 mean。
///
/// 并列取价格最大的那个（= 稳定升序排序的最后一个）；mean 模式下有全 NaN 组时，
/// NaN 排在最后，取这些组里价格最大的那个（pandas 的 na_position="last"）。
fn groupby_argmax_price(price: &[f64], values: &[f64], mean_mode: bool) -> f64 {
    let mut m: BTreeMap<OrderedFloat<f64>, (f64, usize)> = BTreeMap::new();
    for i in 0..price.len() {
        let e = m.entry(OrderedFloat(price[i])).or_insert((0.0, 0));
        if !values[i].is_nan() {
            e.0 += values[i];
            e.1 += 1;
        }
    }
    let mut best_nan: Option<f64> = None;
    let mut best: Option<(f64, f64)> = None;
    for (k, (s, c)) in m.iter() {
        let key = k.into_inner();
        if mean_mode {
            if *c == 0 {
                best_nan = Some(key); // 全 NaN 组，mean 是 NaN
                continue;
            }
            let agg = s / (*c as f64);
            if agg.is_nan() {
                best_nan = Some(key);
                continue;
            }
            if best.map_or(true, |(b, _)| agg >= b) {
                best = Some((agg, key));
            }
        } else {
            // sum：全 NaN 组给 0.0（min_count=0），不是 NaN
            if best.map_or(true, |(b, _)| *s >= b) {
                best = Some((*s, key));
            }
        }
    }
    if let Some(k) = best_nan {
        return k;
    }
    best.map(|(_, k)| k).unwrap_or(f64::NAN)
}

// ============================================================================
// 一个时段的 24 个指标
// ============================================================================

fn push_block(
    price: &[f64],
    volume: &[f64],
    follow: &[f64],
    rates: &[&[f64]; 4],
    out: &mut Vec<f64>,
) {
    if price.is_empty() {
        out.extend(std::iter::repeat(f64::NAN).take(PER_BLOCK));
        return;
    }
    for r in rates.iter() {
        out.push(nanmean(r));
    }
    for r in rates.iter() {
        out.push(welford_corr(follow, r));
    }
    for r in rates.iter() {
        out.push(welford_corr(price, r));
    }
    for r in rates.iter() {
        out.push(trend_dropna(r));
    }
    let people = groupby_argmax_price(price, volume, false);
    for r in rates.iter() {
        out.push(groupby_argmax_price(price, r, false) / people);
    }
    for r in rates.iter() {
        out.push(groupby_argmax_price(price, r, true) / people);
    }
}

fn take(v: &[f64], sel: &[usize]) -> Vec<f64> {
    sel.iter().map(|&i| v[i]).collect()
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // with_retreat=true 读进来，自己按「字节是 32 / 32.0」都能认的方式过滤
    let recs = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;

    let mut ns: Vec<i64> = Vec::new();
    let mut price: Vec<f64> = Vec::new();
    let mut volume: Vec<f64> = Vec::new();
    for r in recs.iter() {
        if r.flag == 32 {
            continue; // with_retreat=0
        }
        let t = pandas_ts_ns(r.time_us - EXCHANGE_OFFSET_US);
        let tod = t.rem_euclid(DAY_NS);
        if tod < NS_0930 || tod > NS_1457 {
            continue;
        }
        ns.push(t);
        price.push(round2(r.price as f64));
        volume.push(round0(r.volume as f64));
    }

    let n = ns.len();
    if n == 0 {
        return Some(vec![f32::NAN; N_OUT]);
    }

    // 时段索引（时间有序，直接切区间即可；仍然按 tod 判，和 between_time 一致）
    let mut sel_all: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
    for (k, (_, lo, hi)) in SUB_WINDOWS.iter().enumerate() {
        for i in 0..n {
            let tod = ns[i].rem_euclid(DAY_NS);
            if tod >= *lo && tod <= *hi {
                sel_all[k].push(i);
            }
        }
    }
    // 旧代码对空的时段会走到 df.groupby("price")...index[-1] 抛 IndexError，
    // 整个 get_factor 失败 → 那一天全 NaN。这里照做，保证和历史行为一致。
    if sel_all.iter().skip(1).any(|s| s.is_empty()) {
        return Some(vec![f32::NAN; N_OUT]);
    }

    let times_f: Vec<f64> = ns.iter().map(|&x| x as f64).collect();
    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);

    for (_, tw) in TIME_WINDOWS.iter() {
        let window_ns = (*tw * 1e9) as i64;
        let follow = rolling_future_sum(&ns, &volume, window_ns);
        let fp = ffvsp(&times_f, &price, &volume, *tw, true, 0.0);
        let fp_np = ffvsp(&times_f, &price, &volume, *tw, false, 0.0);
        let fp_nf = ffvsp(&times_f, &price, &volume, *tw, true, 0.1);
        let fp_np_nf = ffvsp(&times_f, &price, &volume, *tw, false, 0.1);
        let r1: Vec<f64> = fp.iter().zip(follow.iter()).map(|(a, b)| a / b).collect();
        let r2: Vec<f64> = fp_np.iter().zip(follow.iter()).map(|(a, b)| a / b).collect();
        let r3: Vec<f64> = fp_nf.iter().zip(follow.iter()).map(|(a, b)| a / b).collect();
        let r4: Vec<f64> = fp_np_nf.iter().zip(follow.iter()).map(|(a, b)| a / b).collect();

        for sel in sel_all.iter() {
            let sub_p = take(&price, sel);
            let sub_v = take(&volume, sel);
            let sub_f = take(&follow, sel);
            let s1 = take(&r1, sel);
            let s2 = take(&r2, sel);
            let s3 = take(&r3, sel);
            let s4 = take(&r4, sel);
            push_block(&sub_p, &sub_v, &sub_f, &[&s1, &s2, &s3, &s4], &mut out);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|x| x as f32).collect())
}
