//! hm43「撤单韵律」——旧的 design_whatever_out.TickDataBase 框架的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm43_成果hm44_撤单专区/hm43_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔」一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=1)
//!    ！！with_retreat=1 是**保留** flag==32 的撤单行（库里的判断是
//!      `if not self.with_retreat: data = data[data["flag"] != 32]`）。
//!      本系列撤单行正是主要输入，一行都不能丢，所以 Rust 侧
//!      read_trade_fast_inner(code, date, /*with_retreat=*/true, false, usize::MAX)，
//!      并且**不**再自己 filter flag != 32。
//! 2) deal_data（change_time=1）把 exchtime 变成 Timestamp：
//!      pd.to_timedelta(exchtime_us / 1e6, unit="s") + Timestamp("1970-01-01 08:00:00")
//!    只做时间换算，**没有** adjust_afternoon 的下午平移，所以 Rust 侧用
//!    with_afternoon_adjust=false，窗口取真实钟点。
//! 3) df.assign(dura=df.exchtime).set_index("exchtime").between_time("09:30:00","14:57:00")
//!    微秒精度的闭区间，用 day_us 判。
//! 4) df.resample("1min").apply(one)
//!    pandas 2.3 的 Resampler.apply 就是 aggregate，它对**每个 1 分钟桶**调用 one，
//!    空桶也调用；桶标签是左边界（closed="left"），所以 09:30:00.1 落在 09:30 桶。
//!    one 只在桶内行数 > 10 时返回 Series(bs,buy,sell)，否则返回 None → NaN。
//!    桶内顺序 = CSV 原始顺序（不重排）。
//!    · rt = flag==32 的行；rt_buy = rt[ask_order==0]；rt_sell = rt[bid_order==0]
//!    · ac = flag!=32 的行；ac_buy = ac[flag==66]；ac_sell = ac[flag==83]
//!    · dura = (exchtime - 桶内最早 exchtime) 的秒数，**相对桶内最小值**
//!    · bs = dtw(rt_buy.dura, rt_sell.dura)
//!      buy = dtw(rt_buy.dura, ac_sell.dura)
//!      sell = dtw(rt_sell.dura, ac_buy.dura)
//!    注意 rp.dtw_distance 是**无窗口约束**的朴素 DTW，且两序列有一侧为空、
//!    另一侧非空时返回 inf（空对空返回 0）。inf 会被下一步 replace+dropna 丢掉。
//! 5) replace(±inf, nan).dropna() → 只留下 bs/buy/sell 都有限的分钟桶
//! 6) 六个时段（对留下的分钟桶标签做 between_time，两端都含）：
//!      allday 09:30~14:57 / open30 09:30~10:00 / morning 09:30~11:30 /
//!      afternoon 13:00~14:57 / close30 14:30~14:57 / silent 10:00~14:30
//!    注意相邻时段**互相重叠**（10:00 同时属于 open30 与 silent，
//!    14:30 同时属于 close30 与 silent），不是划分。
//! 7) 每个时段 sing(df)：行数 <= 5 时返回 [nan]*51，否则 51 个值：
//!      bs/buy/sell 的 mean、std(ddof=1)、skew、max、min         → 15 个
//!      corrwith(shift(k))，k=1..5，每档 bs/buy/sell             → 15 个
//!      dtw(iloc[:-k], iloc[k:])，k=1..5，每档 bs/buy/sell       → 15 个
//!      corr() 的三个非对角元 corr_bs_buy/corr_bs_sell/corr_buy_sell → 3 个
//!      dtw(bs,buy)/dtw(bs,sell)/dtw(buy,sell)                   → 3 个
//! 8) 输出 = sing(allday)+sing(open30)+sing(morning)+sing(afternoon)+sing(close30)+sing(silent)
//!    = 51 × 6 = 306 个，顺序与 names_single 完全一致。
//!
//! names（names_single）：51 个指标 × 六个时段后缀，后缀顺序就是上面的时段顺序。
//! names_save：52 个（见 hm43_okay_database.py，原样搬过来）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 几个必须照抄的库语义（照数学定义写会差出验收线）
//! ════════════════════════════════════════════════════════════════════
//! · 时间：pandas 2.3 的浮点 to_timedelta 走 conversion.pyx 的
//!   cast_from_unit_vectorized，不是简单的一次乘法：
//!       base = <int64_t>(exchtime_us / 1e6)          // 截断
//!       frac = exchtime_us / 1e6 - base
//!       frac = np.round(frac, 9)                     // 注意这一步
//!       ns   = base * 1e9 + <int64_t>(frac * 1e9)
//!   np.round(x, 9) = rint(x * 1e9) / 1e9（半值取偶），/1e9 再乘回来会把末位
//!   抹掉 1，所以 ns 与 exchtime_us * 1000 可以差几十纳秒。实测这一层：
//!   不抄（直接用微秒差 / 1e6 当 dura）时 000001@20250303 的最大相对误差
//!   1.09e-5（bs_corr5_open30，值只有 -2.8e-4），抄了以后整条链逐位一致。
//! · df.corr() 走 pandas._libs.algos.nancorr 的 **Welford** 在线算法，
//!   不是 np.corrcoef，也不是「两遍法」。见下面 pearson_welford。
//! · df.corrwith(df.shift(k)) 走 DataFrame.corrwith 的 pearson 分支：
//!       num = sum((x_i - mean(x)) * (y_i - mean(y)))     // 配对是 (x[i], x[i-k])
//!       dom = (count - 1) * std(x) * std(y)              // std 是 ddof=1
//!       corr = num / dom
//!   与 nancorr 的写法不同，也不能互替。
//! · pandas 的 skew 是 G1（带偏差修正）：
//!       m2 = sum((x-mean)^2)，m3 = sum((x-mean)^3)，
//!       |m| < 1e-14 先抹成 0，m2==0 直接给 0，
//!       (n * sqrt(n-1) / (n-2)) * (m3 / m2**1.5)
//! · std 是 ddof=1 的两遍法：avg=sum/n，再 sum((avg-x)^2)/(n-1) 开方。
//! · rp.dtw_distance 是 #[pyfunction]，worker 链路里不能调（worker 不带 libpython），
//!   所以下面自己写纯 Rust 的 dtw，算法照 src/time_series/mod.rs 抄。

use std::collections::BTreeMap;

use crate::fast_csv_reader::read_trade_fast_inner;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const MIN_US: i64 = 60 * SEC_US;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 六个时段的闭区间（秒级钟点，这里换成当日微秒）。
/// 顺序 = 输出顺序 = names 后缀顺序。
const WINDOWS: [(i64, i64, &str); 6] = [
    (LD_0930, LD_1457, "allday"),
    (LD_0930, LD_1000, "open30"),
    (LD_0930, LD_1130, "morning"),
    (LD_1300, LD_1457, "afternoon"),
    (LD_1430, LD_1457, "close30"),
    (LD_1000, LD_1430, "silent"),
];

/// 每个时段的指标顺序（必须与旧 sing() 返回的列表逐个对齐）。
const METRICS: [&str; 51] = [
    "bs_mean", "buy_mean", "sell_mean",
    "bs_std", "buy_std", "sell_std",
    "bs_skew", "buy_skew", "sell_skew",
    "bs_max", "buy_max", "sell_max",
    "bs_min", "buy_min", "sell_min",
    "bs_corr1", "buy_corr1", "sell_corr1",
    "bs_corr2", "buy_corr2", "sell_corr2",
    "bs_corr3", "buy_corr3", "sell_corr3",
    "bs_corr4", "buy_corr4", "sell_corr4",
    "bs_corr5", "buy_corr5", "sell_corr5",
    "bs_dtw1", "buy_dtw1", "sell_dtw1",
    "bs_dtw2", "buy_dtw2", "sell_dtw2",
    "bs_dtw3", "buy_dtw3", "sell_dtw3",
    "bs_dtw4", "buy_dtw4", "sell_dtw4",
    "bs_dtw5", "buy_dtw5", "sell_dtw5",
    "corr_bs_buy", "corr_bs_sell", "corr_buy_sell",
    "dtw_bs_buy", "dtw_bs_sell", "dtw_buy_sell",
];

const PER_WINDOW: usize = 51;
pub const N_OUT: usize = PER_WINDOW * 6;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (_, _, tag) in WINDOWS.iter() {
        for m in METRICS.iter() {
            out.push(format!("{}_{}", m, tag));
        }
    }
    out
}

/// 横截面系列不适用。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

/// 北京时间的当日微秒数（time_us 已含 8 小时偏移）。
#[inline]
fn day_us(time_us: i64) -> i64 {
    time_us.rem_euclid(DAY_US)
}

/// 旧框架 deal_data 里 exchtime → Timestamp 的逐位复刻（见文件头）。
///
/// 返回的值含 +8h 偏移，但本系列只用它做「桶内最早一笔」的差，
/// 常量偏移在减法里抵消，所以不影响结果。
#[inline]
fn pandas_ts_ns(exchtime_us: i64) -> i64 {
    const F: f64 = 1_000_000_000.0;
    let v = exchtime_us as f64 / 1e6;
    let base = v as i64; // C 的 <int64_t> 转换：向零截断
    let frac = v - base as f64;
    let rounded = (frac * F).round_ties_even() / F; // np.round(frac, 9)
    base * 1_000_000_000 + (rounded * F) as i64
}

/// 朴素 DTW，与 rp.dtw_distance(s1, s2)（radius=None）逐位一致。
///
/// 边界：D[0][0]=0，第一行/第一列其余为 +inf。
/// 所以有一侧为空、另一侧非空时结果是 +inf，两侧都空是 0。
/// min 的三次比较在 IEEE 下与顺序无关，滚动两行不会改变结果。
fn dtw(s1: &[f64], s2: &[f64]) -> f64 {
    let n = s1.len();
    let m = s2.len();
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut cur = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = f64::INFINITY;
        let a = s1[i - 1];
        for j in 1..=m {
            let cost = (a - s2[j - 1]).abs();
            let best = prev[j].min(cur[j - 1]).min(prev[j - 1]);
            cur[j] = cost + best;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

#[inline]
fn mean(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        s += x;
    }
    s / v.len() as f64
}

/// pandas 的 std(ddof=1)：两遍法，先均值再平方偏差和。
#[inline]
fn std_ddof1(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let avg = mean(v);
    let mut ss = 0.0f64;
    for &x in v {
        let d = avg - x;
        ss += d * d;
    }
    (ss / (n - 1.0)).sqrt()
}

/// pandas 的 skew（G1，带偏差修正）。见文件头。
fn skew(v: &[f64]) -> f64 {
    let count = v.len();
    if count < 3 {
        return f64::NAN;
    }
    let mu = mean(v);
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &x in v {
        let a = x - mu;
        m2 += a * a;
        m3 += a * a * a;
    }
    // _zero_out_fperr：|x| < 1e-14 一律当 0
    if m2.abs() < 1e-14 {
        m2 = 0.0;
    }
    if m3.abs() < 1e-14 {
        m3 = 0.0;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    let c = count as f64;
    (c * (c - 1.0).sqrt() / (c - 2.0)) * (m3 / m2.powf(1.5))
}

/// pandas._libs.algos.nancorr（DataFrame.corr 的 pearson 路径）：Welford 在线算法。
fn pearson_welford(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 1 {
        return f64::NAN;
    }
    let mut nobs = 0.0f64;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    for i in 0..n {
        let vx = x[i];
        let vy = y[i];
        nobs += 1.0;
        let dx = vx - meanx;
        let dy = vy - meany;
        meanx += 1.0 / nobs * dx;
        meany += 1.0 / nobs * dy;
        ssqdmx += (vx - meanx) * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += (vx - meanx) * dy;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// DataFrame.corrwith(shift(k)) 的 pearson 实现（见文件头）。
fn lag_corr(x: &[f64], k: usize) -> f64 {
    let n = x.len();
    if n <= k {
        return f64::NAN;
    }
    let m = n - k;
    let xs = &x[k..]; // left：df[col]
    let ys = &x[..m]; // right：df[col].shift(k) 的有效段
    let mx = mean(xs);
    let my = mean(ys);
    let mut num = 0.0f64;
    for i in 0..m {
        num += (xs[i] - mx) * (ys[i] - my);
    }
    let dom = ((m as f64) - 1.0) * std_ddof1(xs) * std_ddof1(ys);
    num / dom
}

#[inline]
fn max_of(v: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in v {
        if x > m {
            m = x;
        }
    }
    m
}

#[inline]
fn min_of(v: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    for &x in v {
        if x < m {
            m = x;
        }
    }
    m
}

/// 旧代码的 sing(df)，51 个值。行数 <= 5 时整段 NaN。
fn push_sing(bs: &[f64], buy: &[f64], sell: &[f64], out: &mut Vec<f64>) {
    let n = bs.len();
    if n <= 5 {
        out.extend(std::iter::repeat(f64::NAN).take(PER_WINDOW));
        return;
    }
    let cols = [bs, buy, sell];
    for c in cols.iter() {
        out.push(mean(c));
    }
    for c in cols.iter() {
        out.push(std_ddof1(c));
    }
    for c in cols.iter() {
        out.push(skew(c));
    }
    for c in cols.iter() {
        out.push(max_of(c));
    }
    for c in cols.iter() {
        out.push(min_of(c));
    }
    for k in 1..=5usize {
        for c in cols.iter() {
            out.push(lag_corr(c, k));
        }
    }
    for k in 1..=5usize {
        for c in cols.iter() {
            out.push(dtw(&c[..n - k], &c[k..]));
        }
    }
    out.push(pearson_welford(bs, buy));
    out.push(pearson_welford(bs, sell));
    out.push(pearson_welford(buy, sell));
    out.push(dtw(bs, buy));
    out.push(dtw(bs, sell));
    out.push(dtw(buy, sell));
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // with_retreat=true：撤单行要留着；with_afternoon_adjust=false：用真实钟点切窗口。
    let recs = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;

    // 先按分钟分桶。桶号 = 当日微秒 / 60_000_000，与 pandas resample("1min") 的左闭标签一致。
    // 用 BTreeMap 而不是 HashMap：输出顺序必须按时间升序，且不能依赖哈希迭代顺序。
    // 桶里存的是「窗口内行」的序号（ns / keep 的下标，从 0 开始连续）。
    let mut bins: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
    let mut ns: Vec<i64> = Vec::new();
    let mut keep: Vec<u32> = Vec::new(); // 序号 → recs 下标
    for (i, r) in recs.iter().enumerate() {
        let d = day_us(r.time_us);
        if d < LD_0930 || d > LD_1457 {
            continue; // between_time("09:30:00", "14:57:00") 闭区间
        }
        // time_us 已含 +8h，减回去才是原始 exchtime_us
        ns.push(pandas_ts_ns(r.time_us - 8 * 3600 * SEC_US));
        keep.push(i as u32);
        bins.entry(d / MIN_US).or_default().push((ns.len() - 1) as u32);
    }
    if bins.is_empty() {
        // 全部 NaN：个数恒定，不能少几列
        return Some(vec![f32::NAN; N_OUT]);
    }

    // 每个桶算 bs / buy / sell（桶内行数 <= 10 的桶直接没有值）
    let mut series: Vec<(i64, f64, f64, f64)> = Vec::with_capacity(bins.len());
    for (b, idxs) in bins.iter() {
        if idxs.len() <= 10 {
            continue;
        }
        let mut ns_min = i64::MAX;
        for &i in idxs.iter() {
            let v = ns[i as usize];
            if v < ns_min {
                ns_min = v;
            }
        }
        let mut dura: Vec<f64> = Vec::with_capacity(idxs.len());
        for &i in idxs.iter() {
            dura.push((ns[i as usize] - ns_min) as f64 / 1e9);
        }
        let mut rt_buy: Vec<f64> = Vec::new();
        let mut rt_sell: Vec<f64> = Vec::new();
        let mut ac_buy: Vec<f64> = Vec::new();
        let mut ac_sell: Vec<f64> = Vec::new();
        for (k, &i) in idxs.iter().enumerate() {
            let r = &recs[keep[i as usize] as usize];
            let d = dura[k];
            if r.flag == 32 {
                if r.ask_order == 0 {
                    rt_buy.push(d);
                }
                if r.bid_order == 0 {
                    rt_sell.push(d);
                }
            } else {
                if r.flag == 66 {
                    ac_buy.push(d);
                }
                if r.flag == 83 {
                    ac_sell.push(d);
                }
            }
        }
        let bs = dtw(&rt_buy, &rt_sell);
        let buy = dtw(&rt_buy, &ac_sell);
        let sell = dtw(&rt_sell, &ac_buy);
        // 旧代码 replace(±inf, nan).dropna()：三个值里任何一个非有限就丢掉这一分钟
        if !(bs.is_finite() && buy.is_finite() && sell.is_finite()) {
            continue;
        }
        series.push((*b, bs, buy, sell));
    }

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for (lo, hi, _) in WINDOWS.iter() {
        let mut bs = Vec::new();
        let mut buy = Vec::new();
        let mut sell = Vec::new();
        for &(b, a, c, d) in series.iter() {
            let t = b * MIN_US;
            if t >= *lo && t <= *hi {
                bs.push(a);
                buy.push(c);
                sell.push(d);
            }
        }
        push_sing(&bs, &buy, &sell, &mut out);
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|x| x as f32).collect())
}
