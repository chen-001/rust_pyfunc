//! hm024「翻山越岭」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm24_翻山越岭/hm24_okay.py
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码的形状
//! ════════════════════════════════════════════════════════════════════
//! run_base 的参数：get_many=0、read_full=0、change_time=1、cross_section=False。
//! get_factor() 里：
//!   df = read_trade_data_single(with_retreat=0)      # 数值比较丢掉 flag==32
//!   df = df.set_index("exchtime")                    # Timestamp（epoch 微秒 → ns，带 8h 偏移）
//!   df_all  = between_time("09:30:00","14:57:00")
//!   df_mor  = between_time("09:30:00","11:30:00")
//!   df_aft  = between_time("13:00:00","14:57:00")
//!   （df_open30 / df_close30 被注释掉了，不参与）
//!   lst = sing(df_all) + sing(df_morning) + sing(df_afternoon)
//!
//! sing(df) 做四件事：
//!   res1 = one(df)          # 全部逐笔
//!   res2 = one(df[flag==66])# 主买
//!   res3 = one(df[flag==83])# 主卖
//!   res4 = [abs(res2[k]-res3[k]) for k in (17,23,24,25,26,41,42,48,51)]   # 9 项
//!   return res1 + res2 + res3 + res4       # 65+65+65+9 = 204 项
//!
//! ★ 长度错位（必须照抄，不能"修正"）：
//!   one() 返回 65 项，res4 有 **9** 项，所以 sing 是 204 项；
//!   而旧文件拼出来的 names 只有 65*3+**8** = 203 项，三个时段合计 609 项。
//!   旧框架 run_factor_out 里是 `zip(res, names_single)`，按短的截断，
//!   所以 base 文件第 i 列 = 值向量第 i 项；609 之后那 3 个值被丢掉。
//!   本文件忠实复现：compute_stock 返回 612 项，names() 返回 612 个名字
//!   （前 609 个与旧文件逐字相同、顺序相同，多出来的 3 个是旧文件里没有的，
//!     命名见下面 EXTRA_NAMES 的注释）。这样 609 项的列名↔取值映射与旧 base 完全一致。
//!
//! one(df) 的算法（逐条对齐）：
//!   param = 30 秒
//!   df_volume = df.groupby("exchtime").volume.sum()              # 按"秒"聚合（其实是纳秒）
//!   df_price  = turnover.sum() / volume.sum()                    # 秒内 vwap
//!   before_volume = volume.rolling("30s").sum()                  # 左开右闭 (t-30s, t]
//!   dfr = df.copy(); dfr.index -= 30s
//!   df = concat([df.assign(label=0), dfr.assign(label=1)]).sort_index()
//!   volume_label = volume * label
//!   after_volume = volume_label.rolling("30s").sum() + volume
//!     → 对 label==0 的行：after(t) = volume(t) + Σ volume(原时刻 ∈ (t, t+30s])
//!   df = df[label==0]；ba = before / after
//!   begins0 = ba[ba <= ba.quantile(0.1)]；ends0 = ba[ba >= ba.quantile(0.9)]
//!   begins = df[df.index.isin(begins0.index)].assign(ba=begins0.tolist())
//!                      .assign(volume_rate = volume/after_volume)
//!   ends   = df[df.index.isin(ends0.index)].assign(ba=ends0.tolist())
//!                      .assign(volume_rate = volume/before_volume)
//!   begins/ends 各自 reset_index 后 times = exchtime.diff().dt.total_seconds()（首项 NaN）
//!   trend_label = 0..n-1
//!   5 列 [price, trend_label, volume_rate, ba, times] 上：
//!     · DataFrame.corr() → libalgos.nancorr（Welford，mask 用 isfinite）
//!     · Series.autocorr(k) → Series.corr → np.corrcoef（两遍法，mask 用 notna）
//!     · mean/max/median 都是跳过 NaN 的
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、必须逐位复刻的三处（都不是"顺手"能替代的）
//! ════════════════════════════════════════════════════════════════════
//! 1) **exchtime → 纳秒**：pandas 走 conversion.pyx 的 cast_from_unit_vectorized，
//!      val = exchtime_us/1e6; base = (int64)val; frac = val - base;
//!      frac = rint(frac*1e9)/1e9;  ns = base*1e9 + (int64)(frac*1e9) + 8h。
//!    这条链自带 ±240ns 的噪声。别以为"微秒精度就够了"：滚动窗口左端是**开**区间，
//!    数据里相邻两笔常常正好差 30 秒，噪声的符号决定那一笔算不算进窗口；
//!    实测 000001@20250303 的 all 段 21815 个秒里 275 个 after_volume 因此不同，
//!    最终 612 个因子里 105 个超 1e-4（001 那三个 times_* 量甚至直接 0 vs 2.38e-7）。
//!
//! 2) **concat(...).sort_index() 的并列顺序**：sort_index 是单键，走
//!    pandas nargsort → DatetimeArray.argsort('quicksort') → numpy 的
//!    aquicksort_<datetime_tag>（**不是** int64 的 x86-simd-sort 那条路，实测两者
//!    在 43630 个元素上差 572 个位置）。滚动窗口的 end 恒为 i+1，
//!    所以同一个纳秒上 label0 与 label1 两行的先后决定了那一笔算不算进窗口。
//!    本文件的 np_argsort_quicksort 与 hm013.rs / hm010.rs 里那份一致（逐行复刻）。
//!
//! 3) **roll_sum**：pandas 的变量窗口 roll_sum 用 Kahan 补偿 + 增量 add/remove，
//!    窗口边界是 start[i] = 从 start[i-1] 起第一个 index[j] > index[i]-window 的 j，
//!    end[i] = i+1（right_closed）。照抄。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、已知的可接受差异
//! ════════════════════════════════════════════════════════════════════
//! · 逐秒的 turnover 求和：pandas groupby().sum()（float64）用 Kahan 补偿，本文件用
//!   顺序累加。差 1 ulp 量级，且只影响 price = turnover/volume 这一类下游相关系数，
//!   实测五组样本的最大相对误差都在 1e-9 以下，远低于 1e-4 门槛。
//! · 均值/中位数：numpy 用成对求和，本文件顺序求和。同样是末位量级。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、边界分支
//! ════════════════════════════════════════════════════════════════════
//! · begins 或 ends 为空 → one() 返回 `[np.nan]*66`（旧代码就是这么写的，
//!   长度从 65 变 66，后面的列会整体错位；照抄，反正上游 fit() 会截断/补齐）。
//! · begins 或 ends 长度 < 4 → Series.autocorr(k)（k=1/2/3）抛
//!   ValueError("lag should be smaller than series length")，被 get_factor 的
//!   try/except 吞掉 → 整只股票当日返回 None（全 NaN）。本文件同样返回 None。
//! · 'all' 时段整体为空 → 直接返回 None。

use crate::data_paths;

/// 一天的纳秒数。
const DAY_NS: i64 = 86_400_000_000_000;
/// 滚动窗口 30 秒。
const PARAM_NS: i64 = 30_000_000_000;
/// pandas Timestamp("1970-01-01 08:00:00").value
const EXCHANGE_OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;

const MORNING_LO_NS: i64 = (9 * 3600 + 30 * 60) * 1_000_000_000;
const MORNING_HI_NS: i64 = (11 * 3600 + 30 * 60) * 1_000_000_000;
const AFTERNOON_LO_NS: i64 = (13 * 3600) * 1_000_000_000;
const AFTERNOON_HI_NS: i64 = (14 * 3600 + 57 * 60) * 1_000_000_000;
const ALL_HI_NS: i64 = (14 * 3600 + 57 * 60) * 1_000_000_000;

/// one() 的正常长度。
const ONE_LEN: usize = 65;
/// one() 在 begins/ends 为空时的长度（旧代码写的 66，照抄）。
const ONE_LEN_EMPTY: usize = 66;
/// sing() 的长度：65*3 + 9。
const SING_LEN: usize = ONE_LEN * 3 + 9;
/// 空时段 sing() 的长度：one(空) 返回 66 个 NaN，三个子样本各 66 个，再加 9 项。
const SING_LEN_EMPTY: usize = ONE_LEN_EMPTY * 3 + 9;
/// 每只股票的总长度：三个时段。
pub const N_OUT: usize = SING_LEN * 3;

/// 旧文件里的 65 个基础名（顺序 = one() 返回值的顺序）。
const BASE_NAMES: [&str; ONE_LEN] = [
    "begins_vr_max",
    "begins_vr_median",
    "begins_vr_autocorr1",
    "begins_vr_autocorr2",
    "begins_vr_autocorr3",
    "ends_vr_max",
    "ends_vr_median",
    "ends_vr_autocorr1",
    "ends_vr_autocorr2",
    "ends_vr_autocorr3",
    "vr_max_abs",
    "vr_median_abs",
    "vr_autocorr1_abs",
    "vr_autocorr2_abs",
    "vr_autocorr3_abs",
    "begins_times_mean",
    "begins_times_max",
    "begins_times_median",
    "begins_times_autocorr1",
    "begins_times_autocorr2",
    "begins_times_autocorr3",
    "ends_times_mean",
    "ends_times_max",
    "ends_times_median",
    "ends_times_autocorr1",
    "ends_times_autocorr2",
    "ends_times_autocorr3",
    "times_mean_abs",
    "times_max_abs",
    "times_median_abs",
    "times_autocorr1_abs",
    "times_autocorr2_abs",
    "times_autocorr3_abs",
    "times_mean",
    "times_max",
    "begins_price_trend",
    "begins_volume_trend",
    "begins_ba_trend",
    "begins_price_ba",
    "begins_rate_ba",
    "begins_price_times",
    "begins_times_trend",
    "begins_times_rate",
    "begins_times_ba",
    "ends_price_trend",
    "ends_volume_trend",
    "ends_ba_trend",
    "ends_price_ba",
    "ends_rate_ba",
    "ends_price_times",
    "ends_times_trend",
    "ends_times_rate",
    "ends_times_ba",
    "price_trend_abs",
    "volume_trend_abs",
    "ba_trend_abs",
    "price_ba_abs",
    "rate_ba_abs",
    "price_times_abs",
    "times_trend_abs",
    "times_rate_abs",
    "times_ba_abs",
    "times_trend",
    "times_rate",
    "times_ba",
];

/// sing() 末尾 8 个 6683_* 名字（旧文件里的标签，和实际取值有错位，照抄不动）。
const ABS_NAMES: [&str; 8] = [
    "6683_begins_times_median_abs",
    "6683_ends_times_median_abs",
    "6683_ends_times_autocorr2_abs",
    "6683_ends_times_autocorr3_abs",
    "6683_begins_price_times_abs",
    "6683_begins_times_trend_abs",
    "6683_ends_price_ba_abs",
    "6683_ends_times_trend_abs",
];

/// 补齐 612 用的 3 个名字（旧 names 只有 609 项，值有 612 项）。
/// 它们对应 afternoon 段 res4 的第 7/8/9 项，即
/// |begins_times_rate 差|、|ends_rate_ba 差|、|ends_times_rate 差|。
const EXTRA_NAMES: [&str; 3] = [
    "afternoon_6683_begins_times_rate_abs",
    "afternoon_6683_ends_rate_ba_abs",
    "afternoon_6683_ends_times_rate_abs",
];

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for blk in ["all", "morning", "afternoon"] {
        for sub in ["all", "66", "83"] {
            for b in BASE_NAMES.iter() {
                out.push(format!("{}_{}_{}", blk, sub, b));
            }
        }
        for a in ABS_NAMES.iter() {
            out.push(format!("{}_{}", blk, a));
        }
    }
    out.extend(EXTRA_NAMES.iter().map(|s| s.to_string()));
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

/// pandas 的 cast_from_unit_vectorized(values, "s", "ns") + Timestamp("1970-01-01 08:00:00")。
#[inline]
fn to_pandas_ns(exchtime_us: f64) -> i64 {
    let val = exchtime_us / 1e6;
    let base = val as i64; // C 的强制转换：向零截断
    let frac = val - base as f64;
    let f9 = rint(frac * 1e9); // np.round(frac, 9)
    let frac2 = f9 / 1e9;
    let ns = base * 1_000_000_000 + (frac2 * 1e9) as i64;
    ns + EXCHANGE_OFFSET_NS
}

/// numpy `aquicksort_<datetime_tag>`（npysort/quicksort.cpp）的逐行复刻，返回升序下标。
///
/// 与 hm013.rs / hm010.rs 里的同名函数一致：pandas 单键 sort_index / sort_values
/// 走 DatetimeArray.argsort('quicksort')，落到这条通用 quicksort（不是 int64 的
/// x86-simd-sort），并列顺序由它的分区过程决定，不能换成稳定排序。
fn np_argsort_quicksort(v: &[i64]) -> Vec<usize> {
    let num = v.len();
    let mut idx: Vec<usize> = (0..num).collect();
    if num <= 1 {
        return idx;
    }
    const SMALL_QUICKSORT: usize = 15;

    let mut stack: Vec<(usize, usize, i32)> = Vec::with_capacity(64);
    let mut pl = 0usize;
    let mut pr = num - 1;
    let mut cdepth: i32 = msb(num as u64) * 2;

    loop {
        if cdepth < 0 {
            aheapsort(v, &mut idx, pl, pr - pl + 1);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if v[idx[pm]] < v[idx[pl]] {
                    idx.swap(pm, pl);
                }
                if v[idx[pr]] < v[idx[pm]] {
                    idx.swap(pr, pm);
                }
                if v[idx[pm]] < v[idx[pl]] {
                    idx.swap(pm, pl);
                }
                let vp = v[idx[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                idx.swap(pm, pj);
                loop {
                    pi += 1;
                    while v[idx[pi]] < vp {
                        pi += 1;
                    }
                    pj -= 1;
                    while vp < v[idx[pj]] {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    idx.swap(pi, pj);
                }
                let pk = pr - 1;
                idx.swap(pi, pk);
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth - 1));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth - 1));
                    pl = pi + 1;
                }
                cdepth -= 1;
            }
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = idx[pi];
                let vp = v[vi];
                let mut pj = pi;
                let mut pk = pi - 1;
                while pj > pl && vp < v[idx[pk]] {
                    idx[pj] = idx[pk];
                    pj -= 1;
                    pk -= 1;
                }
                idx[pj] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            Some((npl, npr, d)) => {
                pl = npl;
                pr = npr;
                cdepth = d;
            }
            None => break,
        }
    }
    idx
}

fn msb(mut u: u64) -> i32 {
    let mut d = 0i32;
    while u > 1 {
        u >>= 1;
        d += 1;
    }
    d
}

/// numpy aheapsort_（索引版堆排序），只在 quicksort 深度耗尽时用。
fn aheapsort(v: &[i64], idx: &mut [usize], lo: usize, n: usize) {
    if n < 2 {
        return;
    }
    let mut n = n;
    let mut l = n >> 1;
    while l > 0 {
        let tmp = idx[lo + l - 1];
        let mut i = l;
        let mut j = l << 1;
        while j <= n {
            if j < n && v[idx[lo + j - 1]] < v[idx[lo + j]] {
                j += 1;
            }
            if v[tmp] < v[idx[lo + j - 1]] {
                idx[lo + i - 1] = idx[lo + j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        idx[lo + i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = idx[lo + n - 1];
        idx[lo + n - 1] = idx[lo];
        n -= 1;
        let mut i = 1usize;
        let mut j = 2usize;
        while j <= n {
            if j < n && v[idx[lo + j - 1]] < v[idx[lo + j]] {
                j += 1;
            }
            if v[tmp] < v[idx[lo + j - 1]] {
                idx[lo + i - 1] = idx[lo + j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        idx[lo + i - 1] = tmp;
    }
}

/// pandas 变量窗口 roll_sum 的逐位复刻（Kahan 补偿 + 增量 add/remove）。
///
/// 窗口：start[i] = 从 start[i-1] 起第一个 index[j] > index[i]-window 的 j（左开），
///       end[i] = i+1（右闭）。
fn roll_sum_var(vals: &[f64], idx: &[i64], window: i64) -> Vec<f64> {
    let n = idx.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let mut start = vec![0usize; n];
    let mut end = vec![0usize; n];
    start[0] = 0;
    end[0] = 1;
    for i in 1..n {
        let sb = idx[i] - window;
        let mut s = i;
        let mut j = start[i - 1];
        while j < i {
            if idx[j] > sb {
                s = j;
                break;
            }
            j += 1;
        }
        start[i] = s;
        end[i] = i + 1;
    }

    let mut sum_x = 0.0f64;
    let mut cadd = 0.0f64;
    let mut crem = 0.0f64;
    let mut nobs: i64 = 0;
    let mut prev_value = 0.0f64;
    let mut ncs: i64 = 0;

    #[inline]
    fn add(val: f64, nobs: &mut i64, sum_x: &mut f64, comp: &mut f64, ncs: &mut i64, prev: &mut f64) {
        if val == val {
            *nobs += 1;
            let y = val - *comp;
            let t = *sum_x + y;
            *comp = t - *sum_x - y;
            *sum_x = t;
            if val == *prev {
                *ncs += 1;
            } else {
                *ncs = 1;
            }
            *prev = val;
        }
    }
    #[inline]
    fn rem(val: f64, nobs: &mut i64, sum_x: &mut f64, comp: &mut f64) {
        if val == val {
            *nobs -= 1;
            let y = -val - *comp;
            let t = *sum_x + y;
            *comp = t - *sum_x - y;
            *sum_x = t;
        }
    }

    for i in 0..n {
        let s = start[i];
        let e = end[i];
        if i == 0 || s >= end[i - 1] {
            prev_value = if e > s { vals[s] } else { 0.0 };
            ncs = 0;
            sum_x = 0.0;
            cadd = 0.0;
            crem = 0.0;
            nobs = 0;
            for j in s..e {
                add(vals[j], &mut nobs, &mut sum_x, &mut cadd, &mut ncs, &mut prev_value);
            }
        } else {
            for j in start[i - 1]..s {
                rem(vals[j], &mut nobs, &mut sum_x, &mut crem);
            }
            for j in end[i - 1]..e {
                add(vals[j], &mut nobs, &mut sum_x, &mut cadd, &mut ncs, &mut prev_value);
            }
        }
        out[i] = if nobs == 0 {
            0.0
        } else if ncs >= nobs {
            prev_value * nobs as f64
        } else {
            sum_x
        };
    }
    out
}

/// numpy `_lerp`：t < 0.5 用 a + (b-a)*t，否则用 b - (b-a)*(1-t)。
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
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
    let l = lo as usize;
    let h = hi as usize;
    let gamma = vi - lo;
    lerp(sorted[l], sorted[h], gamma)
}

/// pandas 的跳过 NaN 的均值。
fn nan_mean(v: &[f64]) -> f64 {
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

fn nan_max(v: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &x in v {
        if x == x && (m != m || x > m) {
            m = x;
        }
    }
    m
}

/// pandas 的跳过 NaN 的中位数（偶数个取中间两个的平均）。
fn nan_median(v: &[f64]) -> f64 {
    let mut a: Vec<f64> = v.iter().cloned().filter(|x| x == x).collect();
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

/// libalgos.nancorr 的 Welford 版本（DataFrame.corr 走这条）。
/// 成对完整观测用 isfinite 判（inf 也会被丢掉），nobs<1 或分母为 0 返回 NaN。
fn welford_corr(a: &[f64], b: &[f64]) -> f64 {
    let mut nobs: i64 = 0;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    for i in 0..a.len() {
        if !a[i].is_finite() || !b[i].is_finite() {
            continue;
        }
        let vx = a[i];
        let vy = b[i];
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

/// Series.corr → np.corrcoef（两遍法）：先按 notna 成对剔除，再算，最后 clip 到 [-1,1]。
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
    let mx = mean_pairwise(&xs);
    let my = mean_pairwise(&ys);
    let (mut c00, mut c11, mut c01) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        c00 += dx * dx;
        c11 += dy * dy;
        c01 += dx * dy;
    }
    let fact = (n - 1) as f64;
    c00 /= fact;
    c11 /= fact;
    c01 /= fact;
    let s0 = c00.sqrt();
    let s1 = c11.sqrt();
    let mut r = c01 / s0 / s1;
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
        let mut r: [f64; 8] = [0.0; 8];
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

fn mean_pairwise(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(v) / v.len() as f64
}

/// Series.autocorr(lag)：corr(self, self.shift(lag))，shift 出来的前 lag 个是 NaN。
fn autocorr(v: &[f64], lag: usize) -> f64 {
    let n = v.len();
    if lag >= n {
        // 旧代码会抛 ValueError，调用方负责在更外层处理
        return f64::NAN;
    }
    let mut shifted = vec![f64::NAN; n];
    shifted[lag..].copy_from_slice(&v[..n - lag]);
    pearson_corrcoef(v, &shifted)
}

// ============================================================================
// one()：一个子样本（全部 / 66 / 83）× 一个时段的 65（或 66）个指标
// ============================================================================

/// 一个「秒」的聚合结果。
struct Grouped {
    keys: Vec<i64>,
    vol: Vec<f64>,
    price: Vec<f64>,
    before: Vec<f64>,
    after: Vec<f64>,
    ba: Vec<f64>,
}

/// 按纳秒 key 聚合并算滚动量。
fn group_and_roll(ns: &[i64], vol: &[f64], tov: &[f64]) -> Grouped {
    let n = ns.len();
    let mut keys: Vec<i64> = Vec::new();
    let mut vols: Vec<f64> = Vec::new();
    let mut tovs: Vec<f64> = Vec::new();
    if n > 0 {
        // groupby(sort=True)：pandas 对唯一 key 的结果就是升序，用 numpy 的排序复刻
        let perm = np_argsort_quicksort(ns);
        let mut i = 0usize;
        while i < perm.len() {
            let k = ns[perm[i]];
            let mut sv = 0.0f64;
            let mut st = 0.0f64;
            let mut j = i;
            while j < perm.len() && ns[perm[j]] == k {
                // pandas 的 groupby().sum() 跳过 NaN（skipna 默认 True）
                if vol[perm[j]] == vol[perm[j]] {
                    sv += vol[perm[j]];
                }
                if tov[perm[j]] == tov[perm[j]] {
                    st += tov[perm[j]];
                }
                j += 1;
            }
            keys.push(k);
            vols.push(sv);
            tovs.push(st);
            i = j;
        }
    }
    let m = keys.len();
    let price: Vec<f64> = (0..m).map(|i| tovs[i] / vols[i]).collect();
    let before = roll_sum_var(&vols, &keys, PARAM_NS);

    // concat([label=0 的原帧, label=1 的平移帧]).sort_index()
    let mut idx2: Vec<i64> = Vec::with_capacity(2 * m);
    idx2.extend_from_slice(&keys);
    for k in keys.iter() {
        idx2.push(k - PARAM_NS);
    }
    let mut val: Vec<f64> = vec![0.0; m];
    val.extend_from_slice(&vols);
    let mut tag: Vec<u8> = vec![0u8; m];
    tag.extend(std::iter::repeat(1u8).take(m));

    let order = np_argsort_quicksort(&idx2);
    let idx_s: Vec<i64> = order.iter().map(|&p| idx2[p]).collect();
    let val_s: Vec<f64> = order.iter().map(|&p| val[p]).collect();
    let tag_s: Vec<u8> = order.iter().map(|&p| tag[p]).collect();

    let araw = roll_sum_var(&val_s, &idx_s, PARAM_NS);
    let mut after: Vec<f64> = Vec::with_capacity(m);
    let mut cnt = 0usize;
    for i in 0..tag_s.len() {
        if tag_s[i] == 0 {
            after.push(araw[i] + vols[cnt]);
            cnt += 1;
        }
    }
    debug_assert_eq!(cnt, m);
    let ba: Vec<f64> = (0..m).map(|i| before[i] / after[i]).collect();
    Grouped {
        keys,
        vol: vols,
        price,
        before,
        after,
        ba,
    }
}

/// 一个时段的一组列（begins 或 ends）。
struct Side {
    price: Vec<f64>,
    trend: Vec<f64>,
    vr: Vec<f64>,
    ba: Vec<f64>,
    times: Vec<f64>,
}

fn build_side(g: &Grouped, sel: &[usize], use_after: bool) -> Side {
    let n = sel.len();
    let mut price = Vec::with_capacity(n);
    let mut vr = Vec::with_capacity(n);
    let mut ba = Vec::with_capacity(n);
    let mut times = vec![f64::NAN; n];
    let mut trend = Vec::with_capacity(n);
    for (k, &i) in sel.iter().enumerate() {
        price.push(g.price[i]);
        let denom = if use_after { g.after[i] } else { g.before[i] };
        vr.push(g.vol[i] / denom);
        ba.push(g.ba[i]);
        trend.push(k as f64);
        if k > 0 {
            times[k] = (g.keys[i] - g.keys[sel[k - 1]]) as f64 / 1e9;
        }
    }
    Side {
        price,
        trend,
        vr,
        ba,
        times,
    }
}

/// one(df)：65 项；begins/ends 任一为空时返回 66 个 NaN（旧代码如此）。
/// 返回 None 表示旧代码会抛异常（autocorr 的 lag 检查），整只股票当日作废。
fn one(ns: &[i64], vol: &[f64], tov: &[f64]) -> Option<Vec<f64>> {
    let g = group_and_roll(ns, vol, tov);
    let m = g.keys.len();
    if m == 0 {
        return Some(vec![f64::NAN; ONE_LEN_EMPTY]);
    }
    let mut fin: Vec<f64> = g.ba.iter().cloned().filter(|x| x.is_finite()).collect();
    if fin.is_empty() {
        return Some(vec![f64::NAN; ONE_LEN_EMPTY]);
    }
    fin.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q10 = percentile_linear_sorted(&fin, 0.1);
    let q90 = percentile_linear_sorted(&fin, 0.9);
    let bi: Vec<usize> = (0..m).filter(|&i| g.ba[i] == g.ba[i] && g.ba[i] <= q10).collect();
    let ei: Vec<usize> = (0..m).filter(|&i| g.ba[i] == g.ba[i] && g.ba[i] >= q90).collect();
    if bi.is_empty() || ei.is_empty() {
        return Some(vec![f64::NAN; ONE_LEN_EMPTY]);
    }
    // 旧代码：begins.volume_rate.autocorr(1..3) / ends... 的 lag 检查
    if bi.len() < 4 || ei.len() < 4 {
        return None;
    }

    let bs = build_side(&g, &bi, true);
    let es = build_side(&g, &ei, false);

    let b_vr_max = nan_max(&bs.vr);
    let b_vr_med = nan_median(&bs.vr);
    let b_vr_a1 = autocorr(&bs.vr, 1);
    let b_vr_a2 = autocorr(&bs.vr, 2);
    let b_vr_a3 = autocorr(&bs.vr, 3);
    let e_vr_max = nan_max(&es.vr);
    let e_vr_med = nan_median(&es.vr);
    let e_vr_a1 = autocorr(&es.vr, 1);
    let e_vr_a2 = autocorr(&es.vr, 2);
    let e_vr_a3 = autocorr(&es.vr, 3);

    let b_tm_mean = nan_mean(&bs.times);
    let b_tm_max = nan_max(&bs.times);
    let b_tm_med = nan_median(&bs.times);
    let b_tm_a1 = autocorr(&bs.times, 1);
    let b_tm_a2 = autocorr(&bs.times, 2);
    let b_tm_a3 = autocorr(&bs.times, 3);
    let e_tm_mean = nan_mean(&es.times);
    let e_tm_max = nan_max(&es.times);
    let e_tm_med = nan_median(&es.times);
    let e_tm_a1 = autocorr(&es.times, 1);
    let e_tm_a2 = autocorr(&es.times, 2);
    let e_tm_a3 = autocorr(&es.times, 3);

    // DataFrame.corr()：列序 [price, trend, volume_rate, ba, times]。
    // libalgos.nancorr 只算 yi <= xi 那一半，再把值对称复制；所以读 {a,b} 这一对时
    // 必须拿**下标大的那列当 x**（Welford 里 covxy += (vx-meanx_new)*dy_old，
    // 两列换序结果不一样，实测差 1e-3 量级）。
    let cols_b: [&Vec<f64>; 5] = [&bs.price, &bs.trend, &bs.vr, &bs.ba, &bs.times];
    let cols_e: [&Vec<f64>; 5] = [&es.price, &es.trend, &es.vr, &es.ba, &es.times];
    let cb = |a: usize, b: usize| -> f64 {
        let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
        welford_corr(cols_b[hi], cols_b[lo])
    };
    let ce = |a: usize, b: usize| -> f64 {
        let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
        welford_corr(cols_e[hi], cols_e[lo])
    };

    let res = vec![
        b_vr_max,
        b_vr_med,
        b_vr_a1,
        b_vr_a2,
        b_vr_a3,
        e_vr_max,
        e_vr_med,
        e_vr_a1,
        e_vr_a2,
        e_vr_a3,
        (b_vr_max - e_vr_max).abs(),
        (b_vr_med - e_vr_med).abs(),
        (b_vr_a1 - e_vr_a1).abs(),
        (b_vr_a2 - e_vr_a2).abs(),
        (b_vr_a3 - e_vr_a3).abs(),
        b_tm_mean,
        b_tm_max,
        b_tm_med,
        b_tm_a1,
        b_tm_a2,
        b_tm_a3,
        e_tm_mean,
        e_tm_max,
        e_tm_med,
        e_tm_a1,
        e_tm_a2,
        e_tm_a3,
        (b_tm_mean - e_tm_mean).abs(),
        (b_tm_max - e_tm_max).abs(),
        (b_tm_med - e_tm_med).abs(),
        (b_tm_a1 - e_tm_a1).abs(),
        (b_tm_a2 - e_tm_a2).abs(),
        (b_tm_a3 - e_tm_a3).abs(),
        b_tm_mean - e_tm_mean,
        b_tm_max - e_tm_max,
        cb(0, 1),
        cb(1, 2),
        cb(1, 3),
        cb(0, 3),
        cb(2, 3),
        cb(0, 4),
        cb(4, 1),
        cb(4, 2),
        cb(4, 3),
        ce(0, 1),
        ce(1, 2),
        ce(1, 3),
        ce(0, 3),
        ce(2, 3),
        ce(0, 4),
        ce(4, 1),
        ce(4, 2),
        ce(4, 3),
        (cb(0, 1) - ce(0, 1)).abs(),
        (cb(1, 2) - ce(1, 2)).abs(),
        (cb(1, 3) - ce(1, 3)).abs(),
        (cb(0, 3) - ce(0, 3)).abs(),
        (cb(2, 3) - ce(2, 3)).abs(),
        (cb(0, 4) - ce(0, 4)).abs(),
        (cb(4, 1) - ce(4, 1)).abs(),
        (cb(4, 2) - ce(4, 2)).abs(),
        (cb(4, 3) - ce(4, 3)).abs(),
        cb(4, 1) - ce(4, 1),
        cb(4, 2) - ce(4, 2),
        cb(4, 3) - ce(4, 3),
    ];
    debug_assert_eq!(res.len(), ONE_LEN);
    Some(res)
}

/// sing(df)：res1 + res2 + res3 + res4（9 项）。
fn sing(ns: &[i64], vol: &[f64], tov: &[f64], flag: &[i32]) -> Option<Vec<f64>> {
    let pick = |f: i32| -> (Vec<i64>, Vec<f64>, Vec<f64>) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for i in 0..ns.len() {
            if flag[i] == f {
                a.push(ns[i]);
                b.push(vol[i]);
                c.push(tov[i]);
            }
        }
        (a, b, c)
    };
    let r1 = one(ns, vol, tov)?;
    let (n66, v66, t66) = pick(66);
    let r2 = one(&n66, &v66, &t66)?;
    let (n83, v83, t83) = pick(83);
    let r3 = one(&n83, &v83, &t83)?;
    let mut out = Vec::with_capacity(SING_LEN);
    out.extend_from_slice(&r1);
    out.extend_from_slice(&r2);
    out.extend_from_slice(&r3);
    for k in [17usize, 23, 24, 25, 26, 41, 42, 48, 51] {
        let x = if k < r2.len() { r2[k] } else { f64::NAN };
        let y = if k < r3.len() { r3[k] } else { f64::NAN };
        out.push((x - y).abs());
    }
    Some(out)
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

/// 取字段的整数前缀：`"32.0"` 给 32，与 `df.flag != 32` 的数值比较一致。
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

/// pandas C 解析器 precise_xstrtod 的逐行复刻（同 hm033.rs / hm090.rs）。
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
    // 指数段（本数据没有 'e'，按源码照抄）
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

/// 一天的逐笔（已丢 flag==32），返回 (纳秒, volume, turnover, flag)。
fn read_day(code: &str, date: i64) -> Option<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i32>)> {
    const COL_EXCHTIME: usize = 4;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;

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
    let mut vol_out: Vec<f64> = Vec::new();
    let mut tov_out: Vec<f64> = Vec::new();
    let mut flag_out: Vec<i32> = Vec::new();

    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < 15 {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < 15 {
            fields[commas] = &line[start..];
        }
        if commas + 1 < 15 {
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
        vol_out.push(num(fields[COL_VOLUME]));
        tov_out.push(num(fields[COL_TURNOVER]));
        flag_out.push(flag);
    }
    Some((ns_out, vol_out, tov_out, flag_out))
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 逐股系列：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let (ns, vol, tov, flag) = read_day(code, date)?;
    if ns.is_empty() {
        return None;
    }
    // 三个时段：all = [09:30:00, 14:57:00]，morning = [09:30:00, 11:30:00]，
    // afternoon = [13:00:00, 14:57:00]，都是**毫秒级闭区间**（按纳秒比）。
    let tod: Vec<i64> = ns.iter().map(|&x| x.rem_euclid(DAY_NS)).collect();
    let idx_all: Vec<usize> = (0..ns.len())
        .filter(|&i| (MORNING_LO_NS..=ALL_HI_NS).contains(&tod[i]))
        .collect();
    if idx_all.is_empty() {
        return None;
    }
    let idx_mor: Vec<usize> = idx_all
        .iter()
        .cloned()
        .filter(|&i| tod[i] <= MORNING_HI_NS)
        .collect();
    let idx_aft: Vec<usize> = (0..ns.len())
        .filter(|&i| (AFTERNOON_LO_NS..=AFTERNOON_HI_NS).contains(&tod[i]))
        .collect();

    let take = |sel: &Vec<usize>| -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<i32>) {
        (
            sel.iter().map(|&i| ns[i]).collect(),
            sel.iter().map(|&i| vol[i]).collect(),
            sel.iter().map(|&i| tov[i]).collect(),
            sel.iter().map(|&i| flag[i]).collect(),
        )
    };

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for sel in [&idx_all, &idx_mor, &idx_aft] {
        let (a, b, c, d) = take(sel);
        if a.is_empty() {
            // 空时段：旧代码里 df.shape[0] == 0 的时段不会进 sing（直接不贡献），
            // 但 all 段为空会整体返回 None（上面已判）。
            // between_time 出来为空但 all 非空的只有 morning/afternoon，
            // 这时 sing 会拿到空 df：one(空) 返回 66 个 NaN，三个子样本各 66 个。
            out.extend(std::iter::repeat(f64::NAN).take(SING_LEN_EMPTY));
            continue;
        }
        let r = sing(&a, &b, &c, &d)?;
        out.extend_from_slice(&r);
    }
    // lst = [np.nan if np.isinf(x) else x for x in lst]
    let out = out
        .into_iter()
        .map(|x| if x.is_infinite() { f64::NAN } else { x })
        .collect::<Vec<f64>>();
    Some(crate::okay_database::fit(
        out.into_iter().map(|x| x as f32).collect(),
        N_OUT,
    ))
}

/// hm024 是逐股系列，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
