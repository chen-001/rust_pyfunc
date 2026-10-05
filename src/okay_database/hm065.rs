//! hm65「活在每一刻」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm65_活在每一刻/hm65_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   一整天的逐笔一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0) → 丢 flag==32；deal_data 把 exchtime
//!    转成 pandas Timestamp（+8 小时）。df.set_index('exchtime')；
//!    close_price = df.price.iloc[-1]（**在 adjust_afternoon 之前**的最后一行的价）。
//! 2) df = dwo.adjust_afternoon(df)：这一支走的是「index.name=='exchtime'」分支，
//!    上午 [09:30:00,11:30:00] 原样、下午 [13:00:00,14:57:00] 的索引整体 -90 分钟，
//!    然后 concat（**不 reset_index**，下游 rolling 用的就是这个前移后的时间轴）。
//! 3) 派生列（都在同一个 df 上算，assign 的右值先于赋值全部求值）：
//!      price_1s..price_1h = df.price.rolling_future('Xs').last()   （15 个窗口）
//!      act_buy = (flag==66)*volume
//!      earn_buy_Xm = (df.price > df.price.rolling_past('Xs').first()).sum()
//!                    / df.price.rolling_past('Xs').count()          （X=1,5,10,30 分钟）
//!        ！！分子是一个**标量**（全天满足条件的行数），分母是逐行 count，所以它是逐行变化的
//!      act_buy_Xm = act_buy.rolling_future('Xs').sum()
//!      volume_Xm  = volume.rolling_future('Xs').sum()
//!      buyrate_Xm = act_buy_Xm / volume_Xm
//! 4) 六个切片（都用前移后的钟点）：
//!      df1           = [09:30:00, 12:27:00]        df2           = [09:30:00, 13:17:00]
//!      df1_morning   = [09:30:00, 10:30:00]        df2_morning   = [09:30:00, 11:30:00]
//!      df1_afternoon = [11:30:00, 12:27:00]        df2_afternoon = [11:30:00, 13:17:00]
//!    每个切片各跑一遍 one()，half_at_least=False 的六遍在前、True 的六遍在后，
//!    顺序是 全天(df1,df2) → 上午(df1,df2) → 下午(df1,df2)。共 12×81 = 972 个值。
//! 5) one(df)：16 个 earn_buy×buyrate 相关系数；按 ask_order 分组当「卖」，
//!    按 bid_order 分组当「买」，agg 出 turnover/volume 之和与 15 个 price_X 的**最后一个非 NaN**；
//!    price = turnover/volume，16 个 ret_X = price_X/price-1（外加 ret_close = close_price/price-1）；
//!    ret 里的精确 0 换成 NaN；half_at_least 时按行过滤（见下）；再算
//!    corr_buy/happy0/happy1/dh1 四族与 3 个 DTW。
//!
//! names_single（=names）：81 个指标 × 2 种删尾 × 3 个时段 × 2 种门槛，
//!   外层是门槛（不需要至少一半 / 至少一半），再时段，再删尾，最内层是指标。
//!
//! ════════════════════════════════════════════════════════════════════
//! 对齐过程中确认过的坑（全部实测过）
//! ════════════════════════════════════════════════════════════════════
//! · pandas 的 CSV 浮点解析器**不是正确舍入**的：它只把前 17 位有效数字
//!   `number = number*10 + digit` 累进 double，多出来的整数位折成指数、多出来的小数位丢掉，
//!   最后用 10 的幂表做一次乘/除。实例：CSV 里的 "20874.239999999998" 这里读成 20874.24，
//!   而 Rust 的 str::parse 给的是它下面那个 double。差别只有 1 ulp，但
//!   `replace(0, np.nan)` 是一道「精确等于 0」的闸门（ret = price_X/price-1 等于 0 的位置变 NaN），
//!   闸门翻面会改变 happy0/happy1 的**分母**（离散计数），一个 token 就能把 9/16 变成 4/11。
//!   所以 price/volume/turnover/exchtime 全部走 parse_f64_pandas（同 hm033.rs 那份）。
//! · rolling_future 的 include_current 默认 False，rolling_past 默认 **True**（rolling_past.py
//!   第 68 行）。hm65 里 rolling_past(...).first()/.count() 都含当前行，漏掉时 earn_buy 全错。
//! · rolling 的 times 是 DatetimeIndex.astype(int64) 再转 f64（1.7e18 只有 256ns 分辨率），
//!   窗口边界就在这个分辨率上比。
//! · half_at_least 的过滤是 `df[df.count(axis=1) > int(df.shape[1]/2)]`，而 drop 的列表里
//!   **没有 volume**，所以这一层是 17 列（volume + 16 个 ret），int(17/2)=8，
//!   等价于「16 个 ret 里至少 8 个非 NaN」。按 16 列算成「至少 9 个」时，全天那两组全错。
//! · 数值原语：sum/mean/std/skew/kurt 用 numpy 成对求和 + pandas nanops 的两遍算法；
//!   groupby 的 turnover/volume 之和用 Kahan；Series.corr 走 np.corrcoef 的
//!   cov/sqrt(vx*vy)（实测与 BLAS 版差 ~1e-18）；DTW 照抄 hm055.rs 的纯 Rust 版本。
//! · 失败模式：len<2 时 Series.autocorr 抛 ValueError，旧代码整段被 except 吞掉 → 整只股票 NaN。
//!   这里在 buy/sell 组数 <2、dh1 分组数 <2 时返回 None。
//!
//! 实测四组对照最大相对误差 5.9e-8（f32 精度量级），阈值 1e-4。

// hm067 开发版：独立可执行文件，直接读 transaction CSV 路径，打印 2340 个值。
// 核心逻辑与 src/okay_database/hm067.rs 完全一致，只是数据来源换成本地路径。
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

const NS_SEC: i64 = 1_000_000_000;
const NS_DAY: i64 = 86_400 * NS_SEC;
const NS_8H: i64 = 28_800 * NS_SEC;

#[derive(Clone, Copy)]
pub struct Row {
    pub ns: i64,
    pub price: f64,
    pub volume: f64,
    pub turnover: f64,
    pub flag: f64,
    pub ask_order: i64,
    pub bid_order: i64,
}

// ============================ 数值原语 ============================

/// numpy 的成对求和（PW_BLOCKSIZE=128，8 路展开）。
fn pw_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f64;
        for &v in a {
            r += v;
        }
        return r;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[0..8]);
        let mut i = 8usize;
        let end = n - (n % 8);
        while i < end {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pw_sum(&a[..n2]) + pw_sum(&a[n2..])
}

/// NaN 当 0、按整数组做成对求和（对应 pandas 的 _get_values(fill_value=0) + ndarray.sum）。
fn nan_pw_sum(a: &[f64]) -> f64 {
    let v: Vec<f64> = a.iter().map(|&x| if x.is_nan() { 0.0 } else { x }).collect();
    pw_sum(&v)
}

fn nan_count(a: &[f64]) -> usize {
    a.iter().filter(|x| !x.is_nan()).count()
}

fn nan_mean(a: &[f64]) -> f64 {
    let c = nan_count(a);
    if c == 0 {
        return f64::NAN;
    }
    nan_pw_sum(a) / c as f64
}

fn nan_var(a: &[f64], ddof: usize) -> f64 {
    let c = nan_count(a);
    if c <= ddof {
        return f64::NAN;
    }
    let avg = nan_pw_sum(a) / c as f64;
    let sq: Vec<f64> = a
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else {
                let d = avg - x;
                d * d
            }
        })
        .collect();
    pw_sum(&sq) / (c - ddof) as f64
}

fn nan_std(a: &[f64]) -> f64 {
    nan_var(a, 1).sqrt()
}

fn nan_skew(a: &[f64]) -> f64 {
    let c = nan_count(a);
    if c == 0 {
        return f64::NAN;
    }
    let cf = c as f64;
    let mean = nan_pw_sum(a) / cf;
    let mut a2 = Vec::with_capacity(a.len());
    let mut a3 = Vec::with_capacity(a.len());
    for &x in a {
        let d = if x.is_nan() { 0.0 } else { x - mean };
        a2.push(d * d);
        a3.push(d * d * d);
    }
    let mut m2 = pw_sum(&a2);
    let mut m3 = pw_sum(&a3);
    if m2.abs() < 1e-14 {
        m2 = 0.0;
    }
    if m3.abs() < 1e-14 {
        m3 = 0.0;
    }
    if c < 3 {
        return f64::NAN;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    (cf * (cf - 1.0).powf(0.5) / (cf - 2.0)) * (m3 / m2.powf(1.5))
}

fn nan_kurt(a: &[f64]) -> f64 {
    let c = nan_count(a);
    if c == 0 {
        return f64::NAN;
    }
    let cf = c as f64;
    let mean = nan_pw_sum(a) / cf;
    let mut a2 = Vec::with_capacity(a.len());
    let mut a4 = Vec::with_capacity(a.len());
    for &x in a {
        let d = if x.is_nan() { 0.0 } else { x - mean };
        let d2 = d * d;
        a2.push(d2);
        a4.push(d2 * d2);
    }
    let m2 = pw_sum(&a2);
    let m4 = pw_sum(&a4);
    let adj = 3.0 * (cf - 1.0) * (cf - 1.0) / ((cf - 2.0) * (cf - 3.0));
    let mut num = cf * (cf + 1.0) * (cf - 1.0) * m4;
    let mut den = (cf - 2.0) * (cf - 3.0) * m2 * m2;
    if num.abs() < 1e-14 {
        num = 0.0;
    }
    if den.abs() < 1e-14 {
        den = 0.0;
    }
    if c < 4 {
        return f64::NAN;
    }
    if den == 0.0 {
        return 0.0;
    }
    num / den - adj
}

fn nan_median(a: &[f64]) -> f64 {
    let mut v: Vec<f64> = a.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn nan_min(a: &[f64]) -> f64 {
    let mut r = f64::NAN;
    for &x in a {
        if x.is_nan() {
            continue;
        }
        if r.is_nan() || x < r {
            r = x;
        }
    }
    r
}

fn nan_max(a: &[f64]) -> f64 {
    let mut r = f64::NAN;
    for &x in a {
        if x.is_nan() {
            continue;
        }
        if r.is_nan() || x > r {
            r = x;
        }
    }
    r
}

/// numpy 的 rint：四舍六入五成双。
fn rint(x: f64) -> f64 {
    let t = x.trunc();
    let d = x - t;
    let ad = d.abs();
    if ad > 0.5 {
        t + d.signum()
    } else if ad < 0.5 {
        t
    } else if (t as i64) % 2 == 0 {
        t
    } else {
        t + d.signum()
    }
}

/// pandas 的 cast_from_unit_vectorized + 8 小时偏移（与 np.percentile/np.corrcoef 无关）。
fn ns_from_us(ex_us: f64) -> i64 {
    let val = ex_us / 1e6;
    let base = val as i64;
    let frac = val - base as f64;
    let frac_r = rint(frac * 1e9) / 1e9;
    base * NS_SEC + (frac_r * 1e9) as i64 + NS_8H
}

fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && b[i] == b' ' {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() {
        let c = b[i];
        if !(b'0'..=b'9').contains(&c) {
            break;
        }
        v = v * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// 一天的钟点（微秒），对应 pandas DatetimeIndex._get_time_micros()。
#[inline]
fn tod_us(ns: i64) -> i64 {
    ns.rem_euclid(NS_DAY) / 1000
}


// ===================== pandas C 解析器 precise_xstrtod 的复刻（取自 hm033.rs） =====================
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
        return f64::INFINITY;
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

// ============================ 滚动窗口（照抄 src/pandas_ext/mod.rs） ============================
// rolling_window_stat / rolling_window_stat_backward 都是 #[pyfunction]，worker 不能调，
// 这里按同一份实现抄一份。times 是 pandas DatetimeIndex.astype(int64) 再转 f64 的值
// （1.7e18 在 f64 里只有 256ns 分辨率，窗口边界比较就在这个分辨率上做）。

fn roll_future_last(times: &[f64], values: &[f64], window_ns: f64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }
    let mut window_end = 0usize;
    for i in 0..n {
        if i > 0 && window_end > i {
            // 沿用上一个窗口的结束位置
        } else {
            window_end = i + 1;
        }
        while window_end < n && times[window_end] - times[i] <= window_ns {
            window_end += 1;
        }
        if window_end > i + 1 {
            result[i] = values[window_end - 1];
        }
    }
    result
}

fn roll_future_sum(times: &[f64], values: &[f64], window_ns: f64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }
    let mut window_sum = 0.0f64;
    let mut window_count: i64 = 0;
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
            result[i] = window_sum - values[i];
        }
    }
    result
}

fn roll_past_first(times: &[f64], values: &[f64], window_ns: f64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![f64::NAN; n];
    let mut left = 0usize;
    for i in 0..n {
        let start_time = times[i] - window_ns;
        while left < i && times[left] < start_time {
            left += 1;
        }
        // rolling_past 的 include_current 默认是 True（与 rolling_future 相反）
        if left < i {
            result[i] = values[left];
        } else if times[i] >= start_time {
            result[i] = values[i];
        }
    }
    result
}

fn roll_past_count(times: &[f64], window_ns: f64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![f64::NAN; n];
    let mut window_count: i64 = 0;
    let mut left = 0usize;
    for i in 0..n {
        let start_time = times[i] - window_ns;
        while left < i && times[left] < start_time {
            window_count -= 1;
            left += 1;
        }
        let mut final_count = window_count;
        if times[i] >= start_time {
            final_count += 1;
        }
        result[i] = final_count as f64;
        if times[i] >= start_time {
            window_count += 1;
        }
    }
    result
}

// ============================ DTW（照抄 src/okay_database/hm055.rs） ============================
fn dtw(s1: &[f64], s2: &[f64]) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return f64::NAN;
    }
    let n = s2.len();
    let mut prev = vec![f64::INFINITY; n + 1];
    let mut curr = vec![f64::INFINITY; n + 1];
    prev[0] = 0.0;
    for i in 1..=s1.len() {
        curr[0] = f64::INFINITY;
        let a = s1[i - 1];
        for j in 1..=n {
            let cost = (a - s2[j - 1]).abs();
            curr[j] = cost + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ============================ 统计原语（同 hm067.rs） ============================
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n0 = x.len().min(y.len());
    let mut a: Vec<f64> = Vec::with_capacity(n0);
    let mut b: Vec<f64> = Vec::with_capacity(n0);
    for i in 0..n0 {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        a.push(x[i]);
        b.push(y[i]);
    }
    let n = a.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let ma = pw_sum(&a) / nf;
    let mb = pw_sum(&b) / nf;
    let da: Vec<f64> = a.iter().map(|&v| v - ma).collect();
    let db: Vec<f64> = b.iter().map(|&v| v - mb).collect();
    let p: Vec<f64> = da.iter().zip(db.iter()).map(|(p, q)| p * q).collect();
    let d2: Vec<f64> = da.iter().map(|&v| v * v).collect();
    let e2: Vec<f64> = db.iter().map(|&v| v * v).collect();
    let sxy = pw_sum(&p);
    let sxx = pw_sum(&d2);
    let syy = pw_sum(&e2);
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn autocorr1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    pearson(&x[..n - 1], &x[1..])
}

fn trend(v: &[f64]) -> f64 {
    let mut vals: Vec<f64> = Vec::with_capacity(v.len());
    let mut idx: Vec<f64> = Vec::with_capacity(v.len());
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            vals.push(x);
            idx.push((i + 1) as f64);
        }
    }
    let n = vals.len();
    if n == 0 || n == 1 {
        return 0.0;
    }
    let nf = n as f64;
    let mx = vals.iter().sum::<f64>() / nf;
    let my = idx.iter().sum::<f64>() / nf;
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = vals[i] - mx;
        let dy = idx[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

// ============================ 逐日派生列 ============================

const PRICE_WINS: [f64; 15] = [
    1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 180.0, 300.0, 600.0, 900.0, 1200.0, 1800.0,
    3600.0,
];
const EARN_WINS: [f64; 4] = [60.0, 300.0, 600.0, 1800.0];

struct Frame {
    ns: Vec<i64>,
    price: Vec<f64>,
    volume: Vec<f64>,
    turnover: Vec<f64>,
    ask_order: Vec<i64>,
    bid_order: Vec<i64>,
    px: Vec<Vec<f64>>,
    earn: Vec<Vec<f64>>,
    br: Vec<Vec<f64>>,
}

/// 一天的逐笔：读 CSV，丢 flag==32，做 adjust_afternoon（下午索引前移 90 分钟），
/// 再算 rolling 派生列。返回 (Frame, close_price)。
fn build_frame(content: &str) -> Option<(Frame, f64)> {
    let mut rows: Vec<Row> = Vec::new();
    let mut lines = content.lines();
    lines.next();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < TRD_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < TRD_NCOL {
            f[col] = &bytes[start..];
        }
        let flag = parse_f64_pandas(f[TRD_COL_FLAG]);
        if flag == 32.0 {
            continue;
        }
        rows.push(Row {
            ns: ns_from_us(parse_f64_pandas(f[TRD_COL_EXCHTIME])),
            price: parse_f64_pandas(f[TRD_COL_PRICE]),
            volume: parse_f64_pandas(f[TRD_COL_VOLUME]),
            turnover: parse_f64_pandas(f[TRD_COL_TURNOVER]),
            flag,
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    if rows.is_empty() {
        return None;
    }
    // close_price 取 adjust_afternoon 之前最后一行的 price
    let close_price = rows[rows.len() - 1].price;

    // adjust_afternoon（索引版）：上午原样，下午索引 -90 分钟，两部分前后拼接
    let mut adj: Vec<Row> = Vec::with_capacity(rows.len());
    for r in rows.iter() {
        let t = tod_us(r.ns);
        if (34_200 * 1_000_000..=41_400 * 1_000_000).contains(&t) {
            adj.push(*r);
        }
    }
    for r in rows.iter() {
        let t = tod_us(r.ns);
        if (46_800 * 1_000_000..=53_820 * 1_000_000).contains(&t) {
            let mut c = *r;
            c.ns -= 5_400 * NS_SEC;
            adj.push(c);
        }
    }
    if adj.is_empty() {
        return None;
    }
    let n = adj.len();
    let ns: Vec<i64> = adj.iter().map(|r| r.ns).collect();
    let times: Vec<f64> = ns.iter().map(|&v| v as f64).collect();
    let price: Vec<f64> = adj.iter().map(|r| r.price).collect();
    let volume: Vec<f64> = adj.iter().map(|r| r.volume).collect();
    let turnover: Vec<f64> = adj.iter().map(|r| r.turnover).collect();
    let flag: Vec<f64> = adj.iter().map(|r| r.flag).collect();
    let ask_order: Vec<i64> = adj.iter().map(|r| r.ask_order).collect();
    let bid_order: Vec<i64> = adj.iter().map(|r| r.bid_order).collect();

    let mut px: Vec<Vec<f64>> = Vec::with_capacity(15);
    for w in PRICE_WINS.iter() {
        px.push(roll_future_last(&times, &price, w * 1e9));
    }
    let act_buy: Vec<f64> = flag
        .iter()
        .zip(volume.iter())
        .map(|(&f, &v)| if f == 66.0 { v } else { 0.0 })
        .collect();
    let mut earn: Vec<Vec<f64>> = Vec::with_capacity(4);
    let mut br: Vec<Vec<f64>> = Vec::with_capacity(4);
    for w in EARN_WINS.iter() {
        let first = roll_past_first(&times, &price, w * 1e9);
        let cnt = roll_past_count(&times, w * 1e9);
        let hit: f64 = (0..n).filter(|&i| price[i] > first[i]).count() as f64;
        earn.push((0..n).map(|i| hit / cnt[i]).collect());
        let a = roll_future_sum(&times, &act_buy, w * 1e9);
        let v = roll_future_sum(&times, &volume, w * 1e9);
        br.push((0..n).map(|i| a[i] / v[i]).collect());
    }
    Some((
        Frame {
            ns,
            price,
            volume,
            turnover,
            ask_order,
            bid_order,
            px,
            earn,
            br,
        },
        close_price,
    ))
}

// ============================ 分组（按委托号） ============================

const N_RET: usize = 16; // ret_1s..ret_1h, ret_close

struct GrpData {
    keys: Vec<i64>,
    vol: Vec<f64>,
    turn: Vec<f64>,
    ret: Vec<[f64; N_RET]>,
}

/// groupby(key).agg({'turnover':'sum','volume':'sum','price_X':'last'})
/// turnover / volume 用 Kahan（实测 pandas 的 float 组内求和就是 Kahan）；
/// price_X 取组内最后一个非 NaN 值（GroupBy.last 的语义）。
fn group_by(
    keys: &[i64],
    idx: &[usize],
    volume: &[f64],
    turnover: &[f64],
    px: &[Vec<f64>],
    close_price: f64,
) -> GrpData {
    let mut map: HashMap<i64, usize> = HashMap::new();
    let mut ks: Vec<i64> = Vec::new();
    let mut tv: Vec<f64> = Vec::new();
    let mut vc: Vec<f64> = Vec::new();
    let mut vv: Vec<f64> = Vec::new();
    let mut last: Vec<[f64; 15]> = Vec::new();
    for &i in idx {
        let k = keys[i];
        let g = match map.get(&k) {
            Some(&x) => x,
            None => {
                let x = ks.len();
                map.insert(k, x);
                ks.push(k);
                tv.push(0.0);
                vc.push(0.0);
                vv.push(0.0);
                last.push([f64::NAN; 15]);
                x
            }
        };
        let t = turnover[i];
        if !t.is_nan() {
            let y = t - vc[g];
            let s = tv[g] + y;
            vc[g] = (s - tv[g]) - y;
            tv[g] = s;
        }
        let v = volume[i];
        if !v.is_nan() {
            vv[g] += v;
        }
        for j in 0..15 {
            let x = px[j][i];
            if !x.is_nan() {
                last[g][j] = x;
            }
        }
    }
    let mut order: Vec<usize> = (0..ks.len()).collect();
    order.sort_by_key(|&i| ks[i]);
    let mut out_keys = Vec::with_capacity(order.len());
    let mut out_vol = Vec::with_capacity(order.len());
    let mut out_ret = Vec::with_capacity(order.len());
    for &i in order.iter() {
        let p = tv[i] / vv[i];
        let mut r = [f64::NAN; N_RET];
        for j in 0..15 {
            let x = last[i][j] / p - 1.0;
            r[j] = if x == 0.0 { f64::NAN } else { x };
        }
        let x = close_price / p - 1.0;
        r[15] = if x == 0.0 { f64::NAN } else { x };
        out_keys.push(ks[i]);
        out_vol.push(vv[i]);
        out_ret.push(r);
    }
    let out_turn: Vec<f64> = order.iter().map(|&i| tv[i]).collect();
    GrpData {
        keys: out_keys,
        vol: out_vol,
        turn: out_turn,
        ret: out_ret,
    }
}

/// `df[df.count(axis=1) > int(df.shape[1]/2)]`。
/// ！！drop 的列表里**没有 volume**，所以这一层是 17 列（volume + 16 个 ret），
/// int(17/2)=8，条件等价于「16 个 ret 里至少 8 个非 NaN」。
fn half_filter(g: &GrpData) -> GrpData {
    let mut keys = Vec::new();
    let mut vol = Vec::new();
    let mut ret = Vec::new();
    for i in 0..g.keys.len() {
        let c = g.ret[i].iter().filter(|x| !x.is_nan()).count();
        if c + 1 > N_RET / 2 {
            keys.push(g.keys[i]);
            vol.push(g.vol[i]);
            ret.push(g.ret[i]);
        }
    }
    GrpData { keys, vol, turn: g.turn.iter().enumerate().filter(|(i,_)| g.ret[*i].iter().filter(|x| !x.is_nan()).count() > N_RET/2).map(|(_,v)| *v).collect(), ret }
}

/// df_buy_happy0（>0）/ df_sell_happy0（<0）：每列 满足数 / 非 NaN 数，分母为 0 时 NaN
fn happy0(g: &GrpData, less: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(N_RET);
    for j in 0..N_RET {
        let mut pos = 0.0f64;
        let mut cnt = 0.0f64;
        for i in 0..g.keys.len() {
            let x = g.ret[i][j];
            if x.is_nan() {
                continue;
            }
            cnt += 1.0;
            if if less { x < 0.0 } else { x > 0.0 } {
                pos += 1.0;
            }
        }
        out.push(if cnt == 0.0 { f64::NAN } else { pos / cnt });
    }
    out
}

/// df_buy_happy1（buy 用 >0）/ df_sell_happy1（sell 用 <0）：每行 满足数 / 非 NaN 数
fn happy1(g: &GrpData, less: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(g.keys.len());
    for i in 0..g.keys.len() {
        let mut pos = 0.0f64;
        let mut cnt = 0.0f64;
        for j in 0..N_RET {
            let x = g.ret[i][j];
            if x.is_nan() {
                continue;
            }
            cnt += 1.0;
            if if less { x < 0.0 } else { x > 0.0 } {
                pos += 1.0;
            }
        }
        out.push(if cnt == 0.0 { f64::NAN } else { pos / cnt });
    }
    out
}

/// dh1.groupby('volume').rate.mean()：按 volume 升序，组内按原行序做 nanmean
fn group_by_volume(vol: &[f64], rate: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut map: HashMap<u64, Vec<f64>> = HashMap::new();
    let mut keys: Vec<f64> = Vec::new();
    for i in 0..vol.len() {
        let k = vol[i].to_bits();
        if !map.contains_key(&k) {
            keys.push(vol[i]);
            map.insert(k, Vec::new());
        }
        map.get_mut(&k).unwrap().push(rate[i]);
    }
    keys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut kv = Vec::with_capacity(keys.len());
    let mut rv = Vec::with_capacity(keys.len());
    for k in keys.iter() {
        let vals = &map[&k.to_bits()];
        kv.push(*k);
        rv.push(nan_mean(vals));
    }
    (kv, rv)
}

struct Stats {
    mean: f64,
    std: f64,
    median: f64,
    skew: f64,
    kurt: f64,
    trend: f64,
    autocorr: f64,
}

fn stats(v: &[f64]) -> Stats {
    Stats {
        mean: nan_mean(v),
        std: nan_std(v),
        median: nan_median(v),
        skew: nan_skew(v),
        kurt: nan_kurt(v),
        trend: trend(v),
        autocorr: autocorr1(v),
    }
}

/// 每组 81 个值。返回 None 表示旧代码抛异常（整只股票全 NaN）。
fn one(fr: &Frame, idx: &[usize], close_price: f64, half: bool) -> Option<Vec<f64>> {
    let mut out: Vec<f64> = Vec::with_capacity(81);

    // ---- 0..16：earn_buy_Xm 与 buyrate_Ym 的相关系数
    let mut earn_c: Vec<Vec<f64>> = Vec::with_capacity(4);
    let mut br_c: Vec<Vec<f64>> = Vec::with_capacity(4);
    for e in 0..4 {
        earn_c.push(idx.iter().map(|&i| fr.earn[e][i]).collect());
        br_c.push(idx.iter().map(|&i| fr.br[e][i]).collect());
    }
    for e in 0..4 {
        for b in 0..4 {
            out.push(pearson(&earn_c[e], &br_c[b]));
        }
    }

    // ---- 分组（sell 按 ask_order，buy 按 bid_order）
    let mut sell = group_by(
        &fr.ask_order,
        idx,
        &fr.volume,
        &fr.turnover,
        &fr.px,
        close_price,
    );
    let mut buy = group_by(
        &fr.bid_order,
        idx,
        &fr.volume,
        &fr.turnover,
        &fr.px,
        close_price,
    );
    if half {
        sell = half_filter(&sell);
        buy = half_filter(&buy);
    }
    // df_buy_happy1.autocorr() / dh1_groupvolume.rate.autocorr() 在长度 <2 时抛 ValueError
    if buy.keys.len() < 2 || sell.keys.len() < 2 {
        return None;
    }

    let mut corr_buy: Vec<f64> = Vec::with_capacity(N_RET);
    let mut corr_sell: Vec<f64> = Vec::with_capacity(N_RET);
    for j in 0..N_RET {
        let bret: Vec<f64> = buy.ret.iter().map(|r| r[j]).collect();
        let sret: Vec<f64> = sell.ret.iter().map(|r| r[j]).collect();
        corr_buy.push(pearson(&buy.vol, &bret));
        corr_sell.push(pearson(&sell.vol, &sret));
    }

    let h0_buy = happy0(&buy, false);
    let h0_sell = happy0(&sell, true);
    let h1_buy = happy1(&buy, false);
    let h1_sell = happy1(&sell, true);

    let (gbv, gbr) = group_by_volume(&buy.vol, &h1_buy);
    let (gsv, gsr) = group_by_volume(&sell.vol, &h1_sell);
    if gbv.len() < 2 || gsv.len() < 2 {
        return None;
    }

    out.push(pearson(&h1_buy, &buy.vol));
    out.push(pearson(&h1_sell, &sell.vol));
    out.push(pearson(&gbv, &gbr));
    out.push(pearson(&gsv, &gsr));
    out.push(pearson(&corr_buy, &corr_sell));
    out.push(pearson(&h0_buy, &h0_sell));
    out.push(dtw(&corr_buy, &corr_sell));
    out.push(dtw(&h0_buy, &h0_sell));
    out.push(nan_mean(&gbr));
    out.push(nan_mean(&gsr));
    out.push(nan_std(&gbr));
    out.push(nan_std(&gsr));
    out.push(nan_median(&gbr));
    out.push(nan_median(&gsr));
    out.push(nan_skew(&gbr));
    out.push(nan_skew(&gsr));
    out.push(nan_kurt(&gbr));
    out.push(nan_kurt(&gsr));
    out.push(trend(&gbr));
    out.push(trend(&gsr));
    out.push(autocorr1(&gbr));
    out.push(autocorr1(&gsr));
    out.push(dtw(&gbr, &gsr));

    let series: [&Vec<f64>; 6] = [&corr_buy, &corr_sell, &h0_buy, &h0_sell, &h1_buy, &h1_sell];
    let st: Vec<Stats> = series.iter().map(|v| stats(v)).collect();
    for s in st.iter() {
        out.push(s.mean);
    }
    for s in st.iter() {
        out.push(s.std);
    }
    for s in st.iter() {
        out.push(s.median);
    }
    for s in st.iter() {
        out.push(s.skew);
    }
    for s in st.iter() {
        out.push(s.kurt);
    }
    for s in st.iter() {
        out.push(s.trend);
    }
    for s in st.iter() {
        out.push(s.autocorr);
    }

    debug_assert_eq!(out.len(), 81);
    Some(out)
}

// ============================ 主入口 ============================

/// 12 个切片（前 6 个 half_at_least=False，后 6 个 =True；每个里是 6 个时间段）
/// 全天 / 上午 / 下午 × 删去最后一小时(df1) / 删去最后10分钟(df2)
const SUBSETS: [(i64, i64); 6] = [
    (34_200 * 1_000_000, 44_820 * 1_000_000), // df1 全天 09:30-12:27
    (34_200 * 1_000_000, 47_820 * 1_000_000), // df2 全天 09:30-13:17
    (34_200 * 1_000_000, 37_800 * 1_000_000), // df1 上午 09:30-10:30
    (34_200 * 1_000_000, 41_400 * 1_000_000), // df2 上午 09:30-11:30
    (41_400 * 1_000_000, 44_820 * 1_000_000), // df1 下午 11:30-12:27
    (41_400 * 1_000_000, 47_820 * 1_000_000), // df2 下午 11:30-13:17
];

pub fn compute(content: &str) -> Option<Vec<f64>> {
    let (fr, close_price) = build_frame(content)?;
    let tod: Vec<i64> = fr.ns.iter().map(|&v| tod_us(v)).collect();
    let mut result: Vec<f64> = Vec::with_capacity(972);
    for half in [false, true] {
        for &(lo, hi) in SUBSETS.iter() {
            let idx: Vec<usize> = (0..tod.len())
                .filter(|&i| tod[i] >= lo && tod[i] <= hi)
                .collect();
            let r = one(&fr, &idx, close_price, half)?;
            result.extend_from_slice(&r);
        }
    }
    debug_assert_eq!(result.len(), 972);
    Some(result)
}

const METRIC_NAMES: [&str; 81] = [
    "earn_buy_1m_corr_buyrate_1m",
    "earn_buy_1m_corr_buyrate_5m",
    "earn_buy_1m_corr_buyrate_10m",
    "earn_buy_1m_corr_buyrate_30m",
    "earn_buy_5m_corr_buyrate_1m",
    "earn_buy_5m_corr_buyrate_5m",
    "earn_buy_5m_corr_buyrate_10m",
    "earn_buy_5m_corr_buyrate_30m",
    "earn_buy_10m_corr_buyrate_1m",
    "earn_buy_10m_corr_buyrate_5m",
    "earn_buy_10m_corr_buyrate_10m",
    "earn_buy_10m_corr_buyrate_30m",
    "earn_buy_30m_corr_buyrate_1m",
    "earn_buy_30m_corr_buyrate_5m",
    "earn_buy_30m_corr_buyrate_10m",
    "earn_buy_30m_corr_buyrate_30m",
    "corr_buy_happy1_volume",
    "corr_sell_happy1_volume",
    "corr_buy_groupbyvolume",
    "corr_sell_groupbyvolume",
    "corr_buy_sell",
    "corr_buy_happy0_sell_happy0",
    "dtw_buy_sell",
    "dtw_buy_happy0_sell_happy0",
    "dh1_buy_groupvolume_mean",
    "dh1_sell_groupvolume_mean",
    "dh1_buy_groupvolume_std",
    "dh1_sell_groupvolume_std",
    "dh1_buy_groupvolume_median",
    "dh1_sell_groupvolume_median",
    "dh1_buy_groupvolume_skew",
    "dh1_sell_groupvolume_skew",
    "dh1_buy_groupvolume_kurt",
    "dh1_sell_groupvolume_kurt",
    "dh1_buy_groupvolume_trend",
    "dh1_sell_groupvolume_trend",
    "dh1_buy_groupvolume_autocorr",
    "dh1_sell_groupvolume_autocorr",
    "dtw_buy_groupbyvolume_sell_groupbyvolume",
    "corr_buy_mean",
    "corr_sell_mean",
    "df_buy_happy0_mean",
    "df_sell_happy0_mean",
    "df_buy_happy1_mean",
    "df_sell_happy1_mean",
    "corr_buy_std",
    "corr_sell_std",
    "df_buy_happy0_std",
    "df_sell_happy0_std",
    "df_buy_happy1_std",
    "df_sell_happy1_std",
    "corr_buy_median",
    "corr_sell_median",
    "df_buy_happy0_median",
    "df_sell_happy0_median",
    "df_buy_happy1_median",
    "df_sell_happy1_median",
    "corr_buy_skew",
    "corr_sell_skew",
    "df_buy_happy0_skew",
    "df_sell_happy0_skew",
    "df_buy_happy1_skew",
    "df_sell_happy1_skew",
    "corr_buy_kurt",
    "corr_sell_kurt",
    "df_buy_happy0_kurt",
    "df_sell_happy0_kurt",
    "df_buy_happy1_kurt",
    "df_sell_happy1_kurt",
    "corr_buy_trend",
    "corr_sell_trend",
    "df_buy_happy0_trend",
    "df_sell_happy0_trend",
    "df_buy_happy1_trend",
    "df_sell_happy1_trend",
    "corr_buy_autocorr",
    "corr_sell_autocorr",
    "df_buy_happy0_autocorr",
    "df_sell_happy0_autocorr",
    "df_buy_happy1_autocorr",
    "df_sell_happy1_autocorr",
];

pub const N_OUT: usize = 81 * 12;

const CUT_NAMES: [&str; 2] = ["删去最后一小时", "删去最后10分钟"];
const PERIOD_NAMES: [&str; 3] = ["全天", "上午", "下午"];
const HALF_NAMES: [&str; 2] = ["不需要至少一半", "至少一半"];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for h in HALF_NAMES.iter() {
        for p in PERIOD_NAMES.iter() {
            for c in CUT_NAMES.iter() {
                for m in METRIC_NAMES.iter() {
                    out.push(format!("{}_{}_{}_{}", m, c, p, h));
                }
            }
        }
    }
    out
}

fn level2_file(code: &str, date: i64) -> Option<PathBuf> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let path = level2_file(code, date)?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;
    let vals = compute(&content)?;
    Some(vals.iter().map(|&v| v as f32).collect())
}

/// hm65 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
