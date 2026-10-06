//! hm053「最速曲线」—— 旧框架逐股系列的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm53_最速曲线/hm53_okay.py（429 行）
//! 参数：get_many=0, read_full=0, change_time=1。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 get_factor 做了什么
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)（deal_data 把 exchtime 变成 Timestamp）
//! 2) df = df.set_index('exchtime').between_time('09:30:00','14:57:00')
//!    然后 df1 = [09:30,11:30]、df2 = [13:00,14:57]，df2 的 index 前移 90 分钟，
//!    df = concat([df1,df2])。等价于「上午 [09:30,11:30] + 下午 [13:00,14:57] 前移 90 分钟」。
//! 3) df.price = df.price.round(2)；df.volume = df.volume.round(2)（np.round，不是四舍五入）
//! 4) exchtime_int = df.index.to_numpy(dtype=float)   ← datetime64[ns] 转 float64，就是 ns 值
//!    然后 (x - mean)/std 再减 min；price 同理（先标准化再减 min）；volume 标准化后 cumsum。
//!    mean 是 numpy 成对求和 /n，std 是 nanstd(ddof=1)。
//! 5) df = df.drop_duplicates(subset=['exchtime_int'], keep='first')
//! 6) 五个窗口（都按**平移后**的索引钟点，闭区间，微秒判定）：
//!      all      = 整张表
//!      open30   = [09:30:00, 10:00:00]
//!      morning  = [09:30:00, 11:30:00]
//!      afternoon= [11:30:00, 13:27:00]   ← 上午 11:30:00 那一微秒也算进来
//!      close30  = [13:00:00, 13:27:00]
//! 7) 每个窗口跑 single()：先用当日最高价/最低价的第一次与最后一次出现时刻定
//!    [start,end] 区间（先出现极值的那一端的时刻为 start，另一端最后一次出现为 end），
//!    取 df_hl = 该区间内的行；然后对 df_hl 和整张 df 各跑若干次 sing(x, y)，
//!    每次输出 21 个统计量。
//!      df_hl 上的 6 组（后缀 _hl）：time_price / time_volume / volume_time /
//!                                    volume_price / price_time / price_volume
//!      整张表上的 4 组（后缀 _se）：time_price / time_volume / volume_time / volume_price
//!    合计 210 个。5 个窗口 × 210 = 1050。
//! 8) 任何一步抛异常（比如 df_hl 为空导致 .iloc[0] 报错）都被外层 try 吞掉，
//!    get_factor 返回 None → 整天 1050 个 NaN。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、sing(x, y) 的 21 个量
//! ════════════════════════════════════════════════════════════════════
//!   x1,y1 = x.iloc[0], y.iloc[0]；x2,y2 = x.iloc[-1], y.iloc[-1]
//!   y_perfect = rp.brachistochrone_curve(x1, y1, x2, y2, x.to_numpy(float), 1)
//!   xy = DataFrame({x, y, y_perfect}).dropna()      ← y_perfect 里 NaN 的行被丢掉
//!   然后 21 个量见 SING_NAMES；x_shape[0] >= 100000 或抛异常时返回 21 个 NaN。
//!
//! 注意 y_perfect 只在 x 落在 [min(x1,x2), max(x1,x2)] 内才有值，其余是 NaN；
//! 所以 price 当 x 的时候会丢掉一大批行，这是旧代码的原意。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、pandas 语义对齐
//! ════════════════════════════════════════════════════════════════════
//! · Series.corr / Series.autocorr(1) → nanops.nancorr → **np.corrcoef**（两遍法），
//!   两边成对去 NaN 后算 cov/(sqrt(cov00)·sqrt(cov11))，本文件用 nan_corr 复刻。
//!   注意 autocorr 走的是 Series.corr(self.shift(lag))，不是 DataFrame.corrwith，
//!   所以和 hm060 里那条 autocorr 不是同一个公式。
//! · mean / sum 用 numpy 的成对求和（pw_sum），std 是 nanstd(ddof=1)。
//! · quantile(0.2/0.8) 是 numpy 的 'linear' 法：idx = q·(n-1)，
//!   lo=floor(idx)，gamma=idx-lo，然后按 numpy _lerp 的规则（t>=0.5 走 b-(b-a)(1-t)）插值。
//! · round(2) 是 np.round：mult=100，rint(x·100)/100，rint 是四舍六入五成双。
//! · dropna() 只丢 NaN（不丢 inf）；y_perfect 不会产生 inf。
//! · 空 Series 的 sum() 是 0.0（不是 NaN），mean()/max()/quantile() 是 NaN。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、已知差异 / 取舍
//! ════════════════════════════════════════════════════════════════════
//! · **旧代码的 1 秒超时没有复现，这是本文件唯一实质性的偏差，必须知道**。
//!   旧代码给 brachistochrone_curve 传 timeout_seconds=1，函数内部在每个点、每次二分
//!   迭代都调一次 `Instant::now()`。实测这台机器上 `Instant::now()` 约 440ns，
//!   300750@20231020（x 区间覆盖全部 59599 个点）光时钟调用就花 1.3 秒，于是必然超时 →
//!   `sing(df.exchtime_int, df.price)` 与 `sing(df.exchtime_int, df.volume)` 这两组
//!   返回 NaN（每组 21 个，一个窗口 42 个）；同一天 x=volume 的两组因为
//!   cumsum 是随机游走、区间只覆盖 66 个点，8ms 就跑完，不超时。
//!   同一份输入不设超时时只要 31ms，说明这 1 秒完全是时钟调用的开销，与计算量无关。
//!   阈值大约在 45000 个点：000001@20250303（21815 点）不超时，300750@20231020（59599 点）超时。
//!   这个阈值取决于机器的 clock_gettime 代价与当时负载，跨机器/跨次运行不可复现，
//!   会让同一个因子在重算时从「有值」变成 NaN。**本文件选择确定性**：不设超时，
//!   按数学定义算出全部 1050 个值 —— 把机器噪声写进因子库比「和旧数据差一截」更糟，
//!   而且差值只出现在旧代码给 NaN 的地方，新值本身是对的。
//!
//!   受影响的位置：每个窗口里 x 取 exchtime_int 的两组 `_se`（time_price、time_volume），
//!   各 21 个，一个窗口 42 个，理论上最多 210 个；具体哪些窗口命中要看该窗口的行数，
//!   是数据相关的。300750@20231020 只有 all 窗口命中（其余窗口行数少），偏差落在
//!   第 126~167 号，实测「旧 NaN / 新有值」42 处，其余 1008 个值逐值一致（见下）。
//!
//!   受影响的入选 base（窗口 = all 的那几个）：
//!     distance_mean_time_price_se_all
//!     x_mean_high20_distance_time_price_se_all
//!     x_mean_high20_distance_time_price_se_all_fold
//!     y_mean_high20_distance_abs_time_price_se_all
//!   最终名单里由它们派生的 4 个：
//!     distance_mean_time_price_se_all_smooth_1
//!     distance_mean_time_price_se_all_min_smooth_5
//!     x_mean_high20_distance_time_price_se_all_fold_mean_smooth_5
//!     y_mean_high20_distance_abs_time_price_se_all_mean_smooth_10
//!   其它窗口的同类 base 只有在对应窗口也超时时才受影响（数据相关）。
//!
//!   对账口径：4 组选在旧代码**没有**超时的股票日（严格逐值一致），
//!   另外 1 组刻意选会超时的 300750@20231020，用来把上面这 42 处偏差记录在案。
//! · brachistochrone_curve 是 #[pyfunction]，worker 链路上不能碰，算法整段抄进来，
//!   没有改公共文件 src/sequence/mod.rs。
//! · names() 的 1050 个名字无重名（已逐项核对），不需要 _dup2 后缀。

use crate::data_paths;

const NCOL: usize = 21;
const PER_FRAME: usize = 210;
const N_TOTAL: usize = 1050;

const DAY_NS: i64 = 86_400_000_000_000;
const EXCHANGE_OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
const AFTERNOON_SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;

const MS_0930: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MS_1000: i64 = (10 * 3600) * 1_000_000;
const MS_1130: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const MS_1300: i64 = (13 * 3600) * 1_000_000;
const MS_1327: i64 = (13 * 3600 + 27 * 60) * 1_000_000;
const MS_1457: i64 = (14 * 3600 + 57 * 60) * 1_000_000;

const COL_EXCHTIME: usize = 4;
const COL_PRICE: usize = 7;
const COL_VOLUME: usize = 8;
const COL_FLAG: usize = 10;

/// sing 的 21 个名字，顺序与 SING 输出一致。
const SING_NAMES: [&str; 21] = [
    "corr_y_yperfect",
    "corr_y_yperfect_yshift1",
    "corr_y_yperfect_yperfectshift1",
    "distance_mean",
    "distance_max",
    "distance_sum",
    "distance_autocorr1",
    "distance_abs_mean",
    "distance_abs_max",
    "distance_abs_sum",
    "distance_abs_autocorr1",
    "y_mean_low20_distance",
    "y_mean_high20_distance",
    "y_mean_low20_distance_abs",
    "y_mean_high20_distance_abs",
    "x_mean_low20_distance",
    "x_mean_high20_distance",
    "x_mean_low20_distance_abs",
    "x_mean_high20_distance_abs",
    "y_mean_high20_minus_low20_distance",
    "x_mean_high20_minus_low20_distance",
];

/// df_hl 上的 6 组 (x, y)，值 0=exchtime_int、1=price、2=volume。
const GROUPS_HL: [(&str, usize, usize); 6] = [
    ("time_price", 0, 1),
    ("time_volume", 0, 2),
    ("volume_time", 2, 0),
    ("volume_price", 2, 1),
    ("price_time", 1, 0),
    ("price_volume", 1, 2),
];
/// 整张表上的 4 组 (x, y)。
const GROUPS_SE: [(&str, usize, usize); 4] = [
    ("time_price", 0, 1),
    ("time_volume", 0, 2),
    ("volume_time", 2, 0),
    ("volume_price", 2, 1),
];
const WINDOWS: [&str; 5] = ["all", "open30", "morning", "afternoon", "close30"];

// ══════════════════════════════════════════════════════════════════════
// 通用数值
// ══════════════════════════════════════════════════════════════════════

const PW_BLOCKSIZE: usize = 128;

/// numpy 的成对求和（pairwise_sum_DOUBLE）。
fn pw_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for i in 0..n {
            res += a[i];
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let stop = n - (n % 8);
        let mut i = 8usize;
        while i < stop {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        let mut leftover = 0.0f64;
        for k in stop..n {
            leftover += a[k];
        }
        res += leftover;
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pw_sum(&a[..n2]) + pw_sum(&a[n2..])
}

/// nanops.nanstd(ddof=1)：sqrt(Σ(x-mean)²/(n-1))，n<2 给 NaN。
fn std_of(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = pw_sum(v) / n as f64;
    let sqr: Vec<f64> = v.iter().map(|&x| (mean - x) * (mean - x)).collect();
    (pw_sum(&sqr) / (n - 1) as f64).sqrt()
}

/// Series.corr / nanops.nancorr 的 pearson：np.corrcoef(a, b)[0, 1]。
fn nan_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let nf = n as f64;
    let ma = pw_sum(a) / nf;
    let mb = pw_sum(b) / nf;
    let mut num = 0.0f64;
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        num += da * db;
        sa += da * da;
        sb += db * db;
    }
    let fact = (n - 1) as f64;
    let c01 = num / fact;
    let s0 = (sa / fact).sqrt();
    let s1 = (sb / fact).sqrt();
    (c01 / s0) / s1
}

/// Series.corr(self.shift(lag))：成对丢掉 NaN 之后的 np.corrcoef。
fn autocorr1(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    nan_corr(&v[1..], &v[..n - 1])
}

/// numpy _lerp。
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

/// np.quantile(..., method='linear')。
fn quantile_linear(v: &[f64], q: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = q * (n as f64 - 1.0);
    let lo = idx.floor();
    let li = lo as usize;
    let hi = (li + 1).min(n - 1);
    lerp(s[li], s[hi], idx - lo)
}

/// pandas 的 max()/min()：跳过 NaN，全 NaN（或空）给 NaN。
fn max_of(v: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in v {
        if x > m {
            m = x;
        }
    }
    if m == f64::NEG_INFINITY {
        f64::NAN
    } else {
        m
    }
}

fn min_of(v: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    for &x in v {
        if x < m {
            m = x;
        }
    }
    if m == f64::INFINITY {
        f64::NAN
    } else {
        m
    }
}

fn nanmean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    pw_sum(v) / v.len() as f64
}

// ══════════════════════════════════════════════════════════════════════
// np.rint / pandas 时间戳 / pandas CSV 浮点解析（与 hm060.rs 同一套）
// ══════════════════════════════════════════════════════════════════════

fn rint_ties_even(x: f64) -> f64 {
    let t = x.trunc();
    let f = x - t;
    if f == 0.5 {
        if (t as i64) % 2 == 0 {
            t
        } else {
            t + 1.0
        }
    } else if f == -0.5 {
        if ((t as i64) % 2) == 0 {
            t
        } else {
            t - 1.0
        }
    } else {
        x.round()
    }
}

/// np.round(x, 2)。
fn round2(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    rint_ties_even(x * 100.0) / 100.0
}

fn pandas_ns(us: i64) -> i64 {
    let v = us as f64 / 1e6;
    let base = v as i64;
    let frac = v - base as f64;
    let frac_rounded = rint_ties_even(frac * 1e9) / 1e9;
    base * 1_000_000_000 + (frac_rounded * 1e9) as i64 + EXCHANGE_OFFSET_NS
}

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

fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
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

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（与 hm033.rs / hm060.rs 里那份一致）。
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

// ══════════════════════════════════════════════════════════════════════
// 最速曲线（照抄 src/sequence/mod.rs 的 brachistochrone_curve）
// ══════════════════════════════════════════════════════════════════════

fn brachistochrone_objective(r: f64, theta: f64, x_target: f64, y_target: f64) -> f64 {
    let x_end = r * (theta - theta.sin());
    let y_end = -r * (1.0 - theta.cos());
    (x_end - x_target).powi(2) + (y_end - y_target).powi(2)
}

fn optimize_r_theta(initial_r: f64, initial_theta: f64, x_target: f64, y_target: f64) -> (f64, f64) {
    let mut r = initial_r;
    let mut theta = initial_theta;
    let mut step_size_r = initial_r * 0.1;
    let mut step_size_theta = 0.1;
    let tolerance = 1e-6;
    for _ in 0..100 {
        let current_error = brachistochrone_objective(r, theta, x_target, y_target);
        let r_plus = brachistochrone_objective(r + step_size_r, theta, x_target, y_target);
        let r_minus = brachistochrone_objective(r - step_size_r, theta, x_target, y_target);
        if r_plus < current_error {
            r += step_size_r;
        } else if r_minus < current_error {
            r -= step_size_r;
        } else {
            step_size_r *= 0.5;
        }
        let theta_plus = brachistochrone_objective(r, theta + step_size_theta, x_target, y_target);
        let theta_minus = brachistochrone_objective(r, theta - step_size_theta, x_target, y_target);
        if theta_plus < current_error {
            theta += step_size_theta;
        } else if theta_minus < current_error {
            theta -= step_size_theta;
        } else {
            step_size_theta *= 0.5;
        }
        if step_size_r < tolerance && step_size_theta < tolerance {
            break;
        }
    }
    (r, theta)
}

fn solve_theta_for_x(r: f64, theta_max: f64, x: f64) -> Option<f64> {
    let mut low = 0.0f64;
    let mut high = theta_max;
    let tolerance = 1e-6;
    for _ in 0..50 {
        let mid = (low + high) / 2.0;
        let x_mid = r * (mid - mid.sin());
        if (x_mid - x).abs() < tolerance {
            return Some(mid);
        }
        if x_mid < x {
            low = mid;
        } else {
            high = mid;
        }
    }
    let final_mid = (low + high) / 2.0;
    let final_x = r * (final_mid - final_mid.sin());
    if (final_x - x).abs() < tolerance * 10.0 {
        Some(final_mid)
    } else {
        None
    }
}

/// brachistochrone_curve_impl（timeout_seconds 那一支去掉，理由见文件头）。
fn brachistochrone_curve(x1: f64, y1: f64, x2: f64, y2: f64, x_series: &[f64]) -> Vec<f64> {
    let n = x_series.len();
    let mut result = vec![f64::NAN; n];

    let (x1, y1, x2, y2) = if x1 > x2 {
        (x2, y2, x1, y1)
    } else {
        (x1, y1, x2, y2)
    };

    let x_offset = x1;
    let y_offset = y1;
    let x2_shifted = x2 - x_offset;
    let mut y2_shifted = y2 - y_offset;
    let flip = y2_shifted > 0.0;
    if flip {
        y2_shifted = -y2_shifted;
    }

    let initial_r = (x2_shifted * x2_shifted + y2_shifted * y2_shifted).sqrt() / 2.0;
    let initial_theta = std::f64::consts::PI;
    let (r, theta_max) = optimize_r_theta(initial_r, initial_theta, x2_shifted, y2_shifted);

    for j in 0..n {
        let x_shifted = x_series[j] - x_offset;
        if x_shifted < 0.0 || x_shifted > x2_shifted {
            continue;
        }
        if let Some(theta) = solve_theta_for_x(r, theta_max, x_shifted) {
            let mut y = -r * (1.0 - theta.cos());
            if flip {
                y = -y;
            }
            result[j] = y + y_offset;
        }
    }
    result
}

// ══════════════════════════════════════════════════════════════════════
// sing(x, y)：21 个统计量
// ══════════════════════════════════════════════════════════════════════

fn mean_of_selection(values: &[f64], dist: &[f64], q: f64, below: bool) -> f64 {
    let thr = quantile_linear(dist, q);
    let mut acc: Vec<f64> = Vec::new();
    for i in 0..dist.len() {
        let hit = if below { dist[i] < thr } else { dist[i] > thr };
        if hit {
            acc.push(values[i]);
        }
    }
    nanmean(&acc)
}

fn sing(x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; NCOL];
    let n0 = x.len();
    if n0 >= 100000 || n0 == 0 {
        return out;
    }
    let (x1, y1) = (x[0], y[0]);
    let (x2, y2) = (x[n0 - 1], y[n0 - 1]);
    let yp_all = brachistochrone_curve(x1, y1, x2, y2, x);

    // DataFrame(...).dropna()
    let mut xs: Vec<f64> = Vec::with_capacity(n0);
    let mut ys: Vec<f64> = Vec::with_capacity(n0);
    let mut yps: Vec<f64> = Vec::with_capacity(n0);
    for i in 0..n0 {
        if yp_all[i].is_nan() {
            continue;
        }
        xs.push(x[i]);
        ys.push(y[i]);
        yps.push(yp_all[i]);
    }
    let n = xs.len();
    if n == 0 {
        // 空表的 pandas 取值：sum 是 0.0，其余（mean/max/quantile/corr）都是 NaN
        out[5] = 0.0;
        out[9] = 0.0;
        return out;
    }

    let dist: Vec<f64> = (0..n).map(|i| ys[i] - yps[i]).collect();
    let dist_abs: Vec<f64> = dist.iter().map(|v| v.abs()).collect();

    out[0] = nan_corr(&ys, &yps);
    // y.shift(1) 与 y_perfect 配对：第 i 行是 (y[i-1], yp[i])
    out[1] = nan_corr(&ys[..n - 1], &yps[1..]);
    // y_perfect.shift(1) 与 y 配对：第 i 行是 (yp[i-1], y[i])
    out[2] = nan_corr(&yps[..n - 1], &ys[1..]);
    out[3] = nanmean(&dist);
    out[4] = max_of(&dist);
    out[5] = pw_sum(&dist);
    out[6] = autocorr1(&dist);
    out[7] = nanmean(&dist_abs);
    out[8] = max_of(&dist_abs);
    out[9] = pw_sum(&dist_abs);
    out[10] = autocorr1(&dist_abs);
    out[11] = mean_of_selection(&ys, &dist, 0.2, true);
    out[12] = mean_of_selection(&ys, &dist, 0.8, false);
    out[13] = mean_of_selection(&ys, &dist_abs, 0.2, true);
    out[14] = mean_of_selection(&ys, &dist_abs, 0.8, false);
    out[15] = mean_of_selection(&xs, &dist, 0.2, true);
    out[16] = mean_of_selection(&xs, &dist, 0.8, false);
    out[17] = mean_of_selection(&xs, &dist_abs, 0.2, true);
    out[18] = mean_of_selection(&xs, &dist_abs, 0.8, false);
    out[19] = out[12] - out[11];
    out[20] = out[16] - out[15];
    out
}

// ══════════════════════════════════════════════════════════════════════
// 逐日构表
// ══════════════════════════════════════════════════════════════════════

/// 一天的标准化表格（对应 drop_duplicates 之后的 df）。
struct Frame {
    /// 平移后索引的当天微秒（between_time 判的就是这个）
    tod_us: Vec<i64>,
    /// exchtime_int = datetime64[ns] 转 float64 后标准化再减 min
    ex: Vec<f64>,
    price: Vec<f64>,
    vol: Vec<f64>,
}

impl Frame {
    fn field(&self, code: usize, i: usize) -> f64 {
        match code {
            0 => self.ex[i],
            1 => self.price[i],
            _ => self.vol[i],
        }
    }
}

fn build_frame(code: &str, date: i64) -> Option<Frame> {
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

    let mut morning: Vec<(i64, i64, f64, f64)> = Vec::new();
    let mut afternoon: Vec<(i64, i64, f64, f64)> = Vec::new();

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
        if int_prefix(fields[COL_FLAG]) == 32 {
            continue;
        }
        let ns = pandas_ns(exchtime_us);
        let tod = ns.rem_euclid(DAY_NS) / 1000;
        let price = round2(num(fields[COL_PRICE]));
        let vol = round2(num(fields[COL_VOLUME]));
        if (MS_0930..=MS_1130).contains(&tod) {
            morning.push((ns, tod, price, vol));
        } else if (MS_1300..=MS_1457).contains(&tod) {
            // 下午整体前移 90 分钟（pandas 里减的是 index，微秒判定一样）
            afternoon.push((ns - AFTERNOON_SHIFT_NS, tod - 5_400_000_000, price, vol));
        }
    }
    morning.extend(afternoon);
    let n = morning.len();
    if n == 0 {
        return Some(Frame {
            tod_us: Vec::new(),
            ex: Vec::new(),
            price: Vec::new(),
            vol: Vec::new(),
        });
    }
    let ns_vec: Vec<i64> = morning.iter().map(|r| r.0).collect();
    let tod_vec: Vec<i64> = morning.iter().map(|r| r.1).collect();
    let price_vec: Vec<f64> = morning.iter().map(|r| r.2).collect();
    let vol_vec: Vec<f64> = morning.iter().map(|r| r.3).collect();

    // exchtime_int = ns 转 float64，标准化后减 min
    let ns_f: Vec<f64> = ns_vec.iter().map(|&v| v as f64).collect();
    let ex_mean = nanmean(&ns_f);
    let ex_std = std_of(&ns_f);
    let mut ex: Vec<f64> = ns_f.iter().map(|&v| (v - ex_mean) / ex_std).collect();
    let ex_min = min_of(&ex);
    for v in ex.iter_mut() {
        *v -= ex_min;
    }

    let p_mean = nanmean(&price_vec);
    let p_std = std_of(&price_vec);
    let mut price: Vec<f64> = price_vec.iter().map(|&v| (v - p_mean) / p_std).collect();
    let p_min = min_of(&price);
    for v in price.iter_mut() {
        *v -= p_min;
    }

    let v_mean = nanmean(&vol_vec);
    let v_std = std_of(&vol_vec);
    let mut vol: Vec<f64> = Vec::with_capacity(n);
    let mut acc = 0.0f64;
    for &v in vol_vec.iter() {
        acc += (v - v_mean) / v_std;
        vol.push(acc);
    }

    // drop_duplicates(subset=['exchtime_int'], keep='first')
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::with_capacity(n);
    let mut keep: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if seen.insert(canonical_bits(ex[i])) {
            keep.push(i);
        }
    }
    Some(Frame {
        tod_us: keep.iter().map(|&i| tod_vec[i]).collect(),
        ex: keep.iter().map(|&i| ex[i]).collect(),
        price: keep.iter().map(|&i| price[i]).collect(),
        vol: keep.iter().map(|&i| vol[i]).collect(),
    })
}

/// 把 -0.0 归一成 0.0（khash 的浮点键也这么做），其余取位模式。
fn canonical_bits(x: f64) -> u64 {
    if x == 0.0 {
        0u64
    } else {
        x.to_bits()
    }
}

// ══════════════════════════════════════════════════════════════════════
// 每个窗口的 single()
// ══════════════════════════════════════════════════════════════════════

fn select_window(f: &Frame, window: usize) -> Vec<usize> {
    let (lo, hi) = match window {
        0 => return (0..f.tod_us.len()).collect(),
        1 => (MS_0930, MS_1000),
        2 => (MS_0930, MS_1130),
        3 => (MS_1130, MS_1327),
        _ => (MS_1300, MS_1327),
    };
    (0..f.tod_us.len())
        .filter(|&i| f.tod_us[i] >= lo && f.tod_us[i] <= hi)
        .collect()
}

/// 旧代码里 single(df) 的复刻；任何一步抛异常（返回 None）都会让整天变 NaN。
fn single(f: &Frame, idx: &[usize]) -> Option<Vec<f64>> {
    if idx.is_empty() {
        return None;
    }
    let high_price = {
        let mut m = f64::NEG_INFINITY;
        for &i in idx {
            if f.price[i] > m {
                m = f.price[i];
            }
        }
        m
    };
    let low_price = {
        let mut m = f64::INFINITY;
        for &i in idx {
            if f.price[i] < m {
                m = f.price[i];
            }
        }
        m
    };
    let high_time_first = idx.iter().copied().find(|&i| f.price[i] == high_price)?;
    let low_time_first = idx.iter().copied().find(|&i| f.price[i] == low_price)?;
    let high_time_last = idx.iter().rev().copied().find(|&i| f.price[i] == high_price)?;
    let low_time_last = idx.iter().rev().copied().find(|&i| f.price[i] == low_price)?;

    let (start_time, end_time) = if f.ex[high_time_first] < f.ex[low_time_first] {
        (f.ex[high_time_first], f.ex[low_time_last])
    } else {
        (f.ex[low_time_first], f.ex[high_time_last])
    };
    let hl: Vec<usize> = idx
        .iter()
        .copied()
        .filter(|&i| f.ex[i] >= start_time && f.ex[i] <= end_time)
        .collect();

    let mut out: Vec<f64> = Vec::with_capacity(PER_FRAME);
    for &(_, xa, ya) in GROUPS_HL.iter() {
        let x: Vec<f64> = hl.iter().map(|&i| f.field(xa, i)).collect();
        let y: Vec<f64> = hl.iter().map(|&i| f.field(ya, i)).collect();
        out.extend(sing(&x, &y));
    }
    for &(_, xa, ya) in GROUPS_SE.iter() {
        let x: Vec<f64> = idx.iter().map(|&i| f.field(xa, i)).collect();
        let y: Vec<f64> = idx.iter().map(|&i| f.field(ya, i)).collect();
        out.extend(sing(&x, &y));
    }
    debug_assert_eq!(out.len(), PER_FRAME);
    Some(out)
}

fn all_nan() -> Vec<f32> {
    vec![f32::NAN; N_TOTAL]
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let frame = match build_frame(code, date) {
        Some(f) => f,
        None => return Some(all_nan()),
    };
    if frame.tod_us.is_empty() {
        return Some(all_nan());
    }
    let mut out: Vec<f32> = Vec::with_capacity(N_TOTAL);
    for w in 0..WINDOWS.len() {
        let idx = select_window(&frame, w);
        match single(&frame, &idx) {
            Some(vals) => out.extend(vals.iter().map(|&v| v as f32)),
            None => return Some(all_nan()),
        }
    }
    debug_assert_eq!(out.len(), N_TOTAL);
    Some(out)
}

/// hm053 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_TOTAL);
    for w in WINDOWS.iter() {
        for (g, _, _) in GROUPS_HL.iter() {
            for b in SING_NAMES.iter() {
                out.push(format!("{}_{}_hl_{}", b, g, w));
            }
        }
        for (g, _, _) in GROUPS_SE.iter() {
            for b in SING_NAMES.iter() {
                out.push(format!("{}_{}_se_{}", b, g, w));
            }
        }
    }
    out
}
