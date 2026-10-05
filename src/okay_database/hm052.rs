//! hm52「自由泳之浪」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm52_自由泳之浪/hm52_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   逐股系列（check_my_kind == "level2"），**只用盘口快照**。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()
//!    · 读进整张盘口表（61 列；2018 及更早的文件末尾多一列 num，共 62 列）。
//!    · `data.index = pd.to_datetime(data["datetime"])` —— 索引取自 **datetime 字符串列**，
//!      不是 exchtime 列。
//!    · deal_data(change_time=1)：exchtime / localtime 从 epoch 微秒换成 Timestamp
//!      （+8 小时，北京钟点），并多出一列 delay。
//!    · 所以后面两次 between_time 筛的不是同一个量：
//!        第 3 步那次筛的是 **datetime 字符串**解析出来的时刻（2025 年的文件是毫秒精度，
//!        2018 年的文件连小数都没有，只有整秒），
//!        第 4 步 adjust_afternoon 里那次筛的是 **exchtime 微秒**换出来的时刻。
//!      两者一般相同，但在整秒边界上会差一点点，必须分别复刻。
//! 2) df = df[df.last_prc > 0]                       —— 开盘前 last_prc=0 的快照扔掉
//! 3) df.between_time("09:30:00", "14:57:00")        —— 筛 datetime 索引，两端都含
//! 4) adjust_afternoon(...)：因为索引名是 datetime 而 exchtime 是列，走的是
//!      df1 = set_index('exchtime').between_time('09:30:00','11:30:00')
//!      df2 = set_index('exchtime').between_time('13:00:00','14:57:00'); 时刻 -90min
//!      concat([df1, df2]).reset_index()
//!    **行序 = 全部上午行（文件序）在前、全部下午行（文件序）在后**，这个顺序决定了
//!    「同一个 3 秒格里取哪一行的 last_prc」，必须照抄。
//!    注意：第 3 步那层 between_time 的结果在 adjust_afternoon 里被丢掉了，
//!    真正决定保留哪些行的还是第 4 步的两个窗口。
//! 5) .set_index("exchtime") → resample("3s").last().dropna()
//!    · 3 秒格锚在当天 0 点（resample 默认 origin='start_day'、closed='left'、label='left'）。
//!    · 实测原始盘口表里没有任何 NaN，所以 `.last()` 就是「该格里最后一行的值」，
//!      `.dropna()` 等价于「丢掉没有数据的空格」。
//!    · df.volume = df.volume.diff() 之后 volume **再没被用到**（names 里也没有），
//!      这一步在 Rust 侧直接省略。
//! 6) 五个中间序列（都在这张 3 秒表上算）：
//!      vol_3s  = rp.rolling_volatility(last_prc, lookback=100, interval=1)
//!      vol_15s = rp.rolling_volatility(last_prc, lookback=100, interval=5)
//!      vol_60s = rp.rolling_volatility(last_prc, lookback=100, interval=20)
//!      ret_next = last_prc.rolling_future('300s').last() / last_prc - 1
//!      ret_lag1 = last_prc.shift(-1) / last_prc - 1
//!      resi_de15s   = rp.ols_residuals(X=[vol_15s], y=vol_3s)   ← 注意 x/y 的方向
//!      resi_de60s   = rp.ols_residuals(X=[vol_60s], y=vol_3s)
//!      abresi_de15s = rp.ols_residuals(X=[vol_3s],  y=vol_15s)
//!      abresi_de60s = rp.ols_residuals(X=[vol_3s],  y=vol_60s)
//!      再做 4 个 abs 版本（逐元素取绝对值，NaN 仍是 NaN）。
//! 7) 五个时段（作用于第 5 步那张 3 秒表的**标签**，两端都含）：
//!      all:       整张表
//!      open30:    [09:30:00, 11:30:00]
//!      morning:   [09:30:00, 11:30:00]   ← 与 open30 完全相同（旧代码就是这么写的）
//!      afternoon: [11:30:00, 13:27:00]   ← 11:30:00~13:27:00 正是平移后的整段下午
//!      close30:   [13:00:00, 13:27:00]   ← 平移后的最后 27 分钟
//!    每个时段算 72 个统计量，5 段共 360 个。
//! 8) names_single = 72 个指标名 × 后缀（_all/_open30/_morning/_afternoon/_close30），
//!    顺序 = all、open30、morning、afternoon、close30。
//!    names_save 里带 `_fold` 的是 abs 之后的 base 因子，由新流水线的
//!    write_selected_base 按「与截面均值的距离」自动生成（旧代码是 dwo.get_abs）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、每个时段的 72 个值（顺序必须逐字对上旧代码的 res 列表）
//! ════════════════════════════════════════════════════════════════════
//!   四列 son      = resi_de15s, resi_de60s, abresi_de15s, abresi_de60s
//!   四列 son_abs  = 上面四列取绝对值
//!   第 1..24 个：6 个统计量（median, std, skew, kurt, max, min）× 四列 son
//!   第 25..48 个：同样的 6 个统计量 × 四列 son_abs
//!   第 49..72 个：8 条 corr × 3 个被相关对象
//!        corr 的对象依次是 last_prc（8 个）、ret_lag1（8 个）、ret_next（8 个）
//!        每一组 8 个的顺序：son 四列的 corr，再 son_abs 四列的 corr
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、pandas 口径（照 pandas 2.3.3 的 nanops 抄）
//! ════════════════════════════════════════════════════════════════════
//! · median / max / min：只跳过 NaN（inf 参与计算），全 NaN → NaN，偶数个取中间两个的平均
//! · std：ddof=1，count-1 <= 0 → NaN；先求均值再求平方偏差和（两趟算法）
//! · skew：调整后的 Fisher-Pearson G1，m2=Σ(x-mean)²、m3=Σ(x-mean)³ **都不除 n**，
//!     result = n·√(n-1)/(n-2) · m3/m2^1.5；m2、m3 先 |·|<1e-14 归零；m2==0 → 0；n<3 → NaN
//! · kurt：无偏超额峰度 G2，
//!     adj = 3(n-1)²/((n-2)(n-3))，num = n(n+1)(n-1)m4，den = (n-2)(n-3)m2²，
//!     两者先 |·|<1e-14 归零；n<4 → NaN；den==0 → 0；否则 num/den - adj
//! · corr：Series.corr → nanops.nancorr → np.corrcoef(a,b)[0,1]，
//!     成对完整观测（任一 NaN 丢这一对），n<2 → NaN，
//!     方差不为正时 → NaN，**结果还要 clip 到 [-1, 1]**（np.corrcoef 会 clip）
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、精度
//! ════════════════════════════════════════════════════════════════════
//! · 盘口 CSV 读成 f64（只取 datetime / exchtime / last_prc 三列，本文件自己解析）。
//!   因子值本身是残差与相关系数，f32 读取器的相对误差会被放大，所以不用 f32 读取器。
//! · 3 秒标签一律用【当天 0 点起的纳秒数】。用当日纳秒而不是 epoch 纳秒，是因为
//!   rolling_future 里 pandas 把 int64 纳秒转成 f64，epoch 纳秒 (~1.7e18) 会被舍到
//!   256ns 的整数倍；好在 3 秒格标签都是 256 的整数倍，转 f64 是精确的，差值也精确，
//!   所以用当日纳秒算出的 300 秒窗口判断与旧代码完全一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、names 无重名
//! ════════════════════════════════════════════════════════════════════
//! 360 个名字是「72 个指标 × 5 个后缀」。abs 版的指标名是 `{序列}_{统计量}_abs`，
//! 与旧 names_single 逐字相同，没有重名。

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

/// 盘口 CSV 的列索引（0 基，与文件表头一致）
const COL_DATETIME: usize = 1;
const COL_EXCHTIME: usize = 4;
const COL_LAST_PRC: usize = 6;
/// 2018 及更早的文件末尾多一列 num（共 62 列），本文件只用到前 7 列
const MAX_COLS: usize = 64;

const SEC_US: i64 = 1_000_000;
/// 当天 0 点起的纳秒数（3 秒标签一律用这个口径，见文件头第四节）
const SEC_NS: i64 = 1_000_000_000;
const DAY_NS: i64 = 86_400 * SEC_NS;
/// deal_data 里的 +8 小时（exchange 微秒是 epoch，显示出来是北京钟点）
const BEIJING_SHIFT_NS: i64 = 28_800 * SEC_NS;
/// 早上开盘 09:30:00
const LD_0930_NS: i64 = 34_200 * SEC_NS;
/// 上午收盘 11:30:00
const LD_1130_NS: i64 = 41_400 * SEC_NS;
/// 下午开盘 13:00:00
const LD_1300_NS: i64 = 46_800 * SEC_NS;
/// 收盘窗口上界 14:57:00
const LD_1457_NS: i64 = 53_820 * SEC_NS;
/// 平移后的下午上界 13:27:00
const LD_1327_NS: i64 = (13 * 3600 + 27 * 60) * SEC_NS;
/// 下午整体平移 90 分钟
const SHIFT_NS: i64 = 5_400 * SEC_NS;
/// 3 秒格宽度
const BIN_NS: i64 = 3 * SEC_NS;

// 五个时段（3 秒表标签的当日纳秒闭区间）
const SESSION_NAMES: [&str; 5] = ["all", "open30", "morning", "afternoon", "close30"];
const SESSION_BOUNDS: [(i64, i64); 5] = [
    (0, DAY_NS),
    (LD_0930_NS, LD_1130_NS),
    (LD_0930_NS, LD_1130_NS), // 与 open30 相同
    (LD_1130_NS, LD_1327_NS),
    (LD_1300_NS, LD_1327_NS),
];

/// 四列残差（顺序即旧代码 son 的列序）
const SERIES4: [&str; 4] = ["resi_de15s", "resi_de60s", "abresi_de15s", "abresi_de60s"];
/// 每个时段的 6 个统计量（顺序即旧代码 res 列表的顺序）
const STATS6: [&str; 6] = ["median", "std", "skew", "kurt", "max", "min"];
/// 相关系数的三组对象（空串 = last_prc）
const CORR_LAGS: [&str; 3] = ["", "_lag1", "_next"];

const PER_SESSION: usize = 72;
pub const N_OUT: usize = PER_SESSION * 5;

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for sess in SESSION_NAMES.iter() {
        for stat in STATS6.iter() {
            for s in SERIES4.iter() {
                out.push(format!("{}_{}_{}", s, stat, sess));
            }
        }
        for stat in STATS6.iter() {
            for s in SERIES4.iter() {
                out.push(format!("{}_{}_abs_{}", s, stat, sess));
            }
        }
        for lag in CORR_LAGS.iter() {
            for s in SERIES4.iter() {
                out.push(format!("corr_{}{}_{}", s, lag, sess));
            }
            for s in SERIES4.iter() {
                out.push(format!("corr_{}_abs{}_{}", s, lag, sess));
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// CSV 读取
// ============================================================================

#[inline]
fn split_into<'a>(bytes: &'a [u8], buf: &mut [&'a [u8]; MAX_COLS]) -> usize {
    let mut start = 0usize;
    let mut n = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b',' {
            if n < MAX_COLS {
                buf[n] = &bytes[start..i];
            }
            n += 1;
            start = i + 1;
        }
    }
    if n < MAX_COLS {
        buf[n] = &bytes[start..];
    }
    (n + 1).min(MAX_COLS)
}

#[inline]
fn parse_f64(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 解析 datetime 列的「当天微秒数」。格式 "YYYY-MM-DD HH:MM:SS[.fff…]"。
/// 解析失败返回 None（旧代码里 pd.to_datetime 会抛异常 → 该股该日整条 NaN）。
fn dt_day_us(b: &[u8]) -> Option<i64> {    let s = std::str::from_utf8(b).ok()?.trim();
    let cut = s.find(|c| c == ' ' || c == 'T')?;
    let time = &s[cut + 1..];
    let mut it = time.splitn(2, '.');
    let hms = it.next()?;
    let frac = it.next().unwrap_or("");
    let mut parts = hms.split(':');
    let h: i64 = parts.next()?.trim().parse().ok()?;
    let m: i64 = parts.next()?.trim().parse().ok()?;
    let sec: i64 = parts.next()?.trim().parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    if !(0..24).contains(&h) || !(0..60).contains(&m) || !(0..=60).contains(&sec) {
        return None;
    }
    let mut us = (h * 3600 + m * 60 + sec) * SEC_US;
    // 小数位只取前 6 位（微秒）；边界判断都落在整秒上，余下位数不影响结果
    let mut scale: i64 = 100_000;
    for &d in frac.as_bytes().iter().filter(|c| c.is_ascii_digit()).take(6) {
        us += ((d - b'0') as i64) * scale;
        scale /= 10;
    }
    Some(us)
}

fn market_path(code: &str, date: i64) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("market_data").join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 读一只股票当天的盘口，复刻到旧代码第 5 步为止。
///
/// 返回 (每组 3 秒格的标签（当天 0 点起的纳秒）, 该格的 last_prc)，都按标签升序，两个向量等长。
///
/// 行序：先上午（exchtime ∈ [09:30:00, 11:30:00]）按文件序，再下午
/// （exchtime ∈ [13:00:00, 14:57:00]，整体 -90 分钟）按文件序；
/// 同一个格里后面的行覆盖前面的行 —— 这就是 resample().last()。
fn read_bins(code: &str, date: i64) -> Option<(Vec<i64>, Vec<f64>)> {
    let path = market_path(code, date)?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut morning: Vec<(i64, f64)> = Vec::new();
    let mut afternoon: Vec<(i64, f64)> = Vec::new();
    let mut lines = content.lines();
    lines.next(); // 表头
    for raw in lines {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; MAX_COLS] = [&[][..]; MAX_COLS];
        let nf = split_into(bytes, &mut f);
        if nf <= COL_LAST_PRC {
            continue;
        }
        let last = parse_f64(f[COL_LAST_PRC]);
        if !(last > 0.0) {
            continue; // NaN 与 0 都在这里被丢掉，和 df[df.last_prc > 0] 一致
        }
        // 第 3 步：按 datetime 字符串解析出来的时刻筛 [09:30:00, 14:57:00]
        let dt_ns = dt_day_us(f[COL_DATETIME])? * 1000;
        if dt_ns < LD_0930_NS || dt_ns > LD_1457_NS {
            continue;
        }
        // 第 4 步：按 exchtime 切上午 / 下午。
        // CSV 里的 exchtime 是 **epoch 微秒**，旧框架 deal_data 给它 +8 小时后当北京钟点用，
        // 所以这里也要 +8 小时再取当天纳秒。
        let ex_us = parse_f64(f[COL_EXCHTIME]);
        if !ex_us.is_finite() {
            continue;
        }
        let ex_ns = (ex_us.round() as i64) * 1000 + BEIJING_SHIFT_NS;
        let day_ns = ex_ns.rem_euclid(DAY_NS);
        if (LD_0930_NS..=LD_1130_NS).contains(&day_ns) {
            morning.push((day_ns, last));
        } else if (LD_1300_NS..=LD_1457_NS).contains(&day_ns) {
            afternoon.push((day_ns - SHIFT_NS, last));
        }
    }

    // 3 秒格：标签 = 当天 0 点起的纳秒数向下取整到 3 秒的整数倍
    let mut bins: BTreeMap<i64, f64> = BTreeMap::new();
    for (t, v) in morning.into_iter().chain(afternoon.into_iter()) {
        bins.insert(t.div_euclid(BIN_NS) * BIN_NS, v);
    }
    let mut labels: Vec<i64> = Vec::with_capacity(bins.len());
    let mut prices: Vec<f64> = Vec::with_capacity(bins.len());
    for (k, v) in bins.into_iter() {
        labels.push(k);
        prices.push(v);
    }
    Some((labels, prices))
}

// ============================================================================
// 中间序列
// ============================================================================

/// rp.rolling_volatility：在 [i-(lookback-1), i] 里按 interval 采样，
/// 对相邻样本取对数收益率，再算**总体**标准差（除以 n，不是 n-1）。
/// 方差不为正时给 0；样本或收益率不够 min_periods 时留 NaN。
fn rolling_vol(prices: &[f64], lookback: usize, interval: usize, min_periods: usize) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    if lookback == 0 || interval == 0 {
        return out;
    }
    let mut samples: Vec<f64> = Vec::with_capacity(lookback / interval + 2);
    let mut rets: Vec<f64> = Vec::with_capacity(lookback / interval + 2);
    for i in 0..n {
        if i < lookback - 1 {
            continue;
        }
        samples.clear();
        let mut pos = i - (lookback - 1);
        loop {
            samples.push(prices[pos]);
            pos += interval;
            if pos > i {
                break;
            }
        }
        if samples.len() < min_periods {
            continue;
        }
        rets.clear();
        for k in 0..samples.len() - 1 {
            rets.push((samples[k + 1] / samples[k]).ln());
        }
        if rets.len() >= min_periods - 1 {
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var =
                rets.iter().map(|&r| (r - mean) * (r - mean)).sum::<f64>() / rets.len() as f64;
            out[i] = if var > 0.0 { var.sqrt() } else { 0.0 };
        }
    }
    out
}

/// rp.ols_residuals(X=[x], y)：带截距的一元回归残差。
/// 样本过滤只按 is_nan（inf 也算有效样本，和旧代码一致）；
/// 正规方程用 src/statistics/mod.rs 里那套 LU 逐步解，避免和旧实现分叉。
fn ols_residual(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    let mut idx: Vec<usize> = Vec::new();
    for i in 0..n {
        if y[i].is_nan() || x[i].is_nan() {
            continue;
        }
        idx.push(i);
    }
    if idx.is_empty() {
        return out;
    }
    let (mut a00, mut a01, mut a11, mut b0, mut b1) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &i in idx.iter() {
        a00 += 1.0;
        a01 += x[i];
        a11 += x[i] * x[i];
        b0 += y[i];
        b1 += x[i] * y[i];
    }
    // 2×2 的 LU 分解（写法与 solve_linear_system3 一致）
    let u00 = a00;
    let u01 = a01;
    let l10 = a01 / u00;
    let u11 = a11 - l10 * u01;
    let y0 = b0;
    let y1 = b1 - l10 * y0;
    let c1 = y1 / u11;
    let c0 = (y0 - u01 * c1) / u00;
    for &i in idx.iter() {
        out[i] = y[i] - (c0 + c1 * x[i]);
    }
    out
}

/// last_prc.rolling_future('300s').last()：取 i 之后、时刻差 <= 300 秒的最后一行的值。
/// 时刻用「当天 0 点起的纳秒」，3 秒标签都是 256 的整数倍，转 f64 精确，差值也精确。
fn ret_next(price: &[f64], label_ns: &[f64]) -> Vec<f64> {
    const WINDOW_NS: f64 = 300.0 * 1e9;
    let n = price.len();
    let mut out = vec![f64::NAN; n];
    let mut window_end = 0usize;
    for i in 0..n {
        if !(i > 0 && window_end > i) {
            window_end = i + 1;
        }
        while window_end < n && label_ns[window_end] - label_ns[i] <= WINDOW_NS {
            window_end += 1;
        }
        if window_end > i + 1 {
            out[i] = price[window_end - 1];
        }
    }
    out
}

// ============================================================================
// pandas 口径的统计量
// ============================================================================

#[inline]
fn is_ok(v: f64) -> bool {
    !v.is_nan()
}

/// nanops 里的 _zero_out_fperr：|v| < 1e-14 归零
#[inline]
fn zero_out_fperr(v: f64) -> f64 {
    if v.abs() < 1e-14 {
        0.0
    } else {
        v
    }
}

fn nan_median(v: &[f64]) -> f64 {
    let mut ok: Vec<f64> = v.iter().copied().filter(|&x| is_ok(x)).collect();
    if ok.is_empty() {
        return f64::NAN;
    }
    ok.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = ok.len();
    if n % 2 == 1 {
        ok[n / 2]
    } else {
        (ok[n / 2 - 1] + ok[n / 2]) / 2.0
    }
}

fn nan_std(v: &[f64]) -> f64 {
    let count = v.iter().filter(|&&x| is_ok(x)).count();
    if count <= 1 {
        return f64::NAN;
    }
    let sum: f64 = v.iter().filter(|&&x| is_ok(x)).sum();
    let mean = sum / count as f64;
    let sq: f64 = v
        .iter()
        .filter(|&&x| is_ok(x))
        .map(|&x| (x - mean) * (x - mean))
        .sum();
    (sq / (count - 1) as f64).sqrt()
}

fn nan_skew(v: &[f64]) -> f64 {
    let count = v.iter().filter(|&&x| is_ok(x)).count();
    if count < 3 {
        return f64::NAN;
    }
    let sum: f64 = v.iter().filter(|&&x| is_ok(x)).sum();
    let mean = sum / count as f64;
    let m2 = zero_out_fperr(
        v.iter()
            .filter(|&&x| is_ok(x))
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>(),
    );
    let m3 = zero_out_fperr(
        v.iter()
            .filter(|&&x| is_ok(x))
            .map(|&x| (x - mean) * (x - mean) * (x - mean))
            .sum::<f64>(),
    );
    if m2 == 0.0 {
        return 0.0;
    }
    let c = count as f64;
    (c * (c - 1.0).sqrt() / (c - 2.0)) * (m3 / m2.powf(1.5))
}

fn nan_kurt(v: &[f64]) -> f64 {
    let count = v.iter().filter(|&&x| is_ok(x)).count();
    if count < 4 {
        return f64::NAN;
    }
    let sum: f64 = v.iter().filter(|&&x| is_ok(x)).sum();
    let mean = sum / count as f64;
    let (mut m2, mut m4) = (0.0f64, 0.0f64);
    for &x in v.iter().filter(|&&x| is_ok(x)) {
        let d2 = (x - mean) * (x - mean);
        m2 += d2;
        m4 += d2 * d2;
    }
    let c = count as f64;
    let adj = 3.0 * (c - 1.0) * (c - 1.0) / ((c - 2.0) * (c - 3.0));
    let numerator = zero_out_fperr(c * (c + 1.0) * (c - 1.0) * m4);
    let denominator = zero_out_fperr((c - 2.0) * (c - 3.0) * m2 * m2);
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator - adj
}

fn nan_max(v: &[f64]) -> f64 {
    let mut best = f64::NAN;
    let mut seen = false;
    for &x in v.iter().filter(|&&x| is_ok(x)) {
        if !seen || x > best {
            best = x;
            seen = true;
        }
    }
    best
}

fn nan_min(v: &[f64]) -> f64 {
    let mut best = f64::NAN;
    let mut seen = false;
    for &x in v.iter().filter(|&&x| is_ok(x)) {
        if !seen || x < best {
            best = x;
            seen = true;
        }
    }
    best
}

/// Series.corr → nanops.nancorr → np.corrcoef(a, b)[0, 1]
/// （成对完整观测；n<2 → NaN；任一侧方差不为正 → NaN；结果 clip 到 [-1, 1]）
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..x.len() {
        if !is_ok(x[i]) || !is_ok(y[i]) {
            continue;
        }
        n += 1;
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        syy += y[i] * y[i];
        sxy += x[i] * y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let cov = sxy - sx * sy / nf;
    let vx = sxx - sx * sx / nf;
    let vy = syy - sy * sy / nf;
    if !(vx > 0.0) || !(vy > 0.0) {
        return f64::NAN;
    }
    let r = cov / (vx.sqrt() * vy.sqrt());
    if r > 1.0 {
        1.0
    } else if r < -1.0 {
        -1.0
    } else {
        r
    }
}

// ============================================================================
// 逐股入口
// ============================================================================

struct Cols {
    price: Vec<f64>,
    lag1: Vec<f64>,
    next: Vec<f64>,
    res: [Vec<f64>; 4],
    abs: [Vec<f64>; 4],
}

/// 一个时段（用行下标给出）的 72 个值，顺序见文件头第二节。
fn session_metrics(c: &Cols, idx: &[usize]) -> [f64; PER_SESSION] {
    let mut out = [f64::NAN; PER_SESSION];
    let gather = |src: &[f64]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };

    let stats: [fn(&[f64]) -> f64; 6] = [nan_median, nan_std, nan_skew, nan_kurt, nan_max, nan_min];
    let son: Vec<Vec<f64>> = (0..4).map(|k| gather(&c.res[k])).collect();
    let son_abs: Vec<Vec<f64>> = (0..4).map(|k| gather(&c.abs[k])).collect();

    let mut p = 0usize;
    for st in stats.iter() {
        for col in son.iter() {
            out[p] = st(col);
            p += 1;
        }
    }
    for st in stats.iter() {
        for col in son_abs.iter() {
            out[p] = st(col);
            p += 1;
        }
    }
    let partners: [Vec<f64>; 3] = [gather(&c.price), gather(&c.lag1), gather(&c.next)];
    for partner in partners.iter() {
        for col in son.iter() {
            out[p] = pearson(partner, col);
            p += 1;
        }
        for col in son_abs.iter() {
            out[p] = pearson(partner, col);
            p += 1;
        }
    }
    debug_assert_eq!(p, PER_SESSION);
    out
}

/// 一只股票的一天：算出 360 个因子值。算不出来返回 None（上游补整条 NaN）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let (labels, price) = read_bins(code, date)?;
    let n = price.len();

    let vol3 = rolling_vol(&price, 100, 1, 2);
    let vol15 = rolling_vol(&price, 100, 5, 2);
    let vol60 = rolling_vol(&price, 100, 20, 2);
    let label_ns: Vec<f64> = labels.iter().map(|&t| t as f64).collect();

    // ret_next = rolling_future('300s').last() / last_prc - 1（NaN 一路透传）
    let nx: Vec<f64> = ret_next(&price, &label_ns)
        .iter()
        .zip(price.iter())
        .map(|(&v, &p)| v / p - 1.0)
        .collect();
    let mut lag1 = vec![f64::NAN; n];
    for i in 0..n.saturating_sub(1) {
        lag1[i] = price[i + 1] / price[i] - 1.0;
    }

    let res = [
        ols_residual(&vol15, &vol3),
        ols_residual(&vol60, &vol3),
        ols_residual(&vol3, &vol15),
        ols_residual(&vol3, &vol60),
    ];
    let abs = [
        res[0].iter().map(|v| v.abs()).collect::<Vec<f64>>(),
        res[1].iter().map(|v| v.abs()).collect::<Vec<f64>>(),
        res[2].iter().map(|v| v.abs()).collect::<Vec<f64>>(),
        res[3].iter().map(|v| v.abs()).collect::<Vec<f64>>(),
    ];

    let cols = Cols { price, lag1, next: nx, res, abs };

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for &(lo, hi) in SESSION_BOUNDS.iter() {
        let idx: Vec<usize> = (0..n).filter(|&i| labels[i] >= lo && labels[i] <= hi).collect();
        out.extend_from_slice(&session_metrics(&cols, &idx));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm52 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
