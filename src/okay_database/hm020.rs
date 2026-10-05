//! hm20「撤单数据观察」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm20_撤单数据观察/hm20_okay.py 的 run_base。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码的框架参数与分段
//! ════════════════════════════════════════════════════════════════════
//!   get_many=0、read_full=0、change_time=1
//!   → read_trade_data_single 返回**一整天的单张** DataFrame，不分段。
//!   change_time=1 只做两件事：加一列 delay = localtime - exchtime（单位微秒），
//!   并把 exchtime / localtime 两个数值列转成 Timestamp（数值微秒 + 8 小时）。
//!   全文件只有「09:30:00 ~ 14:57:00」一个窗口，**没有**调 adjust_afternoon，
//!   所以窗口用的是真实钟点，不做下午平移。
//!
//!   with_retreat=1 → 读进来时**保留** flag==32 的撤单行（本系列主题就是撤单）；
//!   紧接着自己 `df = df[df.flag == 32]`，所以最终参与计算的只有撤单行。
//!
//! 二、逐步算式（顺序不能变，逐条对应旧代码第 33~88 行）
//! ────────────────────────────────────────────────────────────────
//!   1. df = df.set_index('exchtime').between_time('09:30:00','14:57:00')
//!      两端都含、微秒精度（不能按整秒截断）
//!   2. df = df[df.flag == 32]                        只留撤单行
//!   3. q95  = df.delay.quantile(0.95)                线性插值，NaN 不参与
//!      high = df.delay >= q95
//!      delay_volume = high.replace(False, nan).replace(True, 1) * df.volume
//!                   = 高延迟的行留 volume，其余为 NaN
//!      ！分位数是在**窗口内的撤单子集**上取的，不是全窗口
//!   4. ask_total_volume = df[df.bid_order == 0].volume.sum()   ← 原始 volume，不是 delay_volume
//!      bid_total_volume = df[df.ask_order == 0].volume.sum()
//!      （bid_order==0 表示这笔撤单撤的是卖单，所以这个和正好是 asks 的分母）
//!   5. df.ask_total_volume = df.delay_volume / ask_total_volume
//!      df.bid_total_volume = df.delay_volume / bid_total_volume
//!      asks = df[df.ask_order != 0] 取 ask_total_volume 这列改名叫 total_volume
//!      bids = df[df.bid_order != 0] 取 bid_total_volume 这列改名叫 total_volume
//!      行序 = 文件里的原始顺序（set_index / between_time 都不改变行序）
//!   6. 对 asks.total_volume 与 bids.total_volume 两条序列各算：
//!        sum（跳过 NaN）
//!        autocorr(1)、autocorr(2)
//!        |autocorr(1)| … |autocorr(5)|
//!      autocorr(k) 在 pandas 里就是 `self.corr(self.shift(k))`，
//!      即序列与自身左移 k 位的 Pearson（成对完整观测），
//!      等价于 pearson(v[..n-k], v[k..])；k >= n 时 shift 全 NaN → NaN。
//!
//!   三、输出顺序（= 旧代码 names_single 的顺序，names_asks + names_bids）
//! ────────────────────────────────────────────────────────────────
//!     ask_total_volume_sum, ask_total_volume_corr_lag1, ask_total_volume_corr_lag2,
//!     ask_abs_total_volume_corr_lag1..lag5,
//!     bid_ 同上 8 个                                       共 16 个
//!
//!   names_save（旧文件里要保存的 base 名，写在 hm20_okay_database.py 里）：
//!     bid_total_volume_corr_lag2, ask_total_volume_corr_lag1, bid_total_volume_corr_lag1,
//!     ask_total_volume_corr_lag2, ask_total_volume_sum, bid_abs_total_volume_corr_lag1,
//!     bid_abs_total_volume_corr_lag4, bid_total_volume_sum, ask_abs_total_volume_corr_lag1,
//!     ask_abs_total_volume_corr_lag5
//!   names_in_db：bid_total_volume_sum, ask_total_volume_sum
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、这里为什么不用 fast_csv_reader 的 TradeRecord，而是自己按 f64 解析 CSV
//! ════════════════════════════════════════════════════════════════════
//!   两个理由，缺一不可：
//!   1. 旧代码的 delay = localtime - exchtime 用的是 **localtime 列**，
//!      而 TradeRecord 里没有 localtime 字段，只能自己解析这一列。
//!   2. delay 是「两个 epoch 微秒大数相减」再拿去和分位数比大小。
//!      2025 年的 exchtime 是 1.74e15 量级，f32 在 1e15 上的精度是 ±1e8（约 100 秒），
//!      用它算出来的 delay 完全是垃圾；f64（以及 i64）才能保证相减精确到 1 微秒。
//!      同理 volume 也是逐行累加再相除，全程按 f64 走。
//!   列索引与 fast_csv_reader 的常量保持一致（2018 年及以前末尾多一列 num，不影响前 15 列）。

use std::fs::File;
use std::io::Read;

/// 撤单标记：成交明细里 flag==32 表示这一行是撤单记录
const FLAG_RETREAT: i64 = 32;

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
const EIGHT_HOURS_US: i64 = 8 * 3600 * SEC_US;

/// 窗口 09:30:00 ~ 14:57:00，真实钟点、闭区间、微秒精度
const LO_CLOCK_US: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const HI_CLOCK_US: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 每个方向 8 个指标，asks + bids 共 16 个
pub const N_OUT: usize = 16;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致）
const COL_EXCHTIME: usize = 4;
const COL_LOCALTIME: usize = 5;
const COL_VOLUME: usize = 8;
const COL_FLAG: usize = 10;
const COL_ASK_ORDER: usize = 13;
const COL_BID_ORDER: usize = 14;
const NCOL: usize = 15;

/// 每个方向的指标名后缀，顺序与 compute_stock 的输出一致
const SUFFIX: [&str; 8] = [
    "total_volume_sum",
    "total_volume_corr_lag1",
    "total_volume_corr_lag2",
    "abs_total_volume_corr_lag1",
    "abs_total_volume_corr_lag2",
    "abs_total_volume_corr_lag3",
    "abs_total_volume_corr_lag4",
    "abs_total_volume_corr_lag5",
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for side in ["ask", "bid"] {
        for s in SUFFIX.iter() {
            out.push(format!("{}_{}", side, s));
        }
    }
    out
}

/// 北京时间的当日微秒数。
/// 旧代码的 Timestamp 是「数值微秒 + 8 小时」，取时刻就是 (exchtime_us + 8h) mod 一天。
#[inline]
fn clock_us(exchtime_us: i64) -> i64 {
    (exchtime_us + EIGHT_HOURS_US).rem_euclid(DAY_US)
}

// ============================================================================
// 逐笔 CSV 的 f64 读取
// ============================================================================

/// 一条逐笔成交（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    exchtime_us: i64,
    /// localtime - exchtime，单位微秒（旧代码的 delay 列）
    delay: f64,
    volume: f64,
    flag: i64,
    ask_order: i64,
    bid_order: i64,
}

#[inline]
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

/// 空单元格按 pandas 的读法给 NaN
#[inline]
fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 读 transaction CSV（f64）。
///
/// 返回 None 只表示「文件找不到或读不了」，与旧代码里 HfDataReader 返回 None 一致
/// （上游会补 NaN）。文件存在但没有数据行时返回空 Vec，此时旧代码照样返回 16 个值
/// （sum 为 0、autocorr 为 NaN），所以这里不能把空表当成 None。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; NCOL] = [&[][..]; NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < NCOL {
            f[col] = &bytes[start..];
        }
        let exchtime = parse_f64_at(f[COL_EXCHTIME]);
        let localtime = parse_f64_at(f[COL_LOCALTIME]);
        out.push(TradeRow {
            exchtime_us: exchtime as i64,
            delay: localtime - exchtime,
            volume: parse_f64_at(f[COL_VOLUME]),
            flag: parse_i64_at(f[COL_FLAG]),
            ask_order: parse_i64_at(f[COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[COL_BID_ORDER]),
        });
    }
    Some(out)
}

// ============================================================================
// 统计量
// ============================================================================

/// pandas 的 `Series.quantile(q, interpolation='linear')`：
/// 先去掉 NaN，升序排序，位置 pos = q*(n-1) 处线性插值。
/// n==1 时直接返回该值；空序列返回 NaN。
///
/// 这里算出来的分位数只用来和 delay 比大小，而 delay 全是整数微秒，
/// 插值结果落在相邻两个整数之间，所以末尾几个比特的差异不会改变比较结果。
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor();
    let lo_i = lo as usize;
    if lo_i + 1 >= n {
        return sorted[n - 1];
    }
    let frac = pos - lo;
    sorted[lo_i] + (sorted[lo_i + 1] - sorted[lo_i]) * frac
}

/// pandas 的 `Series.sum()`：跳过 NaN，空序列或全 NaN 返回 0.0
fn sum_skipna(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v.iter() {
        if !x.is_nan() {
            s += x;
        }
    }
    s
}

/// pandas 的 `Series.corr`（method='pearson'）落到 nanops.nancorr + np.corrcoef：
///   · 有效样本 = 两个都不是 NaN 的位置（inf 也算有效，notna 不管它）
///   · 有效样本 < 1 直接 NaN；样本恰好 1 个时 np.corrcoef 也是 NaN
///   · 先各减自己的均值再算协方差（两遍法，不是 sum(x²)-sum(x)²/n 那种一遍公式）
///   · 任一侧方差为 0 或出现 inf（均值不再是有限数）都返回 NaN
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut mx = 0.0f64;
    let mut my = 0.0f64;
    for i in 0..x.len() {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        n += 1;
        mx += x[i];
        my += y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    mx /= nf;
    my /= nf;
    // inf 参与时 np.corrcoef 的结果必然是 NaN（inf - inf）
    if !mx.is_finite() || !my.is_finite() {
        return f64::NAN;
    }
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..x.len() {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let ddof = (n - 1) as f64;
    let vx = sxx / ddof;
    let vy = syy / ddof;
    let cov = sxy / ddof;
    let d = vx.sqrt();
    (cov / d) / vy.sqrt()
}

/// pandas 的 `Series.autocorr(lag)` = `self.corr(self.shift(lag))`。
/// shift 之后索引不变，配对是位置对齐的，等价于 pearson(v[..n-lag], v[lag..])；
/// lag >= n 时 shift 全是 NaN，nancorr 返回 NaN。
fn autocorr(v: &[f64], lag: usize) -> f64 {
    let n = v.len();
    if lag >= n {
        return f64::NAN;
    }
    pearson(&v[..n - lag], &v[lag..])
}

// ============================================================================
// 入口
// ============================================================================

/// 逐股入口。旧代码只有「读不到文件」才返回 None，别的一律返回 16 个值，
/// 所以这里也不因为子集为空而返回 None（上游对 None 补 NaN，两边要能对上）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let all = read_trade_f64(code, date)?;

    // 1 + 2：窗口（闭区间，微秒精度）内的撤单行，保持文件原始顺序
    let rows: Vec<&TradeRow> = all
        .iter()
        .filter(|r| {
            let c = clock_us(r.exchtime_us);
            c >= LO_CLOCK_US && c <= HI_CLOCK_US
        })
        .filter(|r| r.flag == FLAG_RETREAT)
        .collect();

    // 3：delay 的 0.95 分位数（线性插值），以及 high_delay 判定
    let mut delays: Vec<f64> = rows
        .iter()
        .map(|r| r.delay)
        .filter(|d| d.is_finite())
        .collect();
    delays.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q95 = quantile_linear(&delays, 0.95);

    // 4：两个分母。旧代码用的是原始 volume（不是 delay_volume），sum 跳过 NaN
    let mut ask_denom = 0.0f64; // df[bid_order == 0].volume.sum()，供 asks 用
    let mut bid_denom = 0.0f64; // df[ask_order == 0].volume.sum()，供 bids 用
    for r in rows.iter() {
        if r.volume.is_nan() {
            continue;
        }
        if r.bid_order == 0 {
            ask_denom += r.volume;
        }
        if r.ask_order == 0 {
            bid_denom += r.volume;
        }
    }

    // 5：delay_volume / 分母 → 两条 total_volume 序列，行序与文件一致
    let mut ask_vals: Vec<f64> = Vec::with_capacity(rows.len());
    let mut bid_vals: Vec<f64> = Vec::with_capacity(rows.len());
    for r in rows.iter() {
        // q95 为 NaN 时比较恒为 false，与 pandas 一致
        let high = r.delay >= q95;
        let dv = if high { r.volume } else { f64::NAN };
        if r.ask_order != 0 {
            ask_vals.push(dv / ask_denom);
        }
        if r.bid_order != 0 {
            bid_vals.push(dv / bid_denom);
        }
    }

    // 6：每个方向的 8 个指标，顺序必须与 names() 一致
    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for vals in [&ask_vals, &bid_vals] {
        out.push(sum_skipna(vals) as f32);
        out.push(autocorr(vals, 1) as f32);
        out.push(autocorr(vals, 2) as f32);
        for lag in 1..=5usize {
            out.push(autocorr(vals, lag).abs() as f32);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm20 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
