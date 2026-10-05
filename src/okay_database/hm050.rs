//! hm50「昨日重现」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm50_昨日重现/hm50_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   with_will=1 / with_will_back_days=1 —— 因子 = 今天与**上一个交易日**的盘口帧做
//!   相关与 DTW。旧框架的 will 机制在 design_whatever_out.cal_factor_one_stock 里：
//!   先把回看日跑一遍（只写 self.will、返回 None），再把 will 传给目标日。
//!   Rust 侧 compute_stock(code, date) 没有 will 入参，所以自己回看上一个交易日、
//!   把两天的盘口各读一遍再算 —— 语义等价（与 src/factor_pipeline.rs 的
//!   pipeline_order_pair_hm90 第 92~104 行同一套做法）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()
//!      · 索引 = pd.to_datetime(data["datetime"])（毫秒精度的盘口时间，已含 +8h）
//!      · change_time=1 → deal_data 把 exchtime/localtime 列转成 Timestamp，
//!        但**索引不变**；下面所有 between_time / resample 走的都是 datetime 这一列。
//! 2) df = df.between_time("09:30", "14:57")                 # 两端都含，微秒精度
//! 3) df1 = df.between_time("09:30:00", "11:30:00")
//!    df2 = df.between_time("13:00:00", "14:57:00")
//!    df2.index = df2.index - 90min                          # 下午整体前移 90 分钟
//!    df = pd.concat([df1, df2])                             # 索引连成 09:30~13:27
//!    ！！11:30:00 会同时出现在 df1 末尾和 df2 开头（前移后），索引里有重复值，
//!      但仍是单调不减；resample 按「同一 3 秒桶里最后一个」取，行为不受影响。
//! 4) df_open      = df.between_time("09:30:00", "10:00:00")   # 上午头 30 分钟
//!    df_morning   = df.between_time("09:30:00", "11:30:00")
//!    df_afternoon = df.between_time("11:30:00", "13:27:00")   # 前移后 = 真实 13:00~14:57
//!    df_close     = df.between_time("13:00:00", "13:27:00")   # 前移后 = 真实 14:30~14:57
//! 5) df_3s = df.resample("3s").last()                        # origin='start_day'，左闭
//!      · 桶是「距午夜 3 秒的整数倍」，空桶保留、值为 NaN（.last() 取桶内最后一个
//!        非空值，这一步本项目数据里 volume 无空值，仍照抄）
//!      · df_3s_open      = df_3s.between_time("09:30:00","09:57:00").reset_index(drop=True)
//!        df_3s_morning   = df_3s.between_time("09:30:00","11:27:00").reset_index(drop=True)
//!        df_3s_afternoon = df_3s.between_time("11:30:00","13:27:00").reset_index(drop=True)
//!        df_3s_close     = df_3s.between_time("13:00:00","13:27:00").reset_index(drop=True)
//!        df_3s           = df_3s.reset_index(drop=True)
//!      四个切片都 reset_index(drop=True)，所以后面 Series.corr 的索引对齐是**按位置**
//!      取前 min(len) 个（RangeIndex 内连接）。
//! 6) get_corr(yes, today)：
//!      corr_volume      = yes.volume.corr(today.volume)
//!      corr_volume_diff = yes.volume.diff().corr(today.volume.diff())
//!    get_dtw(yes, today)：
//!      dtw_volume      = rp.fast_dtw_distance(yes.volume.dropna(), today.volume.dropna(), 1, 2)
//!      dtw_volume_diff = rp.fast_dtw_distance(yes.volume.diff().dropna(),
//!                                             today.volume.diff().dropna(), 1, 2)
//! 7) 输出 18 个值，顺序 = 旧 names（见下面 names()）：
//!      7 组 corr（各 2 个，volume / volume.diff） + 2 组 dtw（各 2 个）
//!    配对关（yes→today）依次是：
//!      df_3s_open→df_3s_open, df_3s_morning→df_3s_morning, df_3s_close→df_3s_close,
//!      df_3s_afternoon→df_3s_afternoon, df_3s_close→df_3s_open,
//!      df_3s_afternoon→df_3s_morning, df_3s→df_3s,
//!      dtw: df_close→df（全天原始帧）, dtw: df_afternoon→df
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度与已知不可复现
//! ════════════════════════════════════════════════════════════════════
//! · 盘口价格/量一律在本文件里按 f64 解析 CSV（不用 fast_csv_reader 的 f32）：
//!   DTW 的量级到 1e11，f32 只有 7 位有效数字，必然超 1e-4。
//! · pandas Series.corr 走 nanops.nancorr → np.corrcoef，本质是
//!   c01/(sqrt(c00)*sqrt(c11))，其中三个点积由 BLAS dgemm 累加。这里用 numpy 的
//!   成对求和（pairwise_sum，blocksize 128、8 路展开）代替 BLAS，二者只差最后几位，
//!   相关系数量级 O(1) 时相对差 ~1e-15，不影响 1e-4 验收。
//! · fast_dtw_distance 是 #[pyfunction]，worker 链路不能调（不带 libpython），
//!   这里按 src/time_series/mod.rs 的 compute_dtw + radius 自动放宽带逐行抄成纯 Rust。

use std::fs::File;
use std::io::Read;

/// 盘口 CSV 的列索引（0 基）
const MKT_COL_DATETIME: usize = 1;
const MKT_COL_VOLUME: usize = 14;
const MKT_NCOL: usize = 61;

const US_SEC: i64 = 1_000_000;
const US_MIN: i64 = 60 * US_SEC;

/// 一整天里用到的微秒时刻（真实钟点）。下午帧在 df 里已前移 90 分钟，
/// 但窗口判断用的都是「前移之后」的时刻，所以下面前缀 LD_ 表示 df 坐标。
const LD_0930: i64 = (9 * 3600 + 30 * 60) * US_SEC;
const LD_0957: i64 = (9 * 3600 + 57 * 60) * US_SEC;
const LD_1127: i64 = (11 * 3600 + 27 * 60) * US_SEC;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * US_SEC;
const LD_1300: i64 = 13 * 3600 * US_SEC;
const LD_1327: i64 = (13 * 3600 + 27 * 60) * US_SEC;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * US_SEC;
const AFTERNOON_SHIFT: i64 = 90 * US_MIN;

pub const N_OUT: usize = 18;

pub fn names() -> Vec<String> {
    let corr = ["corr_volume", "corr_volume_diff"];
    let dtw = ["dtw_volume", "dtw_volume_diff"];
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for s in [
        "corr_yes_open_today_open",
        "corr_yes_morning_today_morning",
        "corr_yes_close_today_close",
        "corr_yes_afternoon_today_afternoon",
        "corr_yes_close_today_open",
        "corr_yes_afternoon_today_morning",
        "corr_yes_full_today_full",
    ] {
        for p in corr.iter() {
            out.push(format!("{}_{}", p, s));
        }
    }
    for s in ["dtw_yes_close_today_full", "dtw_yes_afternoon_today_full"] {
        for p in dtw.iter() {
            out.push(format!("{}_{}", p, s));
        }
    }
    out
}

/// 一个盘口快照（只留算因子要用的列）。
#[derive(Clone, Copy, Default)]
struct Row {
    /// df 坐标（下午前移 90 分钟）的当日微秒时刻
    t_us: i64,
    vol: f64,
}

// ============================================================================
// CSV 读取
// ============================================================================

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

/// 从 `datetime` 列取出「当日微秒数」。格式形如 2025-03-04 08:15:09.000
/// （2018 年前后没有小数部分，按 .000 处理）。
fn parse_datetime_us(b: &[u8]) -> Option<i64> {
    let s = std::str::from_utf8(b).ok()?;
    let s = s.trim();
    let sp = s.find(' ')?;
    let t = &s[sp + 1..];
    let tb = t.as_bytes();
    if tb.len() < 8 {
        return None;
    }
    let h: i64 = t.get(0..2)?.parse().ok()?;
    let mi: i64 = t.get(3..5)?.parse().ok()?;
    let se: i64 = t.get(6..8)?.parse().ok()?;
    let ms: i64 = if tb.len() >= 12 && tb[8] == b'.' {
        let frac = &t[9..];
        let digits = frac.len().min(6);
        let mut v: i64 = 0;
        for c in frac.as_bytes()[..digits].iter() {
            if !c.is_ascii_digit() {
                break;
            }
            v = v * 10 + (c - b'0') as i64;
        }
        // 补足到 6 位小数
        for _ in digits..6 {
            v *= 10;
        }
        v
    } else {
        0
    };
    Some(((h * 3600 + mi * 60 + se) * 1_000_000) + ms)
}

/// 读盘口 CSV（f64），返回 df 坐标（下午前移）的行序列，保持文件原始行序。
/// 先按 [09:30, 14:57] 过滤，再拆上午/下午两段、下午前移 90 分钟拼回去。
fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("market_data").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut morning: Vec<Row> = Vec::new();
    let mut afternoon: Vec<Row> = Vec::new();
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; MKT_NCOL] = [&[][..]; MKT_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < MKT_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < MKT_NCOL {
            f[col] = &bytes[start..];
        }
        let t = match parse_datetime_us(f[MKT_COL_DATETIME]) {
            Some(t) => t,
            None => continue,
        };
        // between_time("09:30", "14:57")，用微秒比较（与 pandas _get_time_micros 一致）
        if t < LD_0930 || t > LD_1457 {
            continue;
        }
        let row = Row {
            t_us: t,
            vol: parse_f64_at(f[MKT_COL_VOLUME]),
        };
        if t <= LD_1130 {
            morning.push(row);
        } else if t >= LD_1300 {
            afternoon.push(Row {
                t_us: t - AFTERNOON_SHIFT,
                ..row
            });
        }
    }
    morning.extend(afternoon);
    Some(morning)
}

// ============================================================================
// numpy 成对求和与 pandas Series.corr
// ============================================================================

const PW_BLOCKSIZE: usize = 128;

/// numpy 的成对求和（numpy/_core/src/umath/loops_utils.h 的 pairwise_sum_DOUBLE：
/// blocksize 128、8 路展开；n<8 顺序累加且初值 -0.0）。
fn pw_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &v in a.iter() {
            res += v;
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8usize;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
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

fn np_mean(a: &[f64]) -> f64 {
    pw_sum(a) / a.len() as f64
}

/// pandas Series.corr（pearson）：
///   align(join="inner") 后按位置取前 min(n) 个 → nancorr 丢掉任一为 NaN 的对
///   → np.corrcoef[0,1] = (c01/sqrt(c00))/sqrt(c11)，其中 c = dot/(k-1)。
/// 有效对 < 2 时返回 NaN（k==0 是 min_periods 判定，k==1 是 0*inf）。
fn corr_np(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut xs: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let k = xs.len();
    if k < 2 {
        return f64::NAN;
    }
    let mx = np_mean(&xs);
    let my = np_mean(&ys);
    let mut xx: Vec<f64> = Vec::with_capacity(k);
    let mut yy: Vec<f64> = Vec::with_capacity(k);
    let mut xy: Vec<f64> = Vec::with_capacity(k);
    for i in 0..k {
        let a = xs[i] - mx;
        let b = ys[i] - my;
        xx.push(a * a);
        yy.push(b * b);
        xy.push(a * b);
    }
    let s = 1.0f64 / (k as f64 - 1.0);
    let c00 = pw_sum(&xx) * s;
    let c11 = pw_sum(&yy) * s;
    let c01 = pw_sum(&xy) * s;
    let d0 = c00.sqrt();
    let d1 = c11.sqrt();
    let r = (c01 / d0) / d1;
    if r.is_nan() {
        r
    } else {
        r.clamp(-1.0, 1.0)
    }
}

// ============================================================================
// fast_dtw_distance 的纯 Rust 复刻（src/time_series/mod.rs）
// ============================================================================

fn compute_dtw(s1: &[f64], s2: &[f64], radius: Option<usize>) -> f64 {
    let n1 = s1.len();
    let n2 = s2.len();
    if let Some(r) = radius {
        // 终点 (n1,n2) 的带宽条件，达不到就一定是 INF，省掉一次大分配
        if (n1 as i64 - n2 as i64).unsigned_abs() as usize > r {
            return f64::INFINITY;
        }
    }
    let w = n2 + 1;
    let mut mat = vec![f64::INFINITY; (n1 + 1) * w];
    mat[0] = 0.0;
    for i in 1..=n1 {
        for j in 1..=n2 {
            if let Some(r) = radius {
                if !(i.saturating_sub(r) <= j && j <= i + r) {
                    continue;
                }
            }
            let cost = (s1[i - 1] - s2[j - 1]).abs();
            let best = mat[(i - 1) * w + j]
                .min(mat[i * w + j - 1].min(mat[(i - 1) * w + j - 1]));
            mat[i * w + j] = cost + best;
        }
    }
    mat[n1 * w + n2]
}

/// 复刻 rp.fast_dtw_distance(s1, s2, radius=1, timeout=2) 的 radius 自动放宽逻辑。
/// 超时那条路径不实现（2 秒足够，实测 DTW 单次不到 10ms）。
fn fast_dtw(s1: &[f64], s2: &[f64], radius: usize) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return f64::NAN;
    }
    let maxlen = s1.len().max(s2.len());
    let mut cur = radius;
    let mut res = compute_dtw(s1, s2, Some(cur));
    let mut attempt = 0;
    while res.is_infinite() && attempt < 3 {
        attempt += 1;
        let new_radius = match attempt {
            1 => maxlen / 4,
            2 => maxlen / 2,
            _ => maxlen,
        };
        if new_radius <= cur {
            continue;
        }
        cur = new_radius;
        res = compute_dtw(s1, s2, Some(cur));
    }
    res
}

// ============================================================================
// 数据准备
// ============================================================================

/// 按微秒闭区间取 volume 序列（between_time 两端都含）。
fn slice(rows: &[Row], lo: i64, hi: i64) -> Vec<f64> {
    rows.iter()
        .filter(|r| r.t_us >= lo && r.t_us <= hi)
        .map(|r| r.vol)
        .collect()
}

/// 差分（pandas Series.diff()：第 0 位是 NaN）。
fn diff(v: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    out.push(f64::NAN);
    for i in 1..v.len() {
        out.push(v[i] - v[i - 1]);
    }
    out
}

fn dropna(v: &[f64]) -> Vec<f64> {
    v.iter().copied().filter(|x| !x.is_nan()).collect()
}

/// df.resample("3s").last()：桶按「距午夜 3 秒整数倍」对齐（origin='start_day'、左闭），
/// 首末之间每个桶都保留（空桶为 NaN），桶内取最后一个非空值。
/// 返回 (桶左端点微秒, 值)。
fn resample_3s(rows: &[Row]) -> Vec<(i64, f64)> {
    const BIN: i64 = 3 * US_SEC;
    if rows.is_empty() {
        return Vec::new();
    }
    // pandas _get_time_bins：first=ax.min()、last=ax.max()，origin='start_day'（当天午夜，
    // 在「当日微秒」坐标里就是 0），closed='left'。
    //   lo = first - ((first - 0) % 3s)              → 不超过 first 的最大 3 秒整点
    //   hi = last + (3s - last % 3s)  或 last + 3s   → 严格大于 last 的最小 3 秒整点
    //   binner = lo, lo+3s, ..., hi（共 nb 个）
    //   标签 = binner[:-1]（label='left' 且 len(bins)=nb-1 时截一位），所以输出 nb-1 行
    let first = rows.iter().map(|r| r.t_us).min().unwrap();
    let last = rows.iter().map(|r| r.t_us).max().unwrap();
    let lo = first.div_euclid(BIN) * BIN;
    let loff = last.rem_euclid(BIN);
    let hi = if loff > 0 { last + (BIN - loff) } else { last + BIN };
    let nbin = ((hi - lo) / BIN) as usize;
    let mut vals = vec![f64::NAN; nbin];
    for r in rows.iter() {
        let idx = ((r.t_us - lo) / BIN) as usize;
        if r.vol.is_nan() {
            continue; // group_last 跳过空值，只让非空值覆盖
        }
        if idx < nbin {
            vals[idx] = r.vol;
        }
    }
    (0..nbin).map(|i| (lo + i as i64 * BIN, vals[i])).collect()
}

/// 在 3 秒桶序列上按微秒闭区间取 volume。
fn slice_bins(bins: &[(i64, f64)], lo: i64, hi: i64) -> Vec<f64> {
    bins.iter()
        .filter(|(t, _)| *t >= lo && *t <= hi)
        .map(|(_, v)| *v)
        .collect()
}

/// 一天的 df 坐标帧集合
struct Frames {
    full: Vec<f64>,
    close: Vec<f64>,
    afternoon: Vec<f64>,
    b3_full: Vec<f64>,
    b3_open: Vec<f64>,
    b3_morning: Vec<f64>,
    b3_close: Vec<f64>,
    b3_afternoon: Vec<f64>,
}

fn build(rows: &[Row]) -> Frames {
    let bins = resample_3s(rows);
    Frames {
        full: slice(rows, LD_0930, LD_1327),
        close: slice(rows, LD_1300, LD_1327),
        afternoon: slice(rows, LD_1130, LD_1327),
        b3_full: slice_bins(&bins, LD_0930, LD_1327),
        b3_open: slice_bins(&bins, LD_0930, LD_0957),
        b3_morning: slice_bins(&bins, LD_0930, LD_1127),
        b3_close: slice_bins(&bins, LD_1300, LD_1327),
        b3_afternoon: slice_bins(&bins, LD_1130, LD_1327),
    }
}

// ============================================================================
// 交易日历
// ============================================================================

/// 读 basic_info/calendar.csv，返回升序交易日列表（进程内缓存一次）。
pub fn trading_days() -> &'static [i64] {
    static DAYS: std::sync::OnceLock<Vec<i64>> = std::sync::OnceLock::new();
    DAYS.get_or_init(|| {
        let dir = crate::data_paths::basic_info_path("");
        match std::fs::read_to_string(dir.join("calendar.csv")) {
            Ok(s) => s
                .lines()
                .filter_map(|l| l.trim().parse::<i64>().ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    })
}

/// 严格小于 date 的最大交易日（对齐 rp.td.last_trading_day）。
pub fn last_trading_day(days: &[i64], date: i64) -> Option<i64> {
    let idx = days.partition_point(|&d| d < date);
    if idx == 0 {
        None
    } else {
        Some(days[idx - 1])
    }
}

// ============================================================================
// 入口
// ============================================================================

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let days = trading_days();
    let prev = last_trading_day(&days, date)?;

    let rows_y = read_rows(code, prev)?;
    let rows_t = read_rows(code, date)?;
    let y = build(&rows_y);
    let t = build(&rows_t);

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    // 7 组 corr
    for (a, b) in [
        (&y.b3_open, &t.b3_open),
        (&y.b3_morning, &t.b3_morning),
        (&y.b3_close, &t.b3_close),
        (&y.b3_afternoon, &t.b3_afternoon),
        (&y.b3_close, &t.b3_open),
        (&y.b3_afternoon, &t.b3_morning),
        (&y.b3_full, &t.b3_full),
    ] {
        out.push(corr_np(a, b));
        out.push(corr_np(&diff(a), &diff(b)));
    }
    // 2 组 dtw（用原始帧，非 3 秒重采样）
    for (a, b) in [(&y.close, &t.full), (&y.afternoon, &t.full)] {
        out.push(fast_dtw(&dropna(a), &dropna(b), 1));
        out.push(fast_dtw(&dropna(&diff(a)), &dropna(&diff(b)), 1));
    }

    Some(out.into_iter().map(|v| v as f32).collect())
}
