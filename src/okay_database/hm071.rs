//! hm71「种群繁衍」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm71_种群繁衍/hm71_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   逐股系列（check_my_kind == "level2"），**只用盘口快照**，不碰逐笔。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 0) 读盘口：pd.read_csv(全部列)。datetime 列被拿去当索引（本文件用不到它），
//!    deal_data(change_time=1) 把 exchtime 从 epoch 微秒换成 pandas Timestamp，
//!    换法 = to_timedelta(exchtime/1e6 s) + Timestamp('1970-01-01 08:00:00')。
//!    实测（四个验收样本）这个换算和「整数微秒 + 8 小时取模」完全一致（差 0 ns），
//!    所以本文件按 (ex + 8h).rem_euclid(DAY_US) 得到北京钟点微秒。
//! 1) adjust_afternoon：因为索引名是 datetime 而 exchtime 是列，走 elif 分支，
//!      df1 = set_index('exchtime').between_time('09:30:00','11:30:00')
//!      df2 = set_index('exchtime').between_time('13:00:00','14:57:00')，时刻各 -90min
//!      concat([df1, df2]).reset_index()
//!    行序 = 全部上午行（文件序）在前、全部下午行（文件序）在后；两端都是闭区间。
//! 2) df = df[df.last_prc != 0]  —— NaN != 0 在 pandas 里是 True，所以只丢恰好等于 0 的行。
//! 3) df.set_index('exchtime')，然后
//!      df_1m = df.resample('1min').last().ffill()     ← 1 分钟格，逐列取「最后一个非 NaN」
//!      df_1m.volume = df_1m.volume.diff().fillna(0)   ← 分钟成交量增量
//!      df.volume    = df.volume.diff()                ← 快照级成交量增量（首行 NaN）
//!    两个 volume 都从 **同一条过滤后的行序** 上按位置差分，不是按时间差分。
//! 4) res1 = one(df, '1min')，res2 = one(df_1m.copy(), '10min')；
//!    res_price 走 df_1m 的 10 分钟格、只对 last_prc 算 estimate_r。
//!
//! 【estimate_r】= 旧代码里的闭包，本文估算子的唯一核心：
//!     x = 该格内这一列的值（保持行序）→ 非 NaN 的 min/max → 归一化到 [0,1]
//!     → 丢掉 NaN → z = x[:-1]*(1-x[:-1])、y = x[1:]
//!     → r = dot(z,y)/dot(z,z) → clip(0,4)
//!   三个退化情形必须照抄：
//!     · 非 NaN 值个数 < 2（含空格）→ dot 都是 0 → 0/0 = NaN；
//!     · max == min（列在格内是常数）→ 归一化 0/0 全 NaN → 丢完后为空 → NaN；
//!     · z 全 0 → 分母 0，分子也是 0 → NaN。
//! 5) one(df, duration) 里每个格：
//!      agg  = df.resample(duration).agg({'volume':'sum','last_prc':'last'})
//!             空格：volume.sum() 给 0，last_prc.last() 给 NaN
//!      7 个 rs_* 列按格标签对齐贴上（空格没有 apply 结果 → NaN）
//!      .dropna()：volume / last_prc / 任一 rs_* 是 NaN 就把这一格整行丢掉
//!      log_volume = ln(volume)   ← volume 为 0 时是 -inf，旧代码不清理，本文也不清理
//!      统计量顺序（每个量 7 个值，series 顺序见 SERIES7）：
//!         mean std skew kurt median autocorr trend
//!         corr_volume corr_last_prc corr_log_volume
//!         quantile1 quantile9 01 13 34 3357
//!      · autocorr = df[cols].corrwith(df[cols].shift(1))，即 corr(x[i], x[i-1])
//!      · trend    = rp.trend_2d(...,axis=0) = 值与其 1 基位置序号的 Pearson
//!                   （库实现里方差为 0 或样本 <2 时给 0.0，不是 NaN，照抄）
//!      · corr_*   = Series 两两 Pearson，成对完整观测
//!      · quantileN / 区间桶 = (mask * volume).sum() / volume.sum()，NaN 的 volume 两边都跳过
//! 6) names = 112 个 '_1m' + 112 个 '_10m' + 16 个 rs_price_*（顺序见 names()）
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、精度与已知差异
//! ════════════════════════════════════════════════════════════════════
//! · 盘口 CSV 在本文件里按 f64 自解析（不用 fast_csv_reader 的 f32）：
//!   volume / total_*_vol 是**当日累计**量，000001 一天上亿股，f32 只有 24 位尾数，
//!   两个大数相减的相对误差会到 1e-4 量级，正好压在验收阈值上。
//! · estimate_r 的 num / den / 商走 double-double（精确乘积 + 精确求和 + 补偿除法），
//!   原因是旧代码用 np.dot（OpenBLAS 的 ddot），它的亚 ulp 误差会把一批
//!   「数学上恰好等于 1 或 3」的比值挪到桶边界外。实测这一条足以让
//!   rs_bid_vol2_13_1m 在 600519@20250304 上差 1.7e-2。用顺序 f64 求和或
//!   纯有理数都不能同时对上 OpenBLAS 的舍入方向，double-double 才对得上。
//! · Pearson 用「先减均值再求点积」的两遍法，和 numpy 的 corrcoef 更接近；
//!   一遍法（原始和公式）在 log_volume 这种均值远大于方差的列上会掉 1e-14 的绝对精度，
//!   相关系数本身很小的时候会被相对误差判成不一致。
//! · 集合/映射只用于按键分组，BTreeMap 迭代天然升序，不需要额外排序（确定性 OK）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、names_single 有 240 个，没有重名（旧文件里搜过），不需要 _dup 后缀。
//!    names_save 里 6 个带 `_fold` 的名字由新流水线按「与截面均值的距离」生成。

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

/// 盘口 CSV 的列索引（0 基，与文件表头一致；2018 及更早的文件末尾多一列 num）
const COL_EXCHTIME: usize = 4;
const COL_LAST_PRC: usize = 6;
const COL_VOLUME: usize = 14;
const COL_TOTAL_ASK_VOL: usize = 19;
const COL_TOTAL_BID_VOL: usize = 20;
const COL_ASK_VOL1: usize = 22;
const COL_BID_VOL1: usize = 24;
const COL_ASK_VOL2: usize = 26;
const COL_BID_VOL2: usize = 28;
const MAX_COLS: usize = 64;

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400 * SEC_US;
/// deal_data 的那个 +8 小时
const OFFSET_8H: i64 = 8 * 3600 * SEC_US;
/// 下午整体左移 90 分钟（adjust_afternoon）
const SHIFT_AFTERNOON: i64 = 90 * 60 * SEC_US;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

const MIN_US: i64 = 60 * SEC_US;
const TEN_MIN_US: i64 = 600 * SEC_US;

/// 七个被 estimate_r 处理的量，顺序即 names 里每个统计量块内的顺序
const SERIES7: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];
const SERIES_NAMES7: [&str; 7] = [
    "volume",
    "ask_vol1",
    "bid_vol1",
    "ask_vol2",
    "bid_vol2",
    "total_ask_vol",
    "total_bid_vol",
];
/// 每个量算的 16 个统计量，顺序即旧代码 res 列表的顺序
const STATS16: [&str; 16] = [
    "mean",
    "std",
    "skew",
    "kurt",
    "median",
    "autocorr",
    "trend",
    "corr_volume",
    "corr_last_prc",
    "corr_log_volume",
    "quantile1",
    "quantile9",
    "01",
    "13",
    "34",
    "3357",
];

const PER_DUR: usize = 16 * 7; // 112
pub const N_OUT: usize = PER_DUR * 2 + 16; // 240

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for suffix in ["_1m", "_10m"] {
        for stat in STATS16.iter() {
            for s in SERIES_NAMES7.iter() {
                out.push(format!("rs_{}_{}{}", s, stat, suffix));
            }
        }
    }
    for stat in STATS16.iter() {
        out.push(format!("rs_price_{}", stat));
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

/// 一行盘口。t = 平移后的当日微秒；v[0]=当日累计 volume，v[1..6] 是盘口量。
#[derive(Clone, Copy)]
struct Row {
    t: i64,
    last_prc: f64,
    v: [f64; 7],
}

/// 复刻到旧代码第 2 步为止：读出 adjust_afternoon + last_prc != 0 之后的行。
///
/// 行序 = 上午（文件序）在前、下午（文件序）在后，和 pd.concat([df1, df2]) 一致。
fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
    let path = market_path(code, date)?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let volume_idx = [COL_VOLUME, COL_ASK_VOL1, COL_BID_VOL1, COL_ASK_VOL2, COL_BID_VOL2, COL_TOTAL_ASK_VOL, COL_TOTAL_BID_VOL];

    let mut morning: Vec<Row> = Vec::new();
    let mut afternoon: Vec<Row> = Vec::new();
    let mut lines = content.lines();
    lines.next(); // 表头
    for raw in lines {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let mut f: [&[u8]; MAX_COLS] = [&[][..]; MAX_COLS];
        let nf = split_into(line.as_bytes(), &mut f);
        if nf <= COL_BID_VOL2 {
            continue;
        }
        let ex_f = parse_f64(f[COL_EXCHTIME]);
        if !ex_f.is_finite() {
            continue;
        }
        let ex = ex_f.round() as i64;
        let day = (ex + OFFSET_8H).rem_euclid(DAY_US);
        // adjust_afternoon 的两个闭区间窗口
        let (push_to, t) = if (LD_0930..=LD_1130).contains(&day) {
            (&mut morning, day)
        } else if (LD_1300..=LD_1457).contains(&day) {
            (&mut afternoon, day - SHIFT_AFTERNOON)
        } else {
            continue;
        };
        let last_prc = parse_f64(f[COL_LAST_PRC]);
        // df[df.last_prc != 0]：只丢恰好为 0 的行，NaN 保留
        if last_prc == 0.0 {
            continue;
        }
        let mut v = [f64::NAN; 7];
        for (k, &ci) in volume_idx.iter().enumerate() {
            v[k] = parse_f64(f[ci]);
        }
        push_to.push(Row { t, last_prc, v });
    }

    let mut rows = morning;
    rows.extend(afternoon);
    if rows.is_empty() {
        return None;
    }
    Some(rows)
}

/// df.volume = df.volume.diff()：按位置差分，首行 NaN，任一侧 NaN 传染。
fn diff_volume(rows: &mut [Row]) {
    let mut prev = f64::NAN;
    for (i, r) in rows.iter_mut().enumerate() {
        let cur = r.v[0];
        r.v[0] = if i == 0 { f64::NAN } else { cur - prev };
        prev = cur;
    }
}

// ============================================================================
// 分格与 estimate_r
// ============================================================================

/// 把行按「向下取整到 bin_us 的整数倍」分格，返回**从首格到末格连续**的标签与行下标，
/// 中间没有数据的格返回空 Vec（对应 pandas resample 里存在但没数据的格）。
fn build_groups(rows: &[Row], bin_us: i64) -> Vec<(i64, Vec<usize>)> {
    if rows.is_empty() {
        return Vec::new();
    }
    let mut map: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        let label = r.t.div_euclid(bin_us) * bin_us;
        map.entry(label).or_default().push(i);
    }
    let first = *map.keys().next().unwrap();
    let last = *map.keys().next_back().unwrap();
    let mut out: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut label = first;
    while label <= last {
        out.push((label, map.remove(&label).unwrap_or_default()));
        label += bin_us;
    }
    out
}

/// estimate_r 的一维版本（见文件头）。空输入、常数输入、z 全 0 都给 NaN。
///
/// num 与 den 的求和、以及最后那一次除法都走 double-double（见本文件末尾的 DD）。
/// 为什么不能直接用 f64 顺序求和：旧代码用的是 np.dot，也就是 OpenBLAS 的 ddot。
/// 这个桶边界（>0 / <1 / >1 / <3 / <3 / <4 / <3 / <3.57）正好压在一批
/// **数学上恰好等于 1 或 3** 的比值上，此时 ddot 的亚 ulp 误差会把一个格子挪出桶外。
/// 实测 600519@20250304 的 bid_vol2 09:56 那格：数学值恰好 1，OpenBLAS 给 1.0，
/// 而 f64 顺序求和给 1.0000000000000004，于是 rs_bid_vol2_13_1m 从 0.4605 变成 0.4683
/// （相对差 1.7e-2，远超 1e-4 的验收阈值）。
/// 用 double-double 把「精确乘积 + 精确求和 + 正确舍入的商」算出来，就能在
/// 「两个精确和相等」时给出与 OpenBLAS 一致的整数结果。四个验收样本 240×4 个值全过。
fn estimate_r(vals: &[f64]) -> f64 {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    let mut cnt = 0usize;
    for &x in vals {
        if x.is_nan() {
            continue;
        }
        cnt += 1;
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    if cnt < 2 || mx == mn {
        return f64::NAN;
    }
    let span = mx - mn;
    let mut norm: Vec<f64> = Vec::with_capacity(cnt);
    for &x in vals {
        if x.is_nan() {
            continue;
        }
        norm.push((x - mn) / span);
    }
    let n = norm.len();
    let mut num = Dd::ZERO;
    let mut den = Dd::ZERO;
    for i in 0..n - 1 {
        let zi = norm[i] * (1.0 - norm[i]);
        num = num.add_prod(zi, norm[i + 1]);
        den = den.add_prod(zi, zi);
    }
    if den.hi == 0.0 && den.lo == 0.0 {
        return f64::NAN;
    }
    dd_div(num, den).clamp(0.0, 4.0)
}

// ============================================================================
// double-double（QD 库那一套）：只用于 estimate_r 的 num / den / 商
// ============================================================================

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// 精确乘积：hi = fl(a*b)，lo = 精确乘积 - hi（靠 FMA，不加 FMA 也是精确的，只是慢）。
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

#[derive(Clone, Copy)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };

    /// QD 的 dd_add：把两个双精度双字相加，误差量级 2^-106。
    #[inline]
    fn add_dd(self, b: Dd) -> Dd {
        let s = self.hi + b.hi;
        let v = s - self.hi;
        let t = (self.hi - (s - v)) + (b.hi - v);
        let t = t + self.lo + b.lo;
        let s2 = s + t;
        let t2 = (s - s2) + t;
        Dd { hi: s2, lo: t2 }
    }

    #[inline]
    fn add_prod(self, a: f64, b: f64) -> Dd {
        let (p, e) = two_prod(a, b);
        self.add_dd(Dd { hi: p, lo: e })
    }
}

/// 补偿除法：先把商取一次近似，再用残差修正，结果与「精确商四舍五入」一致。
/// 两个精确和完全相等时残差为 0，商就是精确的整数，这一点是关键。
fn dd_div(num: Dd, den: Dd) -> f64 {
    let q = num.hi / den.hi;
    let (ph, pl) = two_prod(q, den.hi);
    let (dh, dl) = two_sum(num.hi, -ph);
    let r_lo = dl + num.lo - pl - q * den.lo;
    q + (dh + r_lo) / den.hi
}

fn value_of(r: &Row, c: usize) -> f64 {
    if c < 7 {
        r.v[c]
    } else {
        r.last_prc
    }
}

/// 一张做完 dropna 的表：每列长度一致。
struct Table {
    cols: Vec<Vec<f64>>,
    volume: Vec<f64>,
    last_prc: Vec<f64>,
    log_volume: Vec<f64>,
    /// corr_volume 的对照序列：
    /// `one()` 里是 `df[cols].corrwith(df.rs_volume)`（对 rs_volume 自己就是 1.0），
    /// 价格那段是 `df_1m[cols].corrwith(df_1m.volume)`（真正的 volume 列），两者不同。
    corr_volume: Vec<f64>,
}

/// 按格算 rs 列 + volume.sum + last_prc.last，再按旧代码的 .dropna() 丢格。
/// `cols` 给出要算 rs 的序列下标（0..6 是盘口量，7 是 last_prc）。
/// `partner_is_rs_volume` 决定 corr_volume 拿谁当对照（见 Table 的注释）。
fn build_table(
    rows: &[Row],
    groups: &[(i64, Vec<usize>)],
    cols: &[usize],
    partner_is_rs_volume: bool,
) -> Table {
    let m = groups.len();
    let mut rs: Vec<Vec<f64>> = cols.iter().map(|_| vec![f64::NAN; m]).collect();
    let mut volume = vec![0.0f64; m];
    let mut last_prc = vec![f64::NAN; m];
    for (bi, (_, idx)) in groups.iter().enumerate() {
        if idx.is_empty() {
            continue;
        }
        for (ci, &c) in cols.iter().enumerate() {
            let vals: Vec<f64> = idx.iter().map(|&i| value_of(&rows[i], c)).collect();
            rs[ci][bi] = estimate_r(&vals);
        }
        // volume.sum()：跳过 NaN；全 NaN 给 0（pandas min_count=0）
        let mut s = 0.0f64;
        let mut seen = false;
        for &i in idx.iter() {
            let x = rows[i].v[0];
            if !x.is_nan() {
                s += x;
                seen = true;
            }
        }
        volume[bi] = if seen { s } else { 0.0 };
        // last_prc.last()：最后一个非 NaN
        for &i in idx.iter() {
            let x = rows[i].last_prc;
            if !x.is_nan() {
                last_prc[bi] = x;
            }
        }
    }

    // .dropna()：volume / last_prc / 任一 rs_* 是 NaN 就丢掉这一格
    let keep: Vec<usize> = (0..m)
        .filter(|&bi| {
            !volume[bi].is_nan()
                && !last_prc[bi].is_nan()
                && rs.iter().all(|col| !col[bi].is_nan())
        })
        .collect();

    let out_cols: Vec<Vec<f64>> = rs
        .iter()
        .map(|col| keep.iter().map(|&bi| col[bi]).collect())
        .collect();
    let vol: Vec<f64> = keep.iter().map(|&bi| volume[bi]).collect();
    let lprc: Vec<f64> = keep.iter().map(|&bi| last_prc[bi]).collect();
    let lg: Vec<f64> = vol.iter().map(|&v| v.ln()).collect();
    let partner: Vec<f64> = if partner_is_rs_volume {
        out_cols[0].clone()
    } else {
        vol.clone()
    };
    Table { cols: out_cols, volume: vol, last_prc: lprc, log_volume: lg, corr_volume: partner }
}

/// df.resample('1min').last().ffill() + volume 差分，得到分钟表。
/// 入参必须是**未差分**的行（v[0] = 当日累计量），和旧代码的执行顺序一致。
fn build_1min(rows: &[Row], groups: &[(i64, Vec<usize>)]) -> Vec<Row> {
    let mut out: Vec<Row> = Vec::with_capacity(groups.len());
    let mut prev: Option<Row> = None;
    for (label, idx) in groups.iter() {
        let mut r = Row { t: *label, last_prc: f64::NAN, v: [f64::NAN; 7] };
        // .last() 是**逐列**取最后一个非 NaN，不是取最后一行的整行
        for &i in idx.iter() {
            let src = &rows[i];
            if !src.last_prc.is_nan() {
                r.last_prc = src.last_prc;
            }
            for c in 0..7 {
                if !src.v[c].is_nan() {
                    r.v[c] = src.v[c];
                }
            }
        }
        // .ffill()
        if let Some(p) = prev {
            if r.last_prc.is_nan() {
                r.last_prc = p.last_prc;
            }
            for c in 0..7 {
                if r.v[c].is_nan() {
                    r.v[c] = p.v[c];
                }
            }
        }
        out.push(r);
        prev = Some(r);
    }
    // .diff().fillna(0)
    let vols: Vec<f64> = out.iter().map(|r| r.v[0]).collect();
    for i in 0..out.len() {
        let d = if i == 0 { f64::NAN } else { vols[i] - vols[i - 1] };
        out[i].v[0] = if d.is_nan() { 0.0 } else { d };
    }
    out
}

// ============================================================================
// pandas 统计量（语义与 hm052.rs 里那批一致，Pearson 改成两遍法）
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

fn nan_mean(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v.iter().filter(|&&x| is_ok(x)) {
        s += x;
        n += 1;
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

fn nan_median(v: &[f64]) -> f64 {
    let mut ok: Vec<f64> = v.iter().copied().filter(|&x| is_ok(x)).collect();
    if ok.is_empty() {
        return f64::NAN;
    }
    ok.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

/// Series.quantile(q)：默认线性插值，位置 = (n-1)*q，跳过 NaN。
fn nan_quantile(v: &[f64], q: f64) -> f64 {
    let mut ok: Vec<f64> = v.iter().copied().filter(|&x| is_ok(x)).collect();
    if ok.is_empty() {
        return f64::NAN;
    }
    ok.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = ok.len();
    if n == 1 {
        return ok[0];
    }
    let pos = (n as f64 - 1.0) * q;
    let lo = pos.floor();
    let hi = pos.ceil();
    if lo == hi {
        return ok[lo as usize];
    }
    let frac = pos - lo;
    ok[lo as usize] + (ok[hi as usize] - ok[lo as usize]) * frac
}

/// Series.corr：成对完整观测的 Pearson（先减均值），n<2 或任一侧方差不为正 → NaN。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..x.len().min(y.len()) {
        if !is_ok(x[i]) || !is_ok(y[i]) {
            continue;
        }
        n += 1;
        sx += x[i];
        sy += y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let mx = sx / n as f64;
    let my = sy / n as f64;
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len().min(y.len()) {
        if !is_ok(x[i]) || !is_ok(y[i]) {
            continue;
        }
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if !(sxx > 0.0) || !(syy > 0.0) {
        return f64::NAN;
    }
    let r = sxy / (sxx.sqrt() * syy.sqrt());
    r.clamp(-1.0, 1.0)
}

/// df[cols].corrwith(df[cols].shift(1))：corr(x[i], x[i-1])
fn autocorr(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    pearson(&x[1..], &x[..x.len() - 1])
}

/// rp.trend_2d(col, axis=0)：值与其 1 基位置序号的 Pearson；
/// 非有限值先丢掉；样本 < 2 或任一侧方差为 0 时给 0.0（不是 NaN）。
fn trend(x: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::new();
    let mut idx: Vec<f64> = Vec::new();
    for (i, &v) in x.iter().enumerate() {
        if v.is_finite() {
            values.push(v);
            idx.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x: f64 = values.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = idx.iter().sum::<f64>() / n as f64;
    let (mut cov, mut var_x, mut var_y) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = idx[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// (mask * volume).sum() / volume.sum()：NaN 的 volume 分子分母都跳过。
fn bucket(col: &[f64], volume: &[f64], pred: impl Fn(f64) -> bool) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..col.len().min(volume.len()) {
        let w = volume[i];
        if w.is_nan() {
            continue;
        }
        den += w;
        if pred(col[i]) {
            num += w;
        }
    }
    num / den
}

/// 一张表的 16×k 个值，顺序见文件头第 5 步。
fn table_metrics(t: &Table) -> Vec<f64> {
    let k = t.cols.len();
    let mut out: Vec<f64> = Vec::with_capacity(16 * k);
    let stats: [fn(&[f64]) -> f64; 5] = [nan_mean, nan_std, nan_skew, nan_kurt, nan_median];
    for f in stats.iter() {
        for c in t.cols.iter() {
            out.push(f(c));
        }
    }
    for c in t.cols.iter() {
        out.push(autocorr(c));
    }
    for c in t.cols.iter() {
        out.push(trend(c));
    }
    for c in t.cols.iter() {
        out.push(pearson(c, &t.corr_volume));
    }
    for c in t.cols.iter() {
        out.push(pearson(c, &t.last_prc));
    }
    for c in t.cols.iter() {
        out.push(pearson(c, &t.log_volume));
    }
    // quantile1: col <= q10；quantile9: col >= q90
    for &(q, lower) in [(0.1f64, true), (0.9, false)].iter() {
        for c in t.cols.iter() {
            let qv = nan_quantile(c, q);
            if lower {
                out.push(bucket(c, &t.volume, |x: f64| x <= qv));
            } else {
                out.push(bucket(c, &t.volume, |x: f64| x >= qv));
            }
        }
    }
    for &(lo, hi) in [(0.0f64, 1.0f64), (1.0, 3.0), (3.0, 4.0), (3.0, 3.57)].iter() {
        for c in t.cols.iter() {
            out.push(bucket(c, &t.volume, |x: f64| x > lo && x < hi));
        }
    }
    out
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 一只股票的一天：算出 240 个因子值。算不出来返回 None（上游补整条 NaN）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // 旧代码的顺序很关键：df_1m 用的是**累计** volume（先 .last() 再 .diff()），
    // 而快照级的 df.volume 差分在 df_1m 建好之后才做。两个 volume 不是同一个东西。
    let raw = read_rows(code, date)?;
    let g1 = build_groups(&raw, MIN_US);
    let rows_1m = build_1min(&raw, &g1);

    let mut rows = raw;
    diff_volume(&mut rows);
    let t1 = build_table(&rows, &g1, &SERIES7, true);

    let g10 = build_groups(&rows_1m, TEN_MIN_US);
    let t2 = build_table(&rows_1m, &g10, &SERIES7, true);
    let tp = build_table(&rows_1m, &g10, &[7], false);

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    out.extend(table_metrics(&t1));
    out.extend(table_metrics(&t2));
    out.extend(table_metrics(&tp));
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm71 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
