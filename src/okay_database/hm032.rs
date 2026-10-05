//! hm32 量价的适中变化 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm32_量价的适中变化/hm32_okay.py
//! （run_base 里的 Go 类；run_factor_out 的实参是 get_many=0 / read_full=0 / change_time=1）
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（改代码前先核这一节）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()      —— 只读盘口，一天一张表
//!    · 该函数里 data.index = pd.to_datetime(data["datetime"])，之后 between_time
//!      用的就是这个索引，也就是「exchtime + 8 小时的北京时间」。
//!      实测（2016/2018/2020/2023/2025 各抽查若干文件）datetime 与 exchtime+8h
//!      逐行完全一致（精确到微秒），所以下面直接用 exchtime+8h 还原索引。
//!    · change_time=1 走 deal_data，但 deal_data 只改 exchtime/localtime 两列，
//!      不动索引，也不排序；所以本文件不调 adjust_afternoon，窗口用真实钟点。
//!
//! 2) 三列先做变换，**在整张表上做，然后才切时间窗**（顺序不能反）：
//!        df.volume    = log(df.volume.diff().replace(0, 1))
//!        df.ask_vol1  = log(df.ask_vol1.replace(0, 1))
//!        df.bid_vol1  = log(df.bid_vol1.replace(0, 1))
//!    · volume 是当日累计成交量，diff() 是相邻行相减（第一行 NaN）。
//!      这里 diff 用的是 pandas Series.diff，走的是**整表行序**，
//!      集合竞价前的那些行也算进去，所以读文件时必须保留表头之后的全部行。
//!    · replace(0, 1) 只替 0.0，NaN 不变；log(负) 在 numpy 里是 NaN。
//!    · 之后 df = df.between_time('09:30:00','14:57:00')，再 df = df[df.last_prc != 0]。
//!      **两个过滤的顺序**：先切时间、再去掉 last_prc==0 的行。
//!
//! 3) 五个时段（都是 between_time 闭区间，微秒精度，两端都含）
//!        allday    [09:30:00, 14:57:00]（就是上面过滤后的 df 本身）
//!        open30    [09:30:00, 10:00:00]
//!        morning   [09:30:00, 11:30:00]
//!        afternoon [13:00:00, 14:57:00]
//!        close30   [14:30:00, 14:57:00]
//!
//! 4) 每个时段的指标（旧代码的 one()，行数 >0 才算，否则 22 个 NaN）
//!        z(s) = (s - s.mean()) / s.std()          ← pandas，std 是 ddof=1
//!        D_p[i][j] = |z_p[i] - z_p[j]|            ← 价格（last_prc）
//!        D_v / D_a / D_b 同理（volume、ask_vol1、bid_vol1）
//!        M = D_p，把**下三角（含对角线）**换成 D_v 的对应元素：M[i][j] = D_v[i][j] (i>=j)
//!        (max_eigenvalue, evec) = rp.compute_max_eigenvalue(M)   ← 幂迭代，见下
//!        evec 的 11 个统计量：mean, std(ddof=0), skew, kurtosis, max, min,
//!        median, q1, q3, iqr=q3-q1
//!        同一个流程再跑一遍 (ask_vol1, bid_vol1) 得另 11 个。
//!     每个时段输出 22 个（前 11 个 price_volume、后 11 个 ask_bid）。
//!
//! 5) 输出拼接顺序 = names_single 顺序：
//!        时段 allday → open30 → morning → afternoon → close30，
//!        每个时段内 先 price_volume 11 个、再 ask_bid 11 个。共 110 个。
//!        名字形如 vec_iqr_price_volume_close30_unfold，
//!        即 {指标}_{组}_{时段}_unfold。
//!    注意 _fold 版（|log(unfold)|）是旧 run_base 里用 save_factor 另存的，
//!    不属于本文件要产出的 110 列，由因子库那一层处理。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、为什么自己按 f64 解析 CSV
//! ════════════════════════════════════════════════════════════════════
//! fast_csv_reader 的 MarketRecord 里 volume/last_prc/ask_vol1/bid_vol1 都是 f32。
//! 这里 volume 是**当日累计成交量**（可到 1e8 股），f32 的 24 位尾数在这个量级上
//! 间隔已经到 8，diff 出来的每 3 秒增量会被打偏 ±8 股，再做 log 会放大成很大的
//! 相对误差。所以本文件照 hm013 的做法自己按 f64 解析盘口 CSV（列索引与
//! fast_csv_reader 一致）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、幂迭代与统计量的逐条对齐
//! ════════════════════════════════════════════════════════════════════
//! · rp.compute_max_eigenvalue（src/sequence/mod.rs）是固定 30 次、tolerance=1e-4 的
//!   幂迭代，初值 ones(n)/sqrt(n)，norm<1e-5 提前退出。本文件逐行复刻。
//!   矩阵乘向量用朴素循环，实测与 numpy/ndarray 的 gemv 结果差 1e-15 量级，
//!   远小于验收阈值 1e-4（已用 000001@20250303 的五个时段逐个比对过）。
//! · pandas 的 mean/std：mean = sum/count（numpy 成对求和），
//!   std = sqrt( sum((mean-x)^2) / (count-1) )，都跳过 NaN。
//!   本文件的 pairwise_sum 就是 numpy 的成对求和（blocksize 128、8 路展开），
//!   在真实列上验证过与 np.sum 逐位相同。
//! · scipy 1.15.2 的 skew/kurtosis（bias=True、fisher=True）：
//!       m2 = mean(dz^2)，m3 = mean(dz^2*dz)，m4 = mean(dz^2*dz^2)，dz = x - mean
//!       zero = m2 <= (eps * mean)^2      ← eps 是 float64 的 2.22e-16
//!       skew = zero ? NaN : m3 / m2^1.5
//!       kurt = (zero ? NaN : m4 / m2^2) - 3
//!   注意 zero 这一条不只看 m2==0，还跟 **mean** 有关（n=1 这类退化样本会落进 zero，
//!   旧代码给的是 NaN 而不是 0，必须照抄）。
//! · np.percentile(..., 25/50/75) 的 'linear' 法：
//!       virtual = (n-1)*q，lo = floor(virtual)，hi = lo+1（virtual >= n-1 时取末元素）
//!       t = virtual - lo；t < 0.5 → a + (b-a)*t；t >= 0.5 → b - (b-a)*(1-t)
//!   （numpy 的 _lerp 就是这样两个分支，t=0 时退化成正好取 a）
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、因子名与保存名单
//! ════════════════════════════════════════════════════════════════════
//! names_single 共 110 个，没有重名（指标 × 组 × 时段两两不同），不需要 _dup2 后缀。
//! names_save（45 个）在旧文件的 run_base 里，因子库文件里原样搬运。

use std::fs::File;
use std::io::Read;

/// 盘口 CSV 的列索引（0 基，与 fast_csv_reader 的常量一致）
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_COL_VOLUME: usize = 14;
const MKT_COL_ASK_VOL1: usize = 22;
const MKT_COL_BID_VOL1: usize = 24;
const MKT_NCOL: usize = 61;

const US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;

/// 五个时段（真实钟点的微秒闭区间，旧代码用 between_time，两端都含）
const SESSIONS: [(&str, i64, i64); 5] = [
    ("allday", (9 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
    ("open30", (9 * 3600 + 30 * 60) * US, 10 * 3600 * US),
    ("morning", (9 * 3600 + 30 * 60) * US, (11 * 3600 + 30 * 60) * US),
    ("afternoon", 13 * 3600 * US, (14 * 3600 + 57 * 60) * US),
    ("close30", (14 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
];

/// 每个时段 22 个指标（两组各 11 个），五个时段共 110 个
const PER_GROUP: usize = 11;
const PER_SESSION: usize = PER_GROUP * 2;
pub const N_OUT: usize = PER_SESSION * 5;

const METRIC_NAMES: [&str; PER_GROUP] = [
    "max_eigenvalue",
    "vec_mean",
    "vec_std",
    "vec_skew",
    "vec_kurt",
    "vec_max",
    "vec_min",
    "vec_median",
    "vec_q1",
    "vec_q3",
    "vec_iqr",
];

const GROUP_NAMES: [&str; 2] = ["price_volume", "ask_bid"];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (sess, _, _) in SESSIONS.iter() {
        for group in GROUP_NAMES.iter() {
            for m in METRIC_NAMES.iter() {
                out.push(format!("{}_{}_{}_unfold", m, group, sess));
            }
        }
    }
    out
}

#[inline]
fn in_span(t: i64, lo: i64, hi: i64) -> bool {
    let d = t.rem_euclid(DAY_US);
    d >= lo && d <= hi
}

// ============================================================================
// 盘口 CSV 的 f64 读取
// ============================================================================

/// 一天的盘口快照（只留计算要用的列，全部 f64）
#[derive(Clone, Copy, Default)]
struct MktRow {
    time_us: i64,
    last_prc: f64,
    volume: f64,
    ask_vol1: f64,
    bid_vol1: f64,
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

/// 读 market_data CSV（f64）。行集与 pandas 读整张表一致：
/// 表头之后每一行都算（exchtime==0 的行也保留，它参与 volume.diff()）。
fn read_market_f64(code: &str, date: i64) -> Option<Vec<MktRow>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root
            .join(date.to_string())
            .join("market_data")
            .join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<MktRow> = Vec::with_capacity(content.len() / 200 + 1);
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

        out.push(MktRow {
            time_us: parse_i64_at(f[MKT_COL_EXCHTIME]) + 8 * 3600 * US,
            last_prc: parse_f64_at(f[MKT_COL_LAST_PRC]),
            volume: parse_f64_at(f[MKT_COL_VOLUME]),
            ask_vol1: parse_f64_at(f[MKT_COL_ASK_VOL1]),
            bid_vol1: parse_f64_at(f[MKT_COL_BID_VOL1]),
        });
    }
    Some(out)
}

// ============================================================================
// numpy 成对求和（pairwise_sum_FLOAT，blocksize 128、8 路展开）
// ============================================================================

fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a.iter() {
            res += x;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
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
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// pandas Series.mean()：跳过 NaN（inf 不跳），成对求和再除以个数
fn nanmean(a: &[f64]) -> f64 {
    let mut buf: Vec<f64> = Vec::with_capacity(a.len());
    for &x in a.iter() {
        if !x.is_nan() {
            buf.push(x);
        }
    }
    if buf.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(&buf) / (buf.len() as f64)
}

/// pandas Series.std()：跳过 NaN，ddof=1
fn nanstd(a: &[f64]) -> f64 {
    let mut buf: Vec<f64> = Vec::with_capacity(a.len());
    for &x in a.iter() {
        if !x.is_nan() {
            buf.push(x);
        }
    }
    let cnt = buf.len();
    if cnt == 0 {
        return f64::NAN;
    }
    let avg = pairwise_sum(&buf) / (cnt as f64);
    let mut sq: Vec<f64> = Vec::with_capacity(cnt);
    for &x in buf.iter() {
        let d = avg - x;
        sq.push(d * d);
    }
    let d = (cnt as f64) - 1.0;
    (pairwise_sum(&sq) / d).sqrt()
}

// ============================================================================
// 幂迭代（逐行复刻 rp.compute_max_eigenvalue）
// ============================================================================

fn matvec(m: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let row = &m[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for j in 0..n {
            s += row[j] * x[j];
        }
        y[i] = s;
    }
}

#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    s
}

fn max_eigenvalue(m: &[f64], n: usize) -> (f64, Vec<f64>) {
    let mut v = vec![1.0f64 / (n as f64).sqrt(); n];
    let mut new_v = vec![0.0f64; n];
    let mut temp = vec![0.0f64; n];

    matvec(m, n, &v, &mut temp);
    let mut eigenvalue = dot(&v, &temp);

    let max_iter = 30;
    let tolerance = 1e-4;
    for _ in 0..max_iter {
        let prev_eigenvalue = eigenvalue;
        matvec(m, n, &v, &mut new_v);
        let norm = dot(&new_v, &new_v).sqrt();
        // NaN < 1e-5 为 false，与 Rust/ndarray 的写法一致，不会误退出
        if norm < 1e-5 {
            break;
        }
        for x in new_v.iter_mut() {
            *x /= norm;
        }
        matvec(m, n, &new_v, &mut temp);
        eigenvalue = dot(&new_v, &temp);
        std::mem::swap(&mut v, &mut new_v);
        let rel_error = (eigenvalue - prev_eigenvalue).abs();
        if rel_error < tolerance * eigenvalue.abs() {
            break;
        }
    }
    (eigenvalue, v)
}

// ============================================================================
// 特征向量的 11 个统计量
// ============================================================================

/// np.percentile(..., q) 的 'linear' 法。arr 必须已按升序排好（NaN 放最后）。
fn percentile_sorted(arr: &[f64], q: f64) -> f64 {
    let n = arr.len();
    if n == 0 {
        return f64::NAN;
    }
    let virtual_index = (n as f64 - 1.0) * q;
    if virtual_index >= (n as f64 - 1.0) {
        return arr[n - 1];
    }
    if virtual_index < 0.0 {
        return arr[0];
    }
    let lo = virtual_index.floor();
    let hi = lo + 1.0;
    let t = virtual_index - lo;
    let a = arr[lo as usize];
    let b = arr[hi as usize];
    if t >= 0.5 {
        // numpy 的 _lerp：t >= 0.5 走 b - (b-a)*(1-t)
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// 11 个统计量，顺序与 METRIC_NAMES 一致
fn vec_stats(eigenvalue: f64, v: &[f64]) -> [f64; PER_GROUP] {
    let n = v.len() as f64;
    let mean = pairwise_sum(v) / n;

    let mut dz: Vec<f64> = Vec::with_capacity(v.len());
    for &x in v.iter() {
        dz.push(x - mean);
    }
    let mut t2: Vec<f64> = Vec::with_capacity(dz.len());
    let mut t3: Vec<f64> = Vec::with_capacity(dz.len());
    let mut t4: Vec<f64> = Vec::with_capacity(dz.len());
    for &d in dz.iter() {
        let d2 = d * d;
        t2.push(d2);
        t3.push(d2 * d);
        t4.push(d2 * d2);
    }
    let m2 = pairwise_sum(&t2) / n;
    let m3 = pairwise_sum(&t3) / n;
    let m4 = pairwise_sum(&t4) / n;

    // scipy 1.15.2：zero = m2 <= (eps * mean)^2 时 skew/kurtosis 给 NaN
    let eps = f64::EPSILON;
    let thr = eps * mean;
    let zero = m2 <= thr * thr;

    let std = m2.sqrt();
    let skew = if zero { f64::NAN } else { m3 / m2.powf(1.5) };
    let kurt = {
        let base = if zero { f64::NAN } else { m4 / (m2 * m2) };
        base - 3.0
    };

    let mut vmax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    for &x in v.iter() {
        if x > vmax {
            vmax = x;
        }
        if x < vmin {
            vmin = x;
        }
    }

    // 升序，NaN 排到最后（numpy 的 partition/sort 就是这个约定）
    let mut sorted: Vec<f64> = v.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q1 = percentile_sorted(&sorted, 0.25);
    let q3 = percentile_sorted(&sorted, 0.75);

    [
        eigenvalue,
        mean,
        std,
        skew,
        kurt,
        vmax,
        vmin,
        percentile_sorted(&sorted, 0.5),
        q1,
        q3,
        q3 - q1,
    ]
}

// ============================================================================
// 一个时段的 22 个指标
// ============================================================================

/// z = (x - mean)/std，逐元素（NaN 保持 NaN）
fn standardize(x: &[f64]) -> Vec<f64> {
    let mean = nanmean(x);
    let std = nanstd(x);
    let mut out: Vec<f64> = Vec::with_capacity(x.len());
    for &v in x.iter() {
        out.push((v - mean) / std);
    }
    out
}

/// 旧代码的 haha(price, volume)：把 M 的下三角（含对角线）换成 volume 的差值矩阵，
/// 再对 M 做幂迭代。这里一次把 M 构造出来，逐元素与旧代码相同，
/// 只是省掉了两个 n×n 的中间矩阵。
fn haha(zp: &[f64], zv: &[f64]) -> [f64; PER_GROUP] {
    let n = zp.len();
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        let pi = zp[i];
        let vi = zv[i];
        let row = &mut m[i * n..(i + 1) * n];
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i >= j {
                (vi - zv[j]).abs()
            } else {
                (pi - zp[j]).abs()
            };
        }
    }
    let (eigenvalue, v) = max_eigenvalue(&m, n);
    vec_stats(eigenvalue, &v)
}

/// 一个时段的 22 个值：前 11 个 price_volume、后 11 个 ask_bid。
/// 行数为 0 时按旧代码给 22 个 NaN。
fn one_session(price: &[f64], vol: &[f64], ask: &[f64], bid: &[f64]) -> [f64; PER_SESSION] {
    let mut out = [f64::NAN; PER_SESSION];
    if price.is_empty() {
        return out;
    }
    let zp = standardize(price);
    let zv = standardize(vol);
    let za = standardize(ask);
    let zb = standardize(bid);
    out[..PER_GROUP].copy_from_slice(&haha(&zp, &zv));
    out[PER_GROUP..].copy_from_slice(&haha(&za, &zb));
    out
}

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_market_f64(code, date)?;
    if rows.is_empty() {
        return None;
    }
    let n = rows.len();

    // 整表先做三列变换，再切窗口（顺序不能反，见文件头第一步）
    let mut volume = vec![0.0f64; n];
    for i in 0..n {
        let d = if i == 0 {
            f64::NAN
        } else {
            rows[i].volume - rows[i - 1].volume
        };
        // replace(0, 1)：只替 0.0，NaN 不动
        let d = if d == 0.0 { 1.0 } else { d };
        volume[i] = d.ln();
    }
    let mut ask = vec![0.0f64; n];
    let mut bid = vec![0.0f64; n];
    for i in 0..n {
        let a = rows[i].ask_vol1;
        ask[i] = (if a == 0.0 { 1.0 } else { a }).ln();
        let b = rows[i].bid_vol1;
        bid[i] = (if b == 0.0 { 1.0 } else { b }).ln();
    }
    let price: Vec<f64> = rows.iter().map(|r| r.last_prc).collect();

    // 先 between_time('09:30:00','14:57:00')，再去掉 last_prc == 0
    let allday: Vec<usize> = (0..n)
        .filter(|&i| in_span(rows[i].time_us, SESSIONS[0].1, SESSIONS[0].2))
        .filter(|&i| rows[i].last_prc != 0.0)
        .collect();

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for (si, (_sess, lo, hi)) in SESSIONS.iter().enumerate() {
        let idx: Vec<usize> = allday
            .iter()
            .copied()
            .filter(|&i| si == 0 || in_span(rows[i].time_us, *lo, *hi))
            .collect();
        let p: Vec<f64> = idx.iter().map(|&i| price[i]).collect();
        let v: Vec<f64> = idx.iter().map(|&i| volume[i]).collect();
        let a: Vec<f64> = idx.iter().map(|&i| ask[i]).collect();
        let b: Vec<f64> = idx.iter().map(|&i| bid[i]).collect();
        out.extend(one_session(&p, &v, &a, &b).iter().map(|&x| x as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm32 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
