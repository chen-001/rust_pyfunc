//! hm34「动态时间规整」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm34_动态时间规整/hm34_okay.py
//!   框架参数 get_many=0 / read_full=0 / change_time=1，一整天逐笔一张表。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0) → 丢掉 flag==32 的撤单行
//! 2) set_index('exchtime') 之后 resample('1min').apply(one)：
//!    按自然分钟分组（左闭右开），**空分钟也会调用 one**（实测 336 个分钟格里 97 个是空的）。
//!    one(x)：x.volume /= x.volume.sum()（分母是这一分钟里全部行的 volume 之和），
//!            再按 flag 拆成 66 一路、83 一路，各自取 volume 列表（保持行序），
//!            返回 rp.dtw_distance(买列表, 卖列表)。
//!      · 两边都空 → warp 矩阵只有 [0][0]=0，返回 0.0
//!      · 只有一边空 → 返回 +inf
//!      · DTW 是标准动态规划：代价 |a_i - b_j|，累积上/左/左上三者取最小，radius=None 不做窗口约束
//! 3) 结果 Series 依次做 replace(0, NaN)、replace(+inf, NaN)、dropna()，
//!    所以只有「买卖两边都非空、且 DTW 不为 0」的分钟留下来（-inf 不替换）。
//!    注意 allday 用的是整条序列（实测时间范围 09:25 ~ 15:00，含集合竞价和收盘那几分钟）。
//! 4) 四个窗口在分钟标签上做 between_time（两端都含）：
//!      open30 09:30~10:00、morning 09:30~11:30、afternoon 13:00~14:57、close30 14:30~14:57
//! 5) sing1：序列行数 > 0 时取 lag=1..30 的 |autocorr(lag)|，否则 30 个 NaN；
//!    sing2：序列行数 > 20 时取 lag=1..15 的 |autocorr(lag)|，否则 15 个 NaN。
//!    autocorr 是 pandas 的 Series.autocorr(lag) = corr(x, x.shift(lag))，成对完整观测。
//! 6) res = sing1(整天) + sing2(open30) + sing1(morning) + sing1(afternoon) + sing2(close30)
//!    = 30 + 15 + 30 + 30 + 15 = 120，顺序与 names 完全一致。

use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基；2018 年及以前末尾多一列 num，不影响前 15 列）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const MIN_US: i64 = 60 * SEC_US;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

const N_LONG: usize = 30; // sing1 的 lag 个数
const N_SHORT: usize = 15; // sing2 的 lag 个数
pub const N_OUT: usize = N_LONG + N_SHORT + N_LONG + N_LONG + N_SHORT;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for k in 1..=N_LONG {
        out.push(format!("autocorr{}_allday", k));
    }
    for k in 1..=N_SHORT {
        out.push(format!("autocorr{}_open30", k));
    }
    for k in 1..=N_LONG {
        out.push(format!("autocorr{}_morning", k));
    }
    for k in 1..=N_LONG {
        out.push(format!("autocorr{}_afternoon", k));
    }
    for k in 1..=N_SHORT {
        out.push(format!("autocorr{}_close30", k));
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
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

#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    volume: f64,
    flag: i64,
}

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
        out.push(TradeRow {
            time_us: parse_i64_at(f[TRD_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
        });
    }
    Some(out)
}

/// rp.dtw_distance(a, b)：标准 DTW 动态规划，代价 |a_i-b_j|，累积上/左/左上取最小。
/// 两边都空返回 0（warp[0][0]=0），只有一边空返回 +inf。
fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return if n == 0 && m == 0 { 0.0 } else { f64::INFINITY };
    }
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut cur = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = f64::INFINITY;
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let best = prev[j].min(cur[j - 1]).min(prev[j - 1]);
            cur[j] = cost + best;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Series.autocorr(lag) 的绝对值：corr(x[i], x[i-lag])，成对完整观测（Welford，同 hm013）。
fn abs_autocorr(v: &[f64], lag: usize) -> f64 {
    if v.len() <= lag {
        return f64::NAN;
    }
    let x = &v[lag..];
    let y = &v[..v.len() - lag];
    let mut nobs = 0.0f64;
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    let (mut sx, mut sy, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        nobs += 1.0;
        let dx = x[i] - mx;
        let dy = y[i] - my;
        mx += dx / nobs;
        my += dy / nobs;
        sx += (x[i] - mx) * dx;
        sy += (y[i] - my) * dy;
        cxy += (x[i] - mx) * dy;
    }
    if nobs < 1.0 {
        return f64::NAN;
    }
    let divisor = (sx * sy).sqrt();
    let r = if divisor != 0.0 { cxy / divisor } else { f64::NAN };
    r.abs()
}

/// sing1：序列行数 > 0 取 lag 1..30，否则 30 个 NaN
fn sing1(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return vec![f64::NAN; N_LONG];
    }
    (1..=N_LONG).map(|k| abs_autocorr(v, k)).collect()
}

/// sing2：序列行数 > 20 取 lag 1..15，否则 15 个 NaN
fn sing2(v: &[f64]) -> Vec<f64> {
    if v.len() <= 20 {
        return vec![f64::NAN; N_SHORT];
    }
    (1..=N_SHORT).map(|k| abs_autocorr(v, k)).collect()
}

#[inline]
fn in_span(t: i64, lo: i64, hi: i64) -> bool {
    let d = t.rem_euclid(DAY_US);
    d >= lo && d <= hi
}

fn window<'a>(v: &'a [(i64, f64)], lo: i64, hi: i64) -> Vec<f64> {
    v.iter()
        .filter(|(t, _)| in_span(*t, lo, hi))
        .map(|(_, x)| *x)
        .collect()
}

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades_all = read_trade_f64(code, date)?;
    let trades: Vec<TradeRow> = trades_all.into_iter().filter(|r| r.flag != 32).collect();

    // 按自然分钟分桶（空分钟也要留，值算出来是 0 → NaN）
    let mut series: Vec<(i64, f64)> = Vec::new();
    if !trades.is_empty() {
        let n = trades.len();
        let first = trades[0].time_us.div_euclid(MIN_US);
        let last = trades[n - 1].time_us.div_euclid(MIN_US);
        let mut i = 0usize;
        let mut b = first;
        while b <= last {
            let mut v66: Vec<f64> = Vec::new();
            let mut v83: Vec<f64> = Vec::new();
            let mut total = 0.0f64;
            let start = i;
            while i < n && trades[i].time_us.div_euclid(MIN_US) == b {
                total += trades[i].volume;
                i += 1;
            }
            for r in trades[start..i].iter() {
                let nv = r.volume / total;
                if r.flag == 66 {
                    v66.push(nv);
                } else if r.flag == 83 {
                    v83.push(nv);
                }
            }
            let d = dtw_distance(&v66, &v83);
            // replace(0, NaN) + replace(+inf, NaN)；-inf 与 NaN 保持原样留给 dropna
            let val = if d == 0.0 || d == f64::INFINITY {
                f64::NAN
            } else {
                d
            };
            if !val.is_nan() {
                // 窗口过滤用的是**分钟标签**（bin 左端），不是这一分钟第一笔的实际时间：
                // 10:00:00 这一格的第一笔可能落在 10:00:01，按实际时间筛会把整格漏掉。
                series.push((b * MIN_US, val));
            }
            b += 1;
        }
    }

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for v in sing1(&window(&series, 0, DAY_US - 1)) {
        out.push(v as f32);
    }
    for v in sing2(&window(&series, LD_0930, LD_1000)) {
        out.push(v as f32);
    }
    for v in sing1(&window(&series, LD_0930, LD_1130)) {
        out.push(v as f32);
    }
    for v in sing1(&window(&series, LD_1300, LD_1457)) {
        out.push(v as f32);
    }
    for v in sing2(&window(&series, LD_1430, LD_1457)) {
        out.push(v as f32);
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm34 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
