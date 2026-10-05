//! hm9「宿舍中转站」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm9_宿舍中转站/hm9_okay.py
//!   框架参数 get_many=0 / read_full=0 / change_time=1，一整天逐笔一张表，窗口自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 这个系列分两段，必须分开看
//! ════════════════════════════════════════════════════════════════════
//! 第一段：Go.get_factor —— 纯逐股，一次算 54 个量 = one(df,None) + one(df,66) + one(df,83)，
//!   每一路 18 个。这一段由 [`compute_stock`] 复现，也是 verify_series.py 的对账口径
//!   （旧 get_factor 返回的是一串数字，所以 verify 走逐股那条路径）。
//!
//! 第二段：run_base 里 run_factor_out 之后的那一坨 —— 把 54 个量的日频面板做
//!   横截面 rank + `dwo.de_cross` 正交化，最终只有 25 个名字写进因子库的 base 层
//!   （`dormitory_transfer_base_son` 存 54 个原值，`dormitory_transfer_base` 存这 25 个正交化后的）。
//!   这一段由 [`compute_cross`] 复现，因子库文件用 kind='cross_section' 调它。
//!   所以 names() 给的是这 25 个名字，而 compute_stock 给的是 54 个原值 ——
//!   两者长度不同是**故意的**：okay_compute_stock 是调试/对账入口，不做补齐；
//!   真正进因子库的是 compute_cross + names()。
//!
//! ════════════════════════════════════════════════════════════════════
//! 第一段逐步对照
//! ════════════════════════════════════════════════════════════════════
//! df = read_trade_data_single(with_retreat=0) → 丢掉 flag==32；
//!      set_index('exchtime').between_time('09:30:00','14:57:00')（**没有** sort_index，
//!      而原文件本来就按时间有序，所以等价于保持原顺序）。
//! one(df0, flag)：
//!  1) dorm = df.groupby('price').price.count()；dorm = dorm.nlargest(1)
//!     → 成交笔数最多的那个价位；并列时取 index 顺序里的第一个，
//!       而 groupby 的 index 是按 price 升序的，所以并列取**价格最小**的那个
//!       （实测 pandas nlargest 并列取原顺序第一个）。
//!  2) dorms = df[df.price==dorm_price]，dorms_duras = dorms.exchtime.diff().total_seconds()
//!     （只在前一条 dorms 行上做差，第一条 NaN），autocorr(1/2/3)。
//!  3) xixi：把 (price==dorm_price) 标成 1、其余标成 NaN，再把每一段连续 NaN 编号 2、3、4…
//!     然后 df.price 改成这个编号，df = df[df.price!=1]（即只留「不在宿舍价位」的成交）。
//!     编号在 flag 过滤**之前**算好，所以按 flag 过滤只是从各段里抽走一部分行。
//!  4) 若给了 flag 就先按 flag 过滤；然后 df.turnover /= df.turnover.sum()
//!     （分母是过滤之后的这一份，count 的分母 df.shape[0] 也是过滤之后的）。
//!  5) dorms_itv1 = 按段求 turnover 之和；dorms_itv2 = 按段计数 / df.shape[0]。
//!     ！！这两个的 autocorr 是在**没有**按 t 过滤之前算的。
//!  6) tnext/tlast = exchtime 的前一行/后一行（在过滤后的表里按行序），
//!     t = 段内 tnext 的最大值 - 段内 tlast 的最小值（各自跳过 NaN），total_seconds，
//!     把 0 换成 NaN 再 dropna。
//!  7) itv1/itv2 按 t 的 index 过滤之后，s1 = itv1/t，s2 = itv2/t，s4 = itv1/itv2/t。
//!  8) 输出 18 个：duras(1..3)、itv1(1..3)、itv2(1..3)、s1(1..3)、s2(1..3)、s4(1..3)。
//!     相关系数一律是 pandas 的 Series.autocorr(lag) = corr(x, x.shift(lag))，
//!     成对完整观测（Welford，和 hm013 里那份一样）。
//!  df 为空时 one 返回 18 个 NaN。
//!
//! ════════════════════════════════════════════════════════════════════
//! 第二段逐步对照（都只在 names_save 里点名的那几组上做）
//! ════════════════════════════════════════════════════════════════════
//! · s4 组（all / buy / sell 各自的 corr1..3）：先把三条序列各自横截面 rank(axis=1)，
//!   再 de_cross(rank_k, [另外两个 rank])。
//! · itv 组：e_k = |rank(itv1_corrk) - rank(itv2_corrk)|（k=3 时旧代码写的是
//!   `dac1s['..._dorms_itv3_corr3']`，那只是名字里把 itv2 打成了 itv3，取到的还是
//!   itv2_corr3 那一列，所以 e_3 就是 |rank(itv1_corr3) - rank(itv2_corr3)|），
//!   再 de_cross(e_k, [另外两个 e])。all 那组不在 names_save 里，不算。
//! · s1s2 组：f_k = |rank(s1_corrk) - rank(s2_corrk)|，再 de_cross(f_k, [另外两个 f])。
//! · duras 组（只有 all）：de_cross(rank(duras_k), [另外两个 rank])。
//! · de_cross：merge_many(how='inner') 之后按日跑 OLS fac1 ~ fac2 + fac3（含截距），
//!   取残差。inner 的意思是三条序列里只要有一条是 NaN，这只股票当天就被丢掉（输出 NaN）。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use rayon::prelude::*;

use crate::data_paths;

/// 逐笔 CSV 的列索引（0 基；2018 年及以前末尾多一列 num，不影响前 15 列）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 每一路（all / buy / sell）18 个量
const N_PER: usize = 18;
/// 逐股原值一共 54 个
const N_RAW: usize = N_PER * 3;
/// 因子库要的 25 个（names_save）
pub const N_OUT: usize = 25;

/// 18 个量在一路里的偏移
const O_DURAS: usize = 0;
const O_ITV1: usize = 3;
const O_ITV2: usize = 6;
const O_S1: usize = 9;
const O_S2: usize = 12;
const O_S4: usize = 15;

/// names_save 的顺序（旧文件 run_base 末尾那份名单，原样搬过来）
const OUT_NAMES: [&str; N_OUT] = [
    "buy_s4_corr2",
    "all_dorms_duras_corr3",
    "buy_s4_corr3",
    "sell_dorms_itv1_corr3",
    "all_s1_corr1",
    "buy_dorms_itv1_corr3",
    "sell_s4_corr1",
    "sell_s4_corr2",
    "buy_dorms_itv1_corr1",
    "sell_s4_corr3",
    "buy_s1_corr3",
    "sell_s1_corr3",
    "buy_s4_corr1",
    "all_s1_corr3",
    "sell_s1_corr1",
    "sell_s1_corr2",
    "buy_s1_corr2",
    "all_s1_corr2",
    "all_dorms_duras_corr2",
    "all_s4_corr2",
    "buy_dorms_itv1_corr2",
    "buy_s1_corr1",
    "all_s4_corr3",
    "sell_dorms_itv1_corr2",
    "all_dorms_duras_corr1",
];

/// 逐股原值的名字顺序（旧 names，只是给调试看的，不参与进制）
#[allow(dead_code)]
const RAW_BASE: [&str; N_PER] = [
    "dorms_duras_corr1",
    "dorms_duras_corr2",
    "dorms_duras_corr3",
    "dorms_itv1_corr1",
    "dorms_itv1_corr2",
    "dorms_itv1_corr3",
    "dorms_itv2_corr1",
    "dorms_itv2_corr2",
    "dorms_itv3_corr3",
    "s1_corr1",
    "s1_corr2",
    "s1_corr3",
    "s2_corr1",
    "s2_corr2",
    "s2_corr3",
    "s4_corr1",
    "s4_corr2",
    "s4_corr3",
];

pub fn names() -> Vec<String> {
    OUT_NAMES.iter().map(|s| s.to_string()).collect()
}

// ============================================================================
// 读取
// ============================================================================

#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    price: f64,
    turnover: f64,
    flag: i64,
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

fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in data_paths::level2_roots() {
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
            price: parse_f64_at(f[TRD_COL_PRICE]),
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
        });
    }
    Some(out)
}

// ============================================================================
// pandas 语义的小工具
// ============================================================================

/// DataFrame.corr() / Series.autocorr() 的 Pearson：逐行复刻 pandas 的 nancorr（Welford）。
/// 与 hm013 里那份一致：成对完整观测，有效样本 < 1 或某侧方差为 0 返回 NaN。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs = 0.0f64;
    let (mut meanx, mut meany) = (0.0f64, 0.0f64);
    let (mut ssqdmx, mut ssqdmy, mut covxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
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
    if nobs < 1.0 {
        return f64::NAN;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// Series.autocorr(lag)：x[i] 与 x[i-lag] 的 Pearson。
fn autocorr(v: &[f64], lag: usize) -> f64 {
    if v.len() <= lag {
        return f64::NAN;
    }
    pearson(&v[lag..], &v[..v.len() - lag])
}

/// pandas rank(method='average')：升序，并列取平均名次，NaN 保持 NaN。
fn rank_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| values[i].is_finite()).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            out[idx[k]] = avg;
        }
        i = j;
    }
    out
}

/// statsmodels 的 ols('y ~ x1 + x2').fit().resid：含截距，取残差。
/// 先各自减均值再解 2×2 正规方程，等价于含截距的最小二乘，数值上也稳。
fn ols_resid(y: &[f64], x1: &[f64], x2: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let my = y.iter().sum::<f64>() / nf;
    let m1 = x1.iter().sum::<f64>() / nf;
    let m2 = x2.iter().sum::<f64>() / nf;
    let (mut s11, mut s22, mut s12, mut s1y, mut s2y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let a = x1[i] - m1;
        let b = x2[i] - m2;
        let c = y[i] - my;
        s11 += a * a;
        s22 += b * b;
        s12 += a * b;
        s1y += a * c;
        s2y += b * c;
    }
    let det = s11 * s22 - s12 * s12;
    let (b1, b2) = if det != 0.0 {
        ((s1y * s22 - s2y * s12) / det, (s2y * s11 - s1y * s12) / det)
    } else {
        (f64::NAN, f64::NAN)
    };
    (0..n)
        .map(|i| (y[i] - my) - b1 * (x1[i] - m1) - b2 * (x2[i] - m2))
        .collect()
}

// ============================================================================
// 第一段：一路（all / buy / sell）的 18 个量
// ============================================================================

fn one_mode(rows: &[TradeRow], flag: Option<i64>) -> [f64; N_PER] {
    let mut out = [f64::NAN; N_PER];
    if rows.is_empty() {
        return out;
    }

    // 1) 成交笔数最多的价位；并列取价格最小的那个
    let mut cnt: HashMap<u64, (f64, usize)> = HashMap::new();
    for r in rows.iter() {
        let e = cnt.entry(r.price.to_bits()).or_insert((r.price, 0usize));
        e.1 += 1;
    }
    let mut best_price = f64::NAN;
    let mut best_cnt = 0usize;
    let mut first = true;
    for (_, (p, c)) in cnt.iter() {
        if first || *c > best_cnt || (*c == best_cnt && *p < best_price) {
            best_price = *p;
            best_cnt = *c;
            first = false;
        }
    }

    // 2) 该价位上相邻成交的时间间隔（只在该价位的行之间做差，第一条 NaN）
    let mut duras: Vec<f64> = Vec::new();
    let mut prev: Option<i64> = None;
    for r in rows.iter() {
        if r.price == best_price {
            match prev {
                Some(p) => duras.push((r.time_us - p) as f64 * 1000.0 / 1e9),
                None => duras.push(f64::NAN),
            }
            prev = Some(r.time_us);
        }
    }
    out[O_DURAS] = autocorr(&duras, 1);
    out[O_DURAS + 1] = autocorr(&duras, 2);
    out[O_DURAS + 2] = autocorr(&duras, 3);

    // 3) 连续段编号：不在该价位的行，每连续一段一个编号（从 2 开始）
    let mut ids: Vec<f64> = Vec::with_capacity(rows.len());
    let mut run = 0usize;
    let mut cur = 0.0f64;
    for i in 0..rows.len() {
        if rows[i].price == best_price {
            ids.push(1.0);
        } else {
            if i == 0 || rows[i - 1].price == best_price {
                run += 1;
                cur = (run + 1) as f64;
            }
            ids.push(cur);
        }
    }

    // 4) 丢掉编号为 1 的行，按 flag 过滤，turnover 归一化
    let mut rows2: Vec<(f64, i64, f64)> = Vec::new(); // (run id, time, normalized turnover)
    let mut to_sum = 0.0f64;
    for i in 0..rows.len() {
        if ids[i] == 1.0 {
            continue;
        }
        if let Some(w) = flag {
            if rows[i].flag != w {
                continue;
            }
        }
        to_sum += rows[i].turnover;
        rows2.push((ids[i], rows[i].time_us, rows[i].turnover));
    }
    let n_rows2 = rows2.len();

    // 5) 按段求和 / 计数（顺序按段号升序，与 groupby 一致）
    let mut keys: Vec<f64> = Vec::new();
    {
        let mut seen: HashMap<u64, usize> = HashMap::new();
        for (id, _, _) in rows2.iter() {
            let k = id.to_bits();
            if !seen.contains_key(&k) {
                seen.insert(k, keys.len());
                keys.push(*id);
            }
        }
        keys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    let n_keys = keys.len();
    let mut itv1 = vec![0.0f64; n_keys];
    let mut itv2cnt = vec![0.0f64; n_keys];
    {
        let mut pos: HashMap<u64, usize> = HashMap::with_capacity(n_keys);
        for (i, k) in keys.iter().enumerate() {
            pos.insert(k.to_bits(), i);
        }
        for (id, _, to) in rows2.iter() {
            let p = pos[&id.to_bits()];
            itv1[p] += to / to_sum;
            itv2cnt[p] += 1.0;
        }
    }
    let itv2: Vec<f64> = if n_rows2 > 0 {
        itv2cnt.iter().map(|c| c / n_rows2 as f64).collect()
    } else {
        vec![f64::NAN; n_keys]
    };

    // itv1 / itv2 的 autocorr 是在按 t 过滤**之前**算的
    out[O_ITV1] = autocorr(&itv1, 1);
    out[O_ITV1 + 1] = autocorr(&itv1, 2);
    out[O_ITV1 + 2] = autocorr(&itv1, 3);
    out[O_ITV2] = autocorr(&itv2, 1);
    out[O_ITV2 + 1] = autocorr(&itv2, 2);
    out[O_ITV2 + 2] = autocorr(&itv2, 3);

    // 6) t：段内 tnext 的最大值 - 段内 tlast 的最小值
    let mut tmax = vec![f64::NAN; n_keys];
    let mut tmin = vec![f64::NAN; n_keys];
    {
        let mut pos: HashMap<u64, usize> = HashMap::with_capacity(n_keys);
        for (i, k) in keys.iter().enumerate() {
            pos.insert(k.to_bits(), i);
        }
        for j in 0..n_rows2 {
            let p = pos[&rows2[j].0.to_bits()];
            if j + 1 < n_rows2 {
                let v = rows2[j + 1].1 as f64;
                if tmax[p].is_nan() || v > tmax[p] {
                    tmax[p] = v;
                }
            }
            if j > 0 {
                let v = rows2[j - 1].1 as f64;
                if tmin[p].is_nan() || v < tmin[p] {
                    tmin[p] = v;
                }
            }
        }
    }
    let mut t: Vec<f64> = Vec::with_capacity(n_keys);
    for i in 0..n_keys {
        let a = tmax[i];
        let b = tmin[i];
        let v = if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            (a - b) * 1000.0 / 1e9
        };
        // replace(0, nan) 之后 dropna
        t.push(if v == 0.0 { f64::NAN } else { v });
    }

    // 7) 按 t 的 index 过滤 itv1 / itv2，再算 s1 / s2 / s4
    let mut k1: Vec<f64> = Vec::new();
    let mut k2: Vec<f64> = Vec::new();
    let mut kt: Vec<f64> = Vec::new();
    for i in 0..n_keys {
        if t[i].is_nan() {
            continue;
        }
        k1.push(itv1[i]);
        k2.push(itv2[i]);
        kt.push(t[i]);
    }
    let s1: Vec<f64> = (0..k1.len()).map(|i| k1[i] / kt[i]).collect();
    let s2: Vec<f64> = (0..k1.len()).map(|i| k2[i] / kt[i]).collect();
    let s4: Vec<f64> = (0..k1.len()).map(|i| k1[i] / k2[i] / kt[i]).collect();

    out[O_S1] = autocorr(&s1, 1);
    out[O_S1 + 1] = autocorr(&s1, 2);
    out[O_S1 + 2] = autocorr(&s1, 3);
    out[O_S2] = autocorr(&s2, 1);
    out[O_S2 + 1] = autocorr(&s2, 2);
    out[O_S2 + 2] = autocorr(&s2, 3);
    out[O_S4] = autocorr(&s4, 1);
    out[O_S4 + 1] = autocorr(&s4, 2);
    out[O_S4 + 2] = autocorr(&s4, 3);

    out
}

/// 逐股 54 个原值（= 旧 get_factor 的返回）。
fn stock_raw(code: &str, date: i64) -> Option<Vec<f64>> {
    let trades = read_trade_f64(code, date)?;
    let rows: Vec<TradeRow> = trades
        .into_iter()
        .filter(|r| r.flag != 32)
        .filter(|r| {
            let d = r.time_us.rem_euclid(DAY_US);
            d >= LD_0930 && d <= LD_1457
        })
        .collect();
    let mut out: Vec<f64> = Vec::with_capacity(N_RAW);
    for flag in [None, Some(66i64), Some(83i64)] {
        out.extend(one_mode(&rows, flag).iter().copied());
    }
    debug_assert_eq!(out.len(), N_RAW);
    Some(out)
}

/// 第一段入口：给 verify_series.py 对账用的逐股原值（54 个）。
/// 注意它不是因子库的形状（因子库是 25 个，走 compute_cross），所以不按 names() 补齐。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let raw = stock_raw(code, date)?;
    Some(raw.into_iter().map(|v| v as f32).collect())
}

// ============================================================================
// 第二段：横截面 rank + de_cross
// ============================================================================

/// 取一路里 offset 起连续 3 个值，拼成一天全市场的列向量。
fn take3(raws: &[Option<Vec<f64>>], block: usize, off: usize) -> Vec<f64> {
    raws.iter()
        .map(|r| match r {
            Some(v) => v[block * N_PER + off],
            None => f64::NAN,
        })
        .collect()
}

/// de_cross(y, [x1, x2])：三条列向量都非 NaN 的股票参与，含截距最小二乘，返回残差（其余 NaN）。
fn de_cross(y: &[f64], x1: &[f64], x2: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut idx: Vec<usize> = Vec::new();
    let mut yy = Vec::new();
    let mut a1 = Vec::new();
    let mut a2 = Vec::new();
    for i in 0..n {
        if y[i].is_finite() && x1[i].is_finite() && x2[i].is_finite() {
            idx.push(i);
            yy.push(y[i]);
            a1.push(x1[i]);
            a2.push(x2[i]);
        }
    }
    let mut out = vec![f64::NAN; n];
    if yy.is_empty() {
        return out;
    }
    let res = ols_resid(&yy, &a1, &a2);
    for (p, &i) in idx.iter().enumerate() {
        out[i] = res[p];
    }
    out
}

/// 横截面入口：一天全市场，返回 (股票代码, 股票优先排布的 25 个值)。
pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    let codes = data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return None;
    }
    let raws: Vec<Option<Vec<f64>>> = codes
        .par_iter()
        .map(|code| stock_raw(code, date))
        .collect();

    // 各组的横截面 rank
    let rank = |block: usize, off: usize| -> Vec<f64> { rank_average(&take3(&raws, block, off)) };
    let absdiff = |a: &[f64], b: &[f64]| -> Vec<f64> {
        (0..a.len()).map(|i| (a[i] - b[i]).abs()).collect()
    };

    let mut vals: HashMap<&'static str, Vec<f64>> = HashMap::new();

    // s4 组：all / buy / sell
    for (block, tag) in [(0usize, "all"), (1, "buy"), (2, "sell")] {
        let r1 = rank(block, O_S4);
        let r2 = rank(block, O_S4 + 1);
        let r3 = rank(block, O_S4 + 2);
        let name = |k: u8| -> &'static str {
            match (tag, k) {
                ("all", 1) => "all_s4_corr1",
                ("all", 2) => "all_s4_corr2",
                ("all", _) => "all_s4_corr3",
                ("buy", 1) => "buy_s4_corr1",
                ("buy", 2) => "buy_s4_corr2",
                ("buy", _) => "buy_s4_corr3",
                (_, 1) => "sell_s4_corr1",
                (_, 2) => "sell_s4_corr2",
                (_, _) => "sell_s4_corr3",
            }
        };
        for k in 1..=3u8 {
            // 旧代码只对 names_save 里点名的做，其余不算
            let nm = name(k);
            if !OUT_NAMES.contains(&nm) {
                continue;
            }
            let (y, xa, xb) = match k {
                1 => (&r1, &r2, &r3),
                2 => (&r2, &r1, &r3),
                _ => (&r3, &r1, &r2),
            };
            vals.insert(nm, de_cross(y, xa, xb));
        }
    }

    // itv 组：|rank(itv1_k) - rank(itv2_k)|
    for (block, tag) in [(0usize, "all"), (1, "buy"), (2, "sell")] {
        let i1: Vec<Vec<f64>> = (0..3).map(|k| rank(block, O_ITV1 + k)).collect();
        let i2: Vec<Vec<f64>> = (0..3).map(|k| rank(block, O_ITV2 + k)).collect();
        let e: Vec<Vec<f64>> = (0..3).map(|k| absdiff(&i1[k], &i2[k])).collect();
        for k in 1..=3u8 {
            let nm: &'static str = match (tag, k) {
                ("all", 1) => "all_dorms_itv1_corr1",
                ("all", 2) => "all_dorms_itv1_corr2",
                ("all", _) => "all_dorms_itv1_corr3",
                ("buy", 1) => "buy_dorms_itv1_corr1",
                ("buy", 2) => "buy_dorms_itv1_corr2",
                ("buy", _) => "buy_dorms_itv1_corr3",
                (_, 1) => "sell_dorms_itv1_corr1",
                (_, 2) => "sell_dorms_itv1_corr2",
                (_, _) => "sell_dorms_itv1_corr3",
            };
            if !OUT_NAMES.contains(&nm) {
                continue;
            }
            let kk = (k - 1) as usize;
            let (y, xa, xb) = match kk {
                0 => (&e[0], &e[1], &e[2]),
                1 => (&e[1], &e[0], &e[2]),
                _ => (&e[2], &e[0], &e[1]),
            };
            vals.insert(nm, de_cross(y, xa, xb));
        }
    }

    // s1s2 组：|rank(s1_k) - rank(s2_k)|
    for (block, tag) in [(0usize, "all"), (1, "buy"), (2, "sell")] {
        let a1: Vec<Vec<f64>> = (0..3).map(|k| rank(block, O_S1 + k)).collect();
        let a2: Vec<Vec<f64>> = (0..3).map(|k| rank(block, O_S2 + k)).collect();
        let f: Vec<Vec<f64>> = (0..3).map(|k| absdiff(&a1[k], &a2[k])).collect();
        for k in 1..=3u8 {
            let nm: &'static str = match (tag, k) {
                ("all", 1) => "all_s1_corr1",
                ("all", 2) => "all_s1_corr2",
                ("all", _) => "all_s1_corr3",
                ("buy", 1) => "buy_s1_corr1",
                ("buy", 2) => "buy_s1_corr2",
                ("buy", _) => "buy_s1_corr3",
                (_, 1) => "sell_s1_corr1",
                (_, 2) => "sell_s1_corr2",
                (_, _) => "sell_s1_corr3",
            };
            if !OUT_NAMES.contains(&nm) {
                continue;
            }
            let kk = (k - 1) as usize;
            let (y, xa, xb) = match kk {
                0 => (&f[0], &f[1], &f[2]),
                1 => (&f[1], &f[0], &f[2]),
                _ => (&f[2], &f[0], &f[1]),
            };
            vals.insert(nm, de_cross(y, xa, xb));
        }
    }

    // duras 组：只有 all
    {
        let d1 = rank(0, O_DURAS);
        let d2 = rank(0, O_DURAS + 1);
        let d3 = rank(0, O_DURAS + 2);
        for k in 1..=3u8 {
            let nm: &'static str = match k {
                1 => "all_dorms_duras_corr1",
                2 => "all_dorms_duras_corr2",
                _ => "all_dorms_duras_corr3",
            };
            if !OUT_NAMES.contains(&nm) {
                continue;
            }
            let (y, xa, xb) = match k {
                1 => (&d1, &d2, &d3),
                2 => (&d2, &d1, &d3),
                _ => (&d3, &d1, &d2),
            };
            vals.insert(nm, de_cross(y, xa, xb));
        }
    }

    let n_codes = codes.len();
    let mut flat: Vec<f32> = Vec::with_capacity(n_codes * N_OUT);
    for i in 0..n_codes {
        for name in OUT_NAMES.iter() {
            let v = match vals.get(name) {
                Some(v) => v[i],
                None => f64::NAN,
            };
            flat.push(v as f32);
        }
    }
    debug_assert_eq!(flat.len(), n_codes * N_OUT);
    Some((codes, flat))
}
