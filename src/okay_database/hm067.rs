//! hm67「车灯」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm67_车灯/hm67_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔」一张表，窗口由旧代码自己切（adjust_afternoon）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0) → 丢掉 flag==32 的撤单行；
//!    deal_data 把 exchtime 从 epoch 微秒转成 pandas Timestamp（+8 小时）。
//!    这里不能靠 read_trade_fast_inner 的 with_retreat（它按字节比 "32"，
//!    2025 年的明细写的是 "32.0" 过滤不掉），也不能用 f32 读取器
//!    （turnover/volume 的组内求和是比值与分位的输入，f32 精度不够），
//!    所以逐笔 CSV 在本文件里按 f64 自己解析。
//! 2) df = dwo.adjust_afternoon(df)：上午 [09:30:00, 11:30:00] 原样保留，
//!    下午 [13:00:00, 14:57:00] 的**索引**整体前移 90 分钟，然后 concat + reset_index。
//!    前移这一点很重要：order_contamination 拿到的 exchtime 是前移之后的，
//!    所以下午第一笔会和上午最后一笔落在同一个 1 秒/10 秒窗口里。
//!    between_time 走的是 DatetimeIndex._get_time_micros()（微秒精度、两端都含），
//!    这里按「pandas ns 值 rem_euclid(一天) / 1000」算钟点，逐位对齐。
//! 3) sing(df, perc, seconds) 里先算
//!      car_ask = rp.order_contamination(exchtime, ask_order, volume, perc, seconds)
//!      car_bid = 同，用 bid_order
//!    order_contamination 是 #[pyfunction]，worker 二进制不能调（不带 libpython），
//!    所以下面照抄了 src/order_contamination.rs 里那段核心（整数逻辑，逐行一致）。
//! 4) 分组基础：temp_ask = 按 ask_order 的 turnover 之和 / volume 之和，
//!    temp_bid 同理；temps = concat([temp_ask, temp_bid])；
//!    temp_high = temps[temps >= temps.quantile(0.8)]，temp_low 用 0.2 与 <=。
//!    分位照 numpy 的 method='linear'（(n-1)*q，t>=0.5 时用 b-(b-a)*(1-t)）。
//! 5) df_high = df[car_ask.isin(temp_high.index) | car_bid.isin(temp_high.index)]，
//!    df_low 同理。one(df) 在 df、df_high、df_low 上各算 156 个指标，
//!    res = one(df) + one(high) + one(low) + (high - df) + (low - df)。
//! 6) 最后 sing(df,1,1) + sing(df,1,10) + sing(df,10,1)，共 2340 个值。
//!
//! names_single（=names）：指标 156 个 × 分组 5 个 × 参数 3 组，
//!   外层是参数（1perc1s / 1perc10s / 10perc1s），内层是分组，最内层是指标。
//! names_save：见 hm67_okay_database.py 的 names_save（72 个，含 _fold 展开）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 对齐过程中确认过的两处坑
//! ════════════════════════════════════════════════════════════════════
//! · exchtime → pandas Timestamp 不是「微秒 × 1000」：走
//!   cast_from_unit_vectorized（val=us/1e6；base=截断；frac=rint(frac*1e9)/1e9；
//!   ns=base*1e9+(i64)(frac*1e9)），自带 ±120ns 噪声。实测 000001@20250303
//!   有 72787/76488 行与「微秒×1000」不同（最大 114ns）。order_contamination 拿
//!   这个纳秒值做 ±1e9/±1e10 的窗口比较，所以必须照抄这条链。
//!   rint 是四舍六入五成双，Rust 的 f64::round() 不是。
//! · rates_ask = new_ask/old_ask_only 在做分组后的子集上，new_ask 的键
//!   （浸染后的委托号）不一定都出现在该子集的 ask_order 里，这时 pandas 的对齐
//!   会补 NaN。这些 NaN 参与 mean/std/autocorr/trend（NaN 只被跳过、位置仍在），
//!   一开始漏掉这批 NaN 时 allday 组对得上、高/低组全错。
//!
//! 数值原语：sum/mean/std/skew/kurt 走 numpy 的成对求和（PW_BLOCKSIZE=128、8 路展开）
//! 与 pandas nanops 的两遍算法；groupby 的 turnover 之和用 Kahan（实测 pandas 就是
//! Kahan，顺序累加有 20/21990 个组对不上）；corr 用 np.corrcoef 那套
//! cov/sqrt(vx*vy)（实测与 BLAS 版差 ~1e-18，验收阈值 1e-4）。
//! 实测四组对照最大相对误差 3e-9。

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

// ============================ order_contamination ============================
// 照抄 src/order_contamination.rs 的 order_contamination 核心（去掉 PyO3 外壳）。
fn order_contamination(
    exchtime: &[i64],
    order: &[i64],
    volume: &[i64],
    top_percentile: f64,
    time_window_seconds: f64,
) -> Vec<i64> {
    let time_window_ns = (time_window_seconds * 1_000_000_000.0) as i64;
    let n = exchtime.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order_volumes: HashMap<i64, i64> = HashMap::with_capacity(n / 4);
    for i in 0..n {
        *order_volumes.entry(order[i]).or_insert(0) += volume[i];
    }
    let mut volumes: Vec<i64> = order_volumes.values().cloned().collect();
    let percentile_ratio = (top_percentile / 100.0).min(1.0).max(0.01);
    let top_count = ((volumes.len() as f64 * percentile_ratio).ceil() as usize)
        .max(1)
        .min(volumes.len());
    let threshold = if top_count > 0 && top_count <= volumes.len() {
        volumes.select_nth_unstable_by(top_count - 1, |a, b| b.cmp(a));
        volumes[top_count - 1]
    } else {
        volumes.sort_unstable_by(|a, b| b.cmp(a));
        volumes.get(top_count.saturating_sub(1)).cloned().unwrap_or(0)
    };
    let mut is_large_order = vec![false; n];
    let mut large_order_positions = Vec::with_capacity(n / 10);
    for i in 0..n {
        if let Some(&total_vol) = order_volumes.get(&order[i]) {
            if total_vol >= threshold {
                is_large_order[i] = true;
                large_order_positions.push(i);
            }
        }
    }
    let mut result = order.to_vec();
    let large_times: Vec<i64> = large_order_positions
        .iter()
        .map(|&pos| exchtime[pos])
        .collect();
    if large_times.is_empty() {
        return result;
    }
    for i in 0..n {
        if is_large_order[i] {
            continue;
        }
        let current_time = exchtime[i];
        let window_start = current_time - time_window_ns;
        let window_end = current_time + time_window_ns;
        let left_idx = large_times
            .binary_search(&window_start)
            .unwrap_or_else(|i| i);
        let right_idx = large_times.binary_search(&window_end).unwrap_or_else(|i| i);
        let mut closest_large_order = None;
        let mut min_distance = i64::MAX;
        for j in left_idx..=right_idx.min(large_times.len() - 1) {
            if j >= large_times.len() {
                break;
            }
            let large_time = large_times[j];
            if large_time > window_end {
                break;
            }
            if large_time < window_start {
                continue;
            }
            let distance = (current_time - large_time).abs();
            if distance <= time_window_ns && distance < min_distance {
                min_distance = distance;
                let pos = large_order_positions[j];
                closest_large_order = Some(order[pos]);
            }
        }
        if let Some(large_order_id) = closest_large_order {
            result[i] = large_order_id;
        }
    }
    result
}

// ============================ 一键统计原语 ============================

/// pandas 的 np.corrcoef(a, b)[0,1]：成对完整观测，任一侧 NaN 丢掉这一对。
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

/// pandas 的 Series.autocorr(1)：x[..n-1] 与 x[1..] 的 Pearson。
fn autocorr1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    pearson(&x[..n - 1], &x[1..])
}

/// rp.trend：与自然数序列 (1..n) 的 Pearson，只保留有限值，索引用原始位置。
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

/// numpy 的 percentile(..., method='linear')，先丢掉 NaN。
fn quantile_linear(vals: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = vals.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    let vi = (n - 1) as f64 * q;
    let (pr, ne) = if vi >= (n - 1) as f64 {
        (n - 1, n - 1)
    } else if vi < 0.0 {
        (0, 0)
    } else {
        let p = vi.floor() as usize;
        (p, p + 1)
    };
    let a = v[pr];
    let b = v[ne];
    let gamma = vi - pr as f64;
    if gamma >= 0.5 {
        b - (b - a) * (1.0 - gamma)
    } else {
        a + (b - a) * gamma
    }
}

/// 按 key 分组求和。turnover 用 Kahan（与 pandas groupby.sum 一致），volume 顺序累加。
struct GroupSums {
    keys: Vec<i64>,
    turn: Vec<f64>,
    vol: Vec<f64>,
}

fn group_sums(keys: &[i64], rows: &[usize], volume: &[f64], turnover: &[f64]) -> GroupSums {
    let mut map: HashMap<i64, usize> = HashMap::new();
    let mut ks: Vec<i64> = Vec::new();
    let mut tv: Vec<f64> = Vec::new();
    let mut vv: Vec<f64> = Vec::new();
    let mut comp: Vec<f64> = Vec::new();
    for &i in rows {
        let k = keys[i];
        let idx = match map.get(&k) {
            Some(&x) => x,
            None => {
                let x = ks.len();
                map.insert(k, x);
                ks.push(k);
                tv.push(0.0);
                vv.push(0.0);
                comp.push(0.0);
                x
            }
        };
        let vol = volume[i];
        if !vol.is_nan() {
            vv[idx] += vol;
        }
        let t = turnover[i];
        if !t.is_nan() {
            let y = t - comp[idx];
            let s = tv[idx] + y;
            comp[idx] = (s - tv[idx]) - y;
            tv[idx] = s;
        }
    }
    let mut order: Vec<usize> = (0..ks.len()).collect();
    order.sort_by_key(|&i| ks[i]);
    GroupSums {
        keys: order.iter().map(|&i| ks[i]).collect(),
        turn: order.iter().map(|&i| tv[i]).collect(),
        vol: order.iter().map(|&i| vv[i]).collect(),
    }
}

// ============================ one() ============================

fn one(
    ask_order: &[i64],
    bid_order: &[i64],
    car_ask: &[i64],
    car_bid: &[i64],
    price: &[f64],
    volume: &[f64],
    turnover: &[f64],
    rows: &[usize],
) -> Option<Vec<f64>> {
    let mut out: Vec<f64> = Vec::with_capacity(156);
    let mut hs: HashMap<i64, ()> = HashMap::new();
    for &i in rows {
        hs.insert(ask_order[i], ());
    }
    let n_ask_uniq = hs.len() as f64;
    hs.clear();
    for &i in rows {
        hs.insert(bid_order[i], ());
    }
    let n_bid_uniq = hs.len() as f64;
    hs.clear();
    for &i in rows {
        hs.insert(car_ask[i], ());
    }
    let n_car_ask_uniq = hs.len() as f64;
    hs.clear();
    for &i in rows {
        hs.insert(car_bid[i], ());
    }
    let n_car_bid_uniq = hs.len() as f64;

    out.push(n_ask_uniq - n_car_ask_uniq);
    out.push(n_bid_uniq - n_car_bid_uniq);
    out.push(n_car_ask_uniq / n_ask_uniq);
    out.push(n_car_bid_uniq / n_bid_uniq);

    let aff_ask: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&i| car_ask[i] != ask_order[i])
        .collect();
    let aff_bid: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&i| car_bid[i] != bid_order[i])
        .collect();

    let ao: Vec<f64> = aff_ask.iter().map(|&i| ask_order[i] as f64).collect();
    let bo: Vec<f64> = aff_bid.iter().map(|&i| bid_order[i] as f64).collect();
    let ap: Vec<f64> = aff_ask.iter().map(|&i| price[i]).collect();
    let bp: Vec<f64> = aff_bid.iter().map(|&i| price[i]).collect();
    let av: Vec<f64> = aff_ask.iter().map(|&i| volume[i]).collect();
    let bv: Vec<f64> = aff_bid.iter().map(|&i| volume[i]).collect();

    // 旧代码在 len<2 时 Series.autocorr 抛 ValueError，整段被 except 吞掉 → 全 NaN
    if ao.len() < 2 || bo.len() < 2 {
        return None;
    }

    out.push(nan_mean(&ao));
    out.push(nan_mean(&bo));
    out.push(nan_std(&ao));
    out.push(nan_std(&bo));
    out.push(nan_skew(&ao));
    out.push(nan_skew(&bo));
    out.push(nan_kurt(&ao));
    out.push(nan_kurt(&bo));
    out.push(nan_median(&ao));
    out.push(nan_median(&bo));
    out.push(nan_min(&ao));
    out.push(nan_min(&bo));
    out.push(nan_max(&ao));
    out.push(nan_max(&bo));
    out.push(nan_pw_sum(&ao));
    out.push(nan_pw_sum(&bo));
    out.push(autocorr1(&ao));
    out.push(autocorr1(&bo));
    out.push(pearson(&ao, &ap));
    out.push(pearson(&bo, &bp));
    out.push(pearson(&ao, &av));
    out.push(pearson(&bo, &bv));
    out.push(trend(&ao));
    out.push(trend(&bo));

    let od: Vec<f64> = aff_ask
        .iter()
        .map(|&i| (car_ask[i] - ask_order[i]) as f64)
        .collect();
    let bd: Vec<f64> = aff_bid
        .iter()
        .map(|&i| (car_bid[i] - bid_order[i]) as f64)
        .collect();

    out.push(nan_mean(&od));
    out.push(nan_mean(&bd));
    out.push(nan_std(&od));
    out.push(nan_std(&bd));
    out.push(nan_skew(&od));
    out.push(nan_skew(&bd));
    out.push(nan_kurt(&od));
    out.push(nan_kurt(&bd));
    out.push(nan_median(&od));
    out.push(nan_median(&bd));
    out.push(nan_min(&od));
    out.push(nan_min(&bd));
    out.push(nan_max(&od));
    out.push(nan_max(&bd));
    out.push(nan_pw_sum(&od));
    out.push(nan_pw_sum(&bd));
    out.push(autocorr1(&od));
    out.push(autocorr1(&bd));
    out.push(pearson(&od, &ap));
    out.push(pearson(&bd, &bp));
    out.push(pearson(&od, &av));
    out.push(pearson(&bd, &bv));
    out.push(trend(&od));
    out.push(trend(&bd));

    let oda: Vec<f64> = od.iter().map(|v| v.abs()).collect();
    let bda: Vec<f64> = bd.iter().map(|v| v.abs()).collect();

    out.push(nan_mean(&oda));
    out.push(nan_mean(&bda));
    out.push(nan_std(&oda));
    out.push(nan_std(&bda));
    out.push(nan_skew(&oda));
    out.push(nan_skew(&bda));
    out.push(nan_kurt(&oda));
    out.push(nan_kurt(&bda));
    out.push(nan_median(&oda));
    out.push(nan_median(&bda));
    out.push(nan_max(&oda));
    out.push(nan_max(&bda));
    out.push(nan_pw_sum(&oda));
    out.push(nan_pw_sum(&bda));
    out.push(autocorr1(&oda));
    out.push(autocorr1(&bda));
    out.push(pearson(&oda, &ap));
    out.push(pearson(&bda, &bp));
    out.push(pearson(&oda, &av));
    out.push(pearson(&bda, &bv));
    out.push(trend(&oda));
    out.push(trend(&bda));

    let all_vol: Vec<f64> = rows.iter().map(|&i| volume[i]).collect();
    let tot_vol = nan_pw_sum(&all_vol);

    out.push(nan_pw_sum(&av) / tot_vol);
    out.push(nan_pw_sum(&bv) / tot_vol);
    out.push(nan_mean(&av) / nan_mean(&all_vol));
    out.push(nan_mean(&bv) / nan_mean(&all_vol));
    out.push(nan_std(&av) / nan_std(&all_vol));
    out.push(nan_std(&bv) / nan_std(&all_vol));
    out.push(nan_skew(&av) / nan_skew(&all_vol));
    out.push(nan_skew(&bv) / nan_skew(&all_vol));
    out.push(nan_kurt(&av) / nan_kurt(&all_vol));
    out.push(nan_kurt(&bv) / nan_kurt(&all_vol));
    out.push(autocorr1(&av) / autocorr1(&all_vol));
    out.push(autocorr1(&bv) / autocorr1(&all_vol));
    out.push(pearson(&av, &ap));
    out.push(pearson(&bv, &bp));
    out.push(trend(&av));
    out.push(trend(&bv));

    let avn: Vec<f64> = av.iter().map(|v| v / tot_vol).collect();
    let bvn: Vec<f64> = bv.iter().map(|v| v / tot_vol).collect();
    out.push(nan_std(&avn));
    out.push(nan_std(&bvn));
    out.push(nan_skew(&avn));
    out.push(nan_skew(&bvn));
    out.push(nan_kurt(&avn));
    out.push(nan_kurt(&bvn));
    out.push(nan_median(&avn));
    out.push(nan_median(&bvn));
    out.push(autocorr1(&avn));
    out.push(autocorr1(&bvn));

    // ---- 分组（old = 原始委托号，new = 浸染后的委托号）
    let old_ask = group_sums(ask_order, rows, volume, turnover);
    let old_bid = group_sums(bid_order, rows, volume, turnover);
    let new_ask = group_sums(car_ask, rows, volume, turnover);
    let new_bid = group_sums(car_bid, rows, volume, turnover);

    let old_ask_price: Vec<f64> = old_ask
        .turn
        .iter()
        .zip(old_ask.vol.iter())
        .map(|(t, v)| t / v)
        .collect();
    let old_bid_price: Vec<f64> = old_bid
        .turn
        .iter()
        .zip(old_bid.vol.iter())
        .map(|(t, v)| t / v)
        .collect();
    let new_ask_price: Vec<f64> = new_ask
        .turn
        .iter()
        .zip(new_ask.vol.iter())
        .map(|(t, v)| t / v)
        .collect();
    let new_bid_price: Vec<f64> = new_bid
        .turn
        .iter()
        .zip(new_bid.vol.iter())
        .map(|(t, v)| t / v)
        .collect();

    // old_ask_only = old_ask 中出现在 new_ask 里的部分；new_ask 的键一定是 old_ask 的子集，
    // 所以 rates = new_ask[k]/old_ask[k]，键序就是 new_ask 的键序。
    let pos: HashMap<i64, usize> = old_ask
        .keys
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i))
        .collect();
    let mut rates_ask: Vec<f64> = Vec::with_capacity(new_ask.keys.len());
    let mut rates_ask_key: Vec<f64> = Vec::with_capacity(new_ask.keys.len());
    for (i, &k) in new_ask.keys.iter().enumerate() {
        match pos.get(&k) {
            Some(&j) => rates_ask.push(new_ask.vol[i] / old_ask.vol[j]),
            None => rates_ask.push(f64::NAN),
        }
        rates_ask_key.push(k as f64);
    }
    let posb: HashMap<i64, usize> = old_bid
        .keys
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i))
        .collect();
    let mut rates_bid: Vec<f64> = Vec::with_capacity(new_bid.keys.len());
    let mut rates_bid_key: Vec<f64> = Vec::with_capacity(new_bid.keys.len());
    for (i, &k) in new_bid.keys.iter().enumerate() {
        match posb.get(&k) {
            Some(&j) => rates_bid.push(new_bid.vol[i] / old_bid.vol[j]),
            None => rates_bid.push(f64::NAN),
        }
        rates_bid_key.push(k as f64);
    }

    if new_ask.keys.len() < 2 || new_bid.keys.len() < 2 {
        return None;
    }

    out.push(nan_mean(&rates_ask));
    out.push(nan_mean(&rates_bid));
    out.push(nan_std(&rates_ask));
    out.push(nan_std(&rates_bid));
    out.push(nan_skew(&rates_ask));
    out.push(nan_skew(&rates_bid));
    out.push(nan_kurt(&rates_ask));
    out.push(nan_kurt(&rates_bid));
    out.push(autocorr1(&rates_ask));
    out.push(autocorr1(&rates_bid));
    out.push(trend(&rates_ask));
    out.push(trend(&rates_bid));
    out.push(pearson(&rates_ask, &rates_ask_key));
    out.push(pearson(&rates_bid, &rates_bid_key));

    out.push(nan_mean(&new_ask.vol) / nan_mean(&old_ask.vol));
    out.push(nan_mean(&new_bid.vol) / nan_mean(&old_bid.vol));
    out.push(nan_std(&new_ask.vol) / nan_std(&old_ask.vol));
    out.push(nan_std(&new_bid.vol) / nan_std(&old_bid.vol));
    out.push(nan_skew(&new_ask.vol) / nan_skew(&old_ask.vol));
    out.push(nan_skew(&new_bid.vol) / nan_skew(&old_bid.vol));
    out.push(nan_kurt(&new_ask.vol) / nan_kurt(&old_ask.vol));
    out.push(nan_kurt(&new_bid.vol) / nan_kurt(&old_bid.vol));
    out.push(nan_median(&new_ask.vol) / nan_median(&old_ask.vol));
    out.push(nan_median(&new_bid.vol) / nan_median(&old_bid.vol));

    let nac = pearson(&new_ask.vol, &new_ask_price);
    let nbc = pearson(&new_bid.vol, &new_bid_price);
    out.push(nac);
    out.push(nbc);
    out.push(nac - pearson(&old_ask.vol, &old_ask_price));
    out.push(nbc - pearson(&old_bid.vol, &old_bid_price));

    out.push(nan_mean(&new_ask.vol));
    out.push(nan_mean(&new_bid.vol));
    out.push(nan_std(&new_ask.vol));
    out.push(nan_std(&new_bid.vol));
    out.push(nan_skew(&new_ask.vol));
    out.push(nan_skew(&new_bid.vol));
    out.push(nan_kurt(&new_ask.vol));
    out.push(nan_kurt(&new_bid.vol));
    out.push(nan_median(&new_ask.vol));
    out.push(nan_median(&new_bid.vol));
    out.push(autocorr1(&new_ask.vol));
    out.push(autocorr1(&new_bid.vol));
    out.push(trend(&new_ask.vol));
    out.push(trend(&new_bid.vol));

    out.push(nan_mean(&new_ask.vol) - nan_mean(&old_ask.vol));
    out.push(nan_mean(&new_bid.vol) - nan_mean(&old_bid.vol));
    out.push(nan_std(&new_ask.vol) - nan_std(&old_ask.vol));
    out.push(nan_std(&new_bid.vol) - nan_std(&old_bid.vol));
    out.push(nan_skew(&new_ask.vol) - nan_skew(&old_ask.vol));
    out.push(nan_skew(&new_bid.vol) - nan_skew(&old_bid.vol));
    out.push(nan_kurt(&new_ask.vol) - nan_kurt(&old_ask.vol));
    out.push(nan_kurt(&new_bid.vol) - nan_kurt(&old_bid.vol));
    out.push(nan_median(&new_ask.vol) - nan_median(&old_ask.vol));
    out.push(nan_median(&new_bid.vol) - nan_median(&old_bid.vol));
    out.push(autocorr1(&new_ask.vol) - autocorr1(&old_ask.vol));
    out.push(autocorr1(&new_bid.vol) - autocorr1(&old_bid.vol));
    out.push(trend(&new_ask.vol) - trend(&old_ask.vol));
    out.push(trend(&new_bid.vol) - trend(&old_bid.vol));

    debug_assert_eq!(out.len(), 156);
    Some(out)
}

// ============================ 主入口 ============================

fn read_csv(path: &str) -> Option<Vec<Row>> {
    let mut content = String::new();
    File::open(path).ok()?.read_to_string(&mut content).ok()?;
    let mut out: Vec<Row> = Vec::with_capacity(content.len() / 120 + 1);
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
        let flag = parse_f64_at(f[TRD_COL_FLAG]);
        if flag == 32.0 {
            continue;
        }
        out.push(Row {
            ns: ns_from_us(parse_f64_at(f[TRD_COL_EXCHTIME])),
            price: parse_f64_at(f[TRD_COL_PRICE]),
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

pub fn compute(all: &[Row]) -> Option<Vec<f64>> {
    // adjust_afternoon：上午 [09:30,11:30] 在前，下午 [13:00,14:57]（索引前移 90 分钟）在后
    let adj = adjusted(all);
    if adj.is_empty() {
        return None;
    }
    let n = adj.len();
    let ex: Vec<i64> = adj.iter().map(|r| r.ns).collect();
    let price: Vec<f64> = adj.iter().map(|r| r.price).collect();
    let volume: Vec<f64> = adj.iter().map(|r| r.volume).collect();
    let turnover: Vec<f64> = adj.iter().map(|r| r.turnover).collect();
    let ask_order: Vec<i64> = adj.iter().map(|r| r.ask_order).collect();
    let bid_order: Vec<i64> = adj.iter().map(|r| r.bid_order).collect();
    let vol_i64: Vec<i64> = volume.iter().map(|&v| v as i64).collect();
    let all_rows: Vec<usize> = (0..n).collect();

    let mut result: Vec<f64> = Vec::with_capacity(2340);
    for &(perc, sec) in &[(1.0f64, 1.0f64), (1.0, 10.0), (10.0, 1.0)] {
        let car_ask = order_contamination(&ex, &ask_order, &vol_i64, perc, sec);
        let car_bid = order_contamination(&ex, &bid_order, &vol_i64, perc, sec);

        // temps = concat(ask 组均价, bid 组均价)
        let ga = group_sums(&ask_order, &all_rows, &volume, &turnover);
        let gb = group_sums(&bid_order, &all_rows, &volume, &turnover);
        let ta: Vec<f64> = ga
            .turn
            .iter()
            .zip(ga.vol.iter())
            .map(|(t, v)| t / v)
            .collect();
        let tb: Vec<f64> = gb
            .turn
            .iter()
            .zip(gb.vol.iter())
            .map(|(t, v)| t / v)
            .collect();
        let mut temps: Vec<f64> = Vec::with_capacity(ta.len() + tb.len());
        temps.extend_from_slice(&ta);
        temps.extend_from_slice(&tb);
        let q80 = quantile_linear(&temps, 0.8);
        let q20 = quantile_linear(&temps, 0.2);
        let mut high: HashMap<i64, ()> = HashMap::new();
        let mut low: HashMap<i64, ()> = HashMap::new();
        for (i, &k) in ga.keys.iter().enumerate() {
            if ta[i] >= q80 {
                high.insert(k, ());
            }
            if ta[i] <= q20 {
                low.insert(k, ());
            }
        }
        for (i, &k) in gb.keys.iter().enumerate() {
            if tb[i] >= q80 {
                high.insert(k, ());
            }
            if tb[i] <= q20 {
                low.insert(k, ());
            }
        }
        let rows_high: Vec<usize> = all_rows
            .iter()
            .copied()
            .filter(|&i| high.contains_key(&car_ask[i]) || high.contains_key(&car_bid[i]))
            .collect();
        let rows_low: Vec<usize> = all_rows
            .iter()
            .copied()
            .filter(|&i| low.contains_key(&car_ask[i]) || low.contains_key(&car_bid[i]))
            .collect();

        let r0 = one(
            &ask_order, &bid_order, &car_ask, &car_bid, &price, &volume, &turnover, &all_rows,
        )?;
        let r1 = one(
            &ask_order, &bid_order, &car_ask, &car_bid, &price, &volume, &turnover, &rows_high,
        )?;
        let r2 = one(
            &ask_order, &bid_order, &car_ask, &car_bid, &price, &volume, &turnover, &rows_low,
        )?;
        for i in 0..156 {
            result.push(r0[i]);
        }
        for i in 0..156 {
            result.push(r1[i]);
        }
        for i in 0..156 {
            result.push(r2[i]);
        }
        for i in 0..156 {
            result.push(r1[i] - r0[i]);
        }
        for i in 0..156 {
            result.push(r2[i] - r0[i]);
        }
    }
    debug_assert_eq!(result.len(), 2340);
    Some(result)
}

fn adjusted(all: &[Row]) -> Vec<Row> {
    let mut adj: Vec<Row> = Vec::with_capacity(all.len());
    for r in all.iter() {
        let t = tod_us(r.ns);
        if (34_200 * 1_000_000..=41_400 * 1_000_000).contains(&t) {
            adj.push(*r);
        }
    }
    for r in all.iter() {
        let t = tod_us(r.ns);
        if (46_800 * 1_000_000..=53_820 * 1_000_000).contains(&t) {
            let mut c = *r;
            c.ns -= 5_400 * NS_SEC; // adjust_afternoon: 下午整体前移 90 分钟
            adj.push(c);
        }
    }
    adj
}


const METRIC_NAMES: [&str; 156] = [
    "nunique_diff_ask",
    "nunique_diff_bid",
    "nunique_diff_ask_rate",
    "nunique_diff_bid_rate",
    "ask_affected_order_mean",
    "bid_affected_order_mean",
    "ask_affected_order_std",
    "bid_affected_order_std",
    "ask_affected_order_skew",
    "bid_affected_order_skew",
    "ask_affected_order_kurt",
    "bid_affected_order_kurt",
    "ask_affected_order_median",
    "bid_affected_order_median",
    "ask_affected_order_min",
    "bid_affected_order_min",
    "ask_affected_order_max",
    "bid_affected_order_max",
    "ask_affected_order_sum",
    "bid_affected_order_sum",
    "ask_affected_order_autocorr",
    "bid_affected_order_autocorr",
    "ask_affected_order_corr_price",
    "bid_affected_order_corr_price",
    "ask_affected_order_corr_volume",
    "bid_affected_order_corr_volume",
    "ask_affected_order_trend",
    "bid_affected_order_trend",
    "ask_affected_order_diff_mean",
    "bid_affected_order_diff_mean",
    "ask_affected_order_diff_std",
    "bid_affected_order_diff_std",
    "ask_affected_order_diff_skew",
    "bid_affected_order_diff_skew",
    "ask_affected_order_diff_kurt",
    "bid_affected_order_diff_kurt",
    "ask_affected_order_diff_median",
    "bid_affected_order_diff_median",
    "ask_affected_order_diff_min",
    "bid_affected_order_diff_min",
    "ask_affected_order_diff_max",
    "bid_affected_order_diff_max",
    "ask_affected_order_diff_sum",
    "bid_affected_order_diff_sum",
    "ask_affected_order_diff_autocorr",
    "bid_affected_order_diff_autocorr",
    "ask_affected_order_diff_corr_price",
    "bid_affected_order_diff_corr_price",
    "ask_affected_order_diff_corr_volume",
    "bid_affected_order_diff_corr_volume",
    "ask_affected_order_diff_trend",
    "bid_affected_order_diff_trend",
    "ask_affected_order_diffabs_mean",
    "bid_affected_order_diffabs_mean",
    "ask_affected_order_diffabs_std",
    "bid_affected_order_diffabs_std",
    "ask_affected_order_diffabs_skew",
    "bid_affected_order_diffabs_skew",
    "ask_affected_order_diffabs_kurt",
    "bid_affected_order_diffabs_kurt",
    "ask_affected_order_diffabs_median",
    "bid_affected_order_diffabs_median",
    "ask_affected_order_diffabs_max",
    "bid_affected_order_diffabs_max",
    "ask_affected_order_diffabs_sum",
    "bid_affected_order_diffabs_sum",
    "ask_affected_order_diffabs_autocorr",
    "bid_affected_order_diffabs_autocorr",
    "ask_affected_order_diffabs_corr_price",
    "bid_affected_order_diffabs_corr_price",
    "ask_affected_order_diffabs_corr_volume",
    "bid_affected_order_diffabs_corr_volume",
    "ask_affected_order_diffabs_trend",
    "bid_affected_order_diffabs_trend",
    "ask_affected_volume_sum_rate",
    "bid_affected_volume_sum_rate",
    "ask_affected_volume_mean_rate",
    "bid_affected_volume_mean_rate",
    "ask_affected_volume_std_rate",
    "bid_affected_volume_std_rate",
    "ask_affected_volume_skew_rate",
    "bid_affected_volume_skew_rate",
    "ask_affected_volume_kurt_rate",
    "bid_affected_volume_kurt_rate",
    "ask_affected_volume_autocorr_rate",
    "bid_affected_volume_autocorr_rate",
    "ask_affected_volume_corr_price",
    "bid_affected_volume_corr_price",
    "ask_affected_volume_trend",
    "bid_affected_volume_trend",
    "ask_affected_volume_std",
    "bid_affected_volume_std",
    "ask_affected_volume_skew",
    "bid_affected_volume_skew",
    "ask_affected_volume_kurt",
    "bid_affected_volume_kurt",
    "ask_affected_volume_median",
    "bid_affected_volume_median",
    "ask_affected_volume_autocorr",
    "bid_affected_volume_autocorr",
    "rates_ask_mean",
    "rates_bid_mean",
    "rates_ask_std",
    "rates_bid_std",
    "rates_ask_skew",
    "rates_bid_skew",
    "rates_ask_kurt",
    "rates_bid_kurt",
    "rates_ask_autocorr",
    "rates_bid_autocorr",
    "rates_ask_trend",
    "rates_bid_trend",
    "rates_ask_corr_order",
    "rates_bid_corr_order",
    "ask_new_volume_mean_rate",
    "bid_new_volume_mean_rate",
    "ask_new_volume_std_rate",
    "bid_new_volume_std_rate",
    "ask_new_volume_skew_rate",
    "bid_new_volume_skew_rate",
    "ask_new_volume_kurt_rate",
    "bid_new_volume_kurt_rate",
    "ask_new_volume_median_rate",
    "bid_new_volume_median_rate",
    "ask_new_corr_price_volume",
    "bid_new_corr_price_volume",
    "ask_newold_corr_price_volume",
    "bid_newold_corr_price_volume",
    "ask_new_order_mean",
    "bid_new_order_mean",
    "ask_new_order_std",
    "bid_new_order_std",
    "ask_new_order_skew",
    "bid_new_order_skew",
    "ask_new_order_kurt",
    "bid_new_order_kurt",
    "ask_new_order_median",
    "bid_new_order_median",
    "ask_new_order_autocorr",
    "bid_new_order_autocorr",
    "ask_new_order_trend",
    "bid_new_order_trend",
    "ask_newold_order_mean",
    "bid_newold_order_mean",
    "ask_newold_order_std",
    "bid_newold_order_std",
    "ask_newold_order_skew",
    "bid_newold_order_skew",
    "ask_newold_order_kurt",
    "bid_newold_order_kurt",
    "ask_newold_order_median",
    "bid_newold_order_median",
    "ask_newold_order_autocorr",
    "bid_newold_order_autocorr",
    "ask_newold_order_trend",
    "bid_newold_order_trend",
];

/// 每只股票每个时段 156 个指标 × 5 个分组 × 3 组 (perc, seconds)
pub const N_OUT: usize = 156 * 5 * 3;

const GROUP_NAMES: [&str; 5] = [
    "quan2tian1",
    "gao1jia420perc",
    "di1jia420perc",
    "gao1jia420percjian3quan2tian1",
    "di1jia420percjian3quan2tian1",
];
const PARAM_NAMES: [&str; 3] = ["1perc1s", "1perc10s", "10perc1s"];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for p in PARAM_NAMES.iter() {
        for g in GROUP_NAMES.iter() {
            for m in METRIC_NAMES.iter() {
                out.push(format!("{}_{}_{}", m, g, p));
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
    let rows = read_csv(&path.to_string_lossy())?;
    let vals = compute(&rows)?;
    Some(vals.iter().map(|&v| v as f32).collect())
}

/// hm67 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
