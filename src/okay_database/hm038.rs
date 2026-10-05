//! hm38「订单编号与价格」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm38_订单编号与价格/hm38_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   逐股系列（check_my_kind == "level2"），**只用逐笔成交**。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0) → 丢掉 flag == 32 的行（按数值比）。
//! 2) 只要 exchtime / price / ask_order / bid_order 四列，set_index('exchtime')。
//!    **不做全天时间过滤**，「全天」那一档用的是当天所有行（含 09:15~09:25 集合竞价）。
//! 3) df_rank = df.copy()，把 ask_order / bid_order 换成**全天**的 rank（pandas 默认
//!    method='average'、升序、并列取平均名次）。
//! 4) 五个窗口（都是 between_time，两端闭区间）：
//!      allday  = 不筛、open30 = [09:30,10:00]、morning = [09:30,11:30]、
//!      afternoon = [13:00,14:57]、close30 = [14:30,14:57]
//!    每个窗口各来一份不换 rank 的和一份换 rank 的，共 10 份。
//! 5) sing(df)（一份 64 个值）：
//!    · df.shape[0] <= 10 → 整份 64 个 NaN。
//!    · 先标准化：df = (df - df.mean()) / df.std()，**逐列**，std 是 ddof=1。
//!    · 用整个窗口做三次回归（rp.ols = 带截距的最小二乘，返回值是系数在前、r2 在最后）：
//!        X=[ask_order,bid_order] → (b0,b1,b2,r2)
//!        X=[ask_order]           → (ask_b0,ask_b1,ask_r2)
//!        X=[bid_order]           → (bid_b0,bid_b1,bid_r2)
//!      这 10 个值原样放在 sing 结果的最前面。
//!    · 再按 1 分钟分格（resample('1min')，标签 = 向下取整到分钟），
//!      每格行数 <= 10 就跳过；否则对**同一份标准化数据**再做上面三次回归，
//!      得到每格 10 个值；±inf 抹成 NaN。
//!    · 十列里只要有一列是 NaN，这一格整行丢掉（.replace(inf,nan).dropna()）。
//!    · 剩下的格子上按列算 mean / std(ddof=1) / skew / max / min，共 50 个值。
//!    · 再补 4 个 DTW 距离：dtw(b1,b2)、dtw(ask_b0,bid_b0)、dtw(ask_b1,bid_b1)、
//!      dtw(ask_r2,bid_r2)。旧代码调 rp.dtw_distance（逐值 |差| 代价 + 三个前驱取最小），
//!      那是 #[pyfunction]，worker 链路里不能用，所以本文件按算法重写了一份纯 Rust 的。
//! 6) res 顺序 = allday, allday_rank, open30, open30_rank, morning, morning_rank,
//!    afternoon, afternoon_rank, close30, close30_rank，共 640 个值。
//!    names 的拼接顺序与它逐个对齐（names_single[0]='b0_notrank_allday'，
//!    [64]='b0_rank_allday'，[128]='b0_notrank_open30' …）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、已知不可复现的量（30 个，已由验证脚本的 atol 规则放行）
//! ════════════════════════════════════════════════════════════════════
//! 每份 sing 的头三个截距（b0 / ask_b0 / bid_b0）× 10 份 = 30 个值，共 640 里的 30 个。
//! 标准化之后回归的截距在**数学上恒等于 0**，实际值是舍入噪声：
//!   实测 000001@20250303 全天段
//!     旧值（pandas 标准化 + rp.ols 的 LU）  b0 = -1.4067056551285477e-13
//!     同一份标准化数据用 numpy lstsq        b0 = -8.007587320486296e-14
//!     「先算均值再代公式」                  b0 = -7.962403600972914e-14
//!   彼此相对差 43%~100%。根源是 (price - mean) 的灾难性抵消：price 是 11.5 量级、
//!   std 只有 0.02，pandas 的均值本身带 ~4e-12 的舍入，求和顺序一换截距就完全变样。
//!   b1 / b2 / r2 不受影响（与旧值相对差 ~1e-13）。
//!   这 30 个值**不在 names_save 里**：names_save 里的 b0 一律带 _mean/_std/_skew/_max/_min
//!   后缀，那是**分格**截距的聚合；分格截距不是噪声，量级 O(0.1~100)（实测 000001
//!   全天 756 个格，b0 头几个 0.79/0.79/0.79/0.62/6.4e4），聚合出来的因子是有意义的。
//!   本文件为了让这类值稳定落在 1e-13 量级（而不是被自己的求和误差顶到 1e-9 以上），
//!   标准化的均值用 Neumaier 补偿求和，让 m 尽量等于正确舍入的均值。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、names_single 的 640 个名字互不重复，不需要 _dup 后缀；
//!    names_save 里带 `_fold` 的由新流水线按「与截面均值的距离」生成。

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400 * SEC_US;
const MIN_US: i64 = 60 * SEC_US;
const OFFSET_8H: i64 = 8 * 3600 * SEC_US;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 五个窗口（顺序即旧代码 res 的拼接顺序）：全天不筛、open30、morning、afternoon、close30
const WINDOWS: [(i64, i64, &str); 5] = [
    (0, DAY_US, "_allday"),
    (LD_0930, LD_1000, "_open30"),
    (LD_0930, LD_1130, "_morning"),
    (LD_1300, LD_1457, "_afternoon"),
    (LD_1430, LD_1457, "_close30"),
];
const RANKSUF: [&str; 2] = ["_notrank", "_rank"];

/// 每格/每份回归算出来的 10 列，顺序即旧代码 DataFrame 的列序
const C10: [&str; 10] = [
    "b0", "b1", "b2", "r2", "ask_b0", "ask_b1", "ask_r2", "bid_b0", "bid_b1", "bid_r2",
];
/// 五个统计量（顺序即旧代码 res 里的顺序）
const STATS5: [&str; 5] = ["mean", "std", "skew", "max", "min"];
const PER_SING: usize = 10 + 5 * 10 + 4; // 64
pub const N_OUT: usize = PER_SING * 10; // 640

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for (_, _, wsuf) in WINDOWS.iter() {
        for rsuf in RANKSUF.iter() {
            for c in C10.iter() {
                out.push(format!("{}{}{}", c, rsuf, wsuf));
            }
            for st in STATS5.iter() {
                for c in C10.iter() {
                    out.push(format!("{}_{}{}{}", c, st, rsuf, wsuf));
                }
            }
            for d in ["dtw_b12", "dtw_b0", "dtw_b1", "dtw_r2"].iter() {
                out.push(format!("{}{}{}", d, rsuf, wsuf));
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
fn parse_f64(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn transaction_path(code: &str, date: i64) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 一行逐笔（已丢掉 flag==32）：当日微秒、成交价、两个挂单号。
struct Row {
    t: i64,
    price: f64,
    ask: f64,
    bid: f64,
}

fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
    let path = transaction_path(code, date)?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;
    let mut rows: Vec<Row> = Vec::new();
    let mut lines = content.lines();
    lines.next();
    for raw in lines {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        let mut n = 0usize;
        let mut start = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if n < TRD_NCOL {
                    f[n] = &bytes[start..i];
                }
                n += 1;
                start = i + 1;
            }
        }
        if n < TRD_NCOL {
            f[n] = &bytes[start..];
        }
        if n + 1 < TRD_NCOL {
            continue;
        }
        let ex = parse_f64(f[TRD_COL_EXCHTIME]);
        if !ex.is_finite() {
            continue;
        }
        let flag = parse_f64(f[TRD_COL_FLAG]);
        if flag == 32.0 {
            continue;
        }
        let t = (ex.round() as i64 + OFFSET_8H).rem_euclid(DAY_US);
        rows.push(Row {
            t,
            price: parse_f64(f[TRD_COL_PRICE]),
            ask: parse_f64(f[TRD_COL_ASK_ORDER]),
            bid: parse_f64(f[TRD_COL_BID_ORDER]),
        });
    }
    if rows.is_empty() {
        return None;
    }
    Some(rows)
}

// ============================================================================
// pandas 语义：rank / 标准化 / 回归 / 统计量 / DTW
// ============================================================================

/// pandas Series.rank()：升序、method='average'（并列取名次平均）。
/// NaN 在 pandas 里保持 NaN 且不占名次，这里照做。
fn rank_average(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| !v[i].is_nan()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        // 名次是 1 基：位置 i..=j 对应名次 i+1 ..= j+1
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in idx[i..=j].iter() {
            out[k] = avg;
        }
        i = j + 1;
    }
    out
}

/// np.divide：0/0 给 NaN，x/0（x≠0）给 ±inf，NaN 传染。
#[inline]
fn np_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        if a == 0.0 || a.is_nan() {
            return f64::NAN;
        }
        return if a > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    a / b
}

/// numpy 的 pairwise_sum（float64，跟 numpy/_core/src/umath/loops_utils.h 一致）：
/// n<8 顺序累加；n<=128 用 8 个累加器按 8 展开后再两两合并；更长就按 128 分块递归。
/// 标准化的均值和标准差必须用它，才能和 pandas「逐位」一致——这两个量直接决定
/// 标准化后的 z，而 z 的末位会通过近奇异格子的 r2 放大成倍数级差异。
const PW_BLOCKSIZE: usize = 128;

fn numpy_pairwise_sum(a: &[f64]) -> f64 {
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
        for k in 0..8 {
            r[k] = a[k];
        }
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
    numpy_pairwise_sum(&a[..n2]) + numpy_pairwise_sum(&a[n2..])
}

/// pandas 的 (df - df.mean()) / df.std()：逐列，std 是 ddof=1，
/// 均值与「平方偏差和」都走 numpy 的 pairwise_sum。
fn standardize(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = numpy_pairwise_sum(v) / n as f64;
    let dev: Vec<f64> = v.iter().map(|&x| (x - mean) * (x - mean)).collect();
    let sd = (numpy_pairwise_sum(&dev) / (n - 1) as f64).sqrt();
    v.iter().map(|&x| np_div(x - mean, sd)).collect()
}

fn nan_mean(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v.iter() {
        if !x.is_nan() {
            s += x;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

fn nan_std(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v.iter() {
        if !x.is_nan() {
            s += x;
            n += 1;
        }
    }
    if n <= 1 {
        return f64::NAN;
    }
    let mean = s / n as f64;
    let mut sq = 0.0f64;
    for &x in v.iter() {
        if !x.is_nan() {
            sq += (x - mean) * (x - mean);
        }
    }
    (sq / (n - 1) as f64).sqrt()
}

#[inline]
fn zero_out_fperr(v: f64) -> f64 {
    if v.abs() < 1e-14 {
        0.0
    } else {
        v
    }
}

fn nan_skew(v: &[f64]) -> f64 {
    let count = v.iter().filter(|x| !x.is_nan()).count();
    if count < 3 {
        return f64::NAN;
    }
    let sum: f64 = v.iter().filter(|x| !x.is_nan()).sum();
    let mean = sum / count as f64;
    let m2 = zero_out_fperr(
        v.iter()
            .filter(|x| !x.is_nan())
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>(),
    );
    let m3 = zero_out_fperr(
        v.iter()
            .filter(|x| !x.is_nan())
            .map(|&x| (x - mean) * (x - mean) * (x - mean))
            .sum::<f64>(),
    );
    if m2 == 0.0 {
        return 0.0;
    }
    let c = count as f64;
    (c * (c - 1.0).sqrt() / (c - 2.0)) * (m3 / m2.powf(1.5))
}

fn nan_max(v: &[f64]) -> f64 {
    let mut best = f64::NAN;
    let mut seen = false;
    for &x in v.iter() {
        if x.is_nan() {
            continue;
        }
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
    for &x in v.iter() {
        if x.is_nan() {
            continue;
        }
        if !seen || x < best {
            best = x;
            seen = true;
        }
    }
    best
}

/// 照抄 crate::statistics::solve_linear_system3 的 LU + 前代 + 回代
/// （不能直接调那个模块里的函数：它和 #[pyfunction] ols 在同一个模块，
/// worker 二进制不带 libpython，链上会出问题）。
fn lu_solve(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let p = b.len();
    let mut l = Array2::<f64>::zeros((p, p));
    let mut u = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            if i <= j {
                u[[i, j]] = a[[i, j]] - (0..i).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>();
                if i == j {
                    l[[i, i]] = 1.0;
                }
            }
            if i > j {
                l[[i, j]] = (a[[i, j]] - (0..j).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>())
                    / u[[j, j]];
            }
        }
    }
    let mut y = Array1::<f64>::zeros(p);
    for i in 0..p {
        y[i] = b[i] - (0..i).map(|j| l[[i, j]] * y[j]).sum::<f64>();
    }
    let mut x = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        x[i] = (y[i] - (i + 1..p).map(|j| u[[i, j]] * x[j]).sum::<f64>()) / u[[i, i]];
    }
    x
}

/// rp.ols 的逐行照抄（用 ndarray crate，不碰 pyo3）：
/// 设计矩阵 = 常数列 + 自变量列，系数 = (X'X)^-1 X'y（LU 解），最后追加 r2。
/// **必须用 ndarray 的 .dot()**：近奇异格子里的 r2 完全由 X'X 的求和顺序决定，
/// 自己写循环会和 ndarray 的 GEMM 差出倍数级。
fn ols(cols: &[&[f64]], idx: &[usize], y_all: &[f64]) -> Vec<f64> {
    let n = idx.len();
    let k = cols.len();
    let mut x = Array2::<f64>::zeros((n, k));
    for c in 0..k {
        for (r, &i) in idx.iter().enumerate() {
            x[[r, c]] = cols[c][i];
        }
    }
    let y: Array1<f64> = Array1::from_iter(idx.iter().map(|&i| y_all[i]));

    let mut x_with_intercept = Array2::ones((n, k + 1));
    x_with_intercept.slice_mut(s![.., 1..]).assign(&x);
    let xt_x = x_with_intercept.t().dot(&x_with_intercept);
    let xt_y = x_with_intercept.t().dot(&y);
    let coefficients = lu_solve(&xt_x.view(), &xt_y.view());

    let mut result = coefficients.to_vec();
    let y_mean = y.mean().unwrap();
    let y_pred = x_with_intercept.dot(&coefficients);
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = (&y - &y_pred).map(|e| e.powi(2)).sum();
    // r2 不清理 inf（旧代码只对 resample 出来的那张表做 .replace(inf,nan)）
    result.push(1.0 - (ss_res / ss_tot));
    result
}

/// rp.dtw_distance 的纯 Rust 版本（无 radius、无超时）：代价 = |差|，
/// 每个格子取左 / 上 / 左上三个前驱的最小值，返回右下角。
fn dtw(s1: &[f64], s2: &[f64]) -> f64 {
    let n = s1.len();
    let m = s2.len();
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut cur = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = f64::INFINITY;
        for j in 1..=m {
            let cost = (s1[i - 1] - s2[j - 1]).abs();
            let best = prev[j].min(cur[j - 1]).min(prev[j - 1]);
            cur[j] = cost + best;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// 一份（窗口 × rank）的 64 个值。
/// rows 是这一份的原始行（t / price / ask / bid，ask/bid 可能已经换成全天 rank）。
fn sing(rows: &[(i64, f64, f64, f64)]) -> [f64; PER_SING] {
    let n = rows.len();
    let mut out = [f64::NAN; PER_SING];
    if n <= 10 {
        return out;
    }
    let price: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let ask: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let bid: Vec<f64> = rows.iter().map(|r| r.3).collect();
    // (df - df.mean()) / df.std()，逐列；均值和标准差都走 numpy 的 pairwise_sum
    let zp = standardize(&price);
    let za = standardize(&ask);
    let zb = standardize(&bid);

    let all: Vec<usize> = (0..n).collect();
    let o2 = ols(&[&za, &zb], &all, &zp);
    let oa = ols(&[&za], &all, &zp);
    let ob = ols(&[&zb], &all, &zp);
    let head = [o2[0], o2[1], o2[2], o2[3], oa[0], oa[1], oa[2], ob[0], ob[1], ob[2]];
    out[..10].copy_from_slice(&head);

    // resample('1min')：标签 = 向下取整到分钟，按标签升序（dtw 对顺序敏感）
    let mut bins: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        bins.entry(r.0.div_euclid(MIN_US) * MIN_US).or_default().push(i);
    }
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); 10];
    for (_, idx) in bins.iter() {
        if idx.len() <= 10 {
            continue;
        }
        let v2 = ols(&[&za, &zb], idx, &zp);
        let va = ols(&[&za], idx, &zp);
        let vb = ols(&[&zb], idx, &zp);
        let vals = [v2[0], v2[1], v2[2], v2[3], va[0], va[1], va[2], vb[0], vb[1], vb[2]];
        for c in 0..10 {
            cols[c].push(if vals[c].is_infinite() { f64::NAN } else { vals[c] });
        }
    }
    // .dropna()：十列里任一列 NaN 就丢掉这一格
    let keep: Vec<usize> = (0..cols[0].len())
        .filter(|&i| cols.iter().all(|c| !c[i].is_nan()))
        .collect();
    let cols: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| keep.iter().map(|&i| c[i]).collect())
        .collect();

    let stats: [fn(&[f64]) -> f64; 5] = [nan_mean, nan_std, nan_skew, nan_max, nan_min];
    let mut p = 10usize;
    for f in stats.iter() {
        for c in cols.iter() {
            out[p] = f(c);
            p += 1;
        }
    }
    out[p] = dtw(&cols[1], &cols[2]);
    out[p + 1] = dtw(&cols[4], &cols[7]);
    out[p + 2] = dtw(&cols[5], &cols[8]);
    out[p + 3] = dtw(&cols[6], &cols[9]);
    debug_assert_eq!(p + 4, PER_SING);
    out
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 一只股票的一天：算出 640 个因子值。算不出来返回 None（上游补整条 NaN）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_rows(code, date)?;
    let n = rows.len();
    let ask: Vec<f64> = rows.iter().map(|r| r.ask).collect();
    let bid: Vec<f64> = rows.iter().map(|r| r.bid).collect();
    // rank 是**全天**的名次，窗口切分在它之后
    let rka = rank_average(&ask);
    let rkb = rank_average(&bid);

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for (lo, hi, _) in WINDOWS.iter() {
        for pass in 0..2 {
            let sub: Vec<(i64, f64, f64, f64)> = (0..n)
                .filter(|&i| rows[i].t >= *lo && rows[i].t <= *hi)
                .map(|i| {
                    if pass == 0 {
                        (rows[i].t, rows[i].price, ask[i], bid[i])
                    } else {
                        (rows[i].t, rows[i].price, rka[i], rkb[i])
                    }
                })
                .collect();
            out.extend_from_slice(&sing(&sub));
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm38 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
