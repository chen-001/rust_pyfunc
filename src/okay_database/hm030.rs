//! hm30「局部趋势与整体趋势」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm30_局部趋势与整体趋势/hm30_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()：盘口快照整张表，index = pd.to_datetime(df['datetime'])，
//!    取 df.last_prc 这一列当序列。
//! 2) s = df.last_prc.between_time('09:30','14:57')；再 s = s[s != 0] 丢掉 last_prc==0 的行。
//!    between_time 两端都含、按**时间分量**比较；NaN != 0 为真，所以 NaN 会留下。
//! 3) 五个时段（顺序即输出顺序）：allday[09:30,14:57]、open30[09:30,10:00]、
//!    morning[09:30,11:30]、afternoon[13:00,14:57]、close30[14:30,14:57]。
//! 4) 每个时段调一次 sing(df)：
//!      df.shape[0] <= 10  → 直接给 61 个 NaN
//!      否则
//!        df_max = rp.max_range_loop(df)      # allow_equal 默认 True
//!        df_min = rp.min_range_loop(df)
//!        t = df.to_frame('full').assign(partmax=df_max, partmin=df_min)
//!        三列 full / partmax / partmin 各自的 mean、std(ddof=1)、skew、max
//!        corrwith(t.shift(k))，k=1..10，每档 3 个相关 + 1 个 (partmax - partmin)
//!        corr = t.corr() → 取 [0,1]、[0,2]、[1,2]
//!        corr_rank = t.rank().corr() → 同样三个
//!        dtw = rp.dtw_distance(a, b, 0) 三个：full×partmax、full×partmin、partmax×partmin
//! 5) 输出 = sing(allday) + sing(open30) + sing(morning) + sing(afternoon) + sing(close30)，
//!    每段 61 个，共 305 个。
//!
//! names：61 个基名 × 后缀 _allday/_open30/_morning/_afternoon/_close30，共 305 个，无重名。
//!
//! ════════════════════════════════════════════════════════════════════
//! 两处必须自己实现（不能调库里的 pyfunction）
//! ════════════════════════════════════════════════════════════════════
//! · max_range_loop / min_range_loop：算法照 src/statistics/mod.rs 逐行复刻
//!   （allow_equal 用的是 pyo3 签名里的默认值 true）。
//! · dtw_distance(a, b, 0)：第三个参数是 radius=0，Sakoe-Chiba 窗口退化成 j==i，
//!   DP 只能沿对角线走，结果就是 Σ|a_i - b_i|（两个序列等长时）。
//!   库里那个是 #[pyfunction]，worker 链路不能碰，所以在本文件里按上面的等价式实现。
//!
//! ════════════════════════════════════════════════════════════════════
//! pandas 语义对照
//! ════════════════════════════════════════════════════════════════════
//! · mean/std：std 用 ddof=1，样本 < 2 给 NaN
//! · skew（nanops.nanskew）：count>2 时 m2=mean(dev²)、m3=mean(dev³)、
//!   g1=m3/m2^1.5，再乘 sqrt(n(n-1))/(n-2)；count<=2 给 NaN
//! · corr：Pearson，成对完整观测；有效点 <2 或某侧方差为 0 给 NaN
//! · rank：method='average'，升序，并列取平均名次，NaN 保持 NaN
//! · shift(k)：整体下移 k 行，前 k 行是 NaN；corrwith 按索引对齐后仍是逐行配对
//!
//! 已知偏差：输出是 f32，maxmin_corr* = partmax_corr - partmin_corr 这类「两个大数相减」
//! 的因子在真值很小时相对误差会被放大，实测本机四个验收组合都在阈值内。

use std::fs::File;
use std::io::Read;

const SEC_US: i64 = 1_000_000;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

const N_SEG: usize = 5;
const PER_SEG: usize = 61;
pub const N_OUT: usize = PER_SEG * N_SEG;

const SEG_LO: [i64; N_SEG] = [LD_0930, LD_0930, LD_0930, LD_1300, LD_1430];
const SEG_HI: [i64; N_SEG] = [LD_1457, LD_1000, LD_1130, LD_1457, LD_1457];
const SEG_SUF: [&str; N_SEG] = ["_allday", "_open30", "_morning", "_afternoon", "_close30"];

const BASE_NAMES: [&str; PER_SEG] = [
    "full_mean", "partmax_mean", "partmin_mean",
    "full_std", "partmax_std", "partmin_std",
    "full_skew", "partmax_skew", "partmin_skew",
    "full_max", "partmax_max", "partmin_max",
    "full_corr1", "partmax_corr1", "partmin_corr1", "maxmin_corr1",
    "full_corr2", "partmax_corr2", "partmin_corr2", "maxmin_corr2",
    "full_corr3", "partmax_corr3", "partmin_corr3", "maxmin_corr3",
    "full_corr4", "partmax_corr4", "partmin_corr4", "maxmin_corr4",
    "full_corr5", "partmax_corr5", "partmin_corr5", "maxmin_corr5",
    "full_corr6", "partmax_corr6", "partmin_corr6", "maxmin_corr6",
    "full_corr7", "partmax_corr7", "partmin_corr7", "maxmin_corr7",
    "full_corr8", "partmax_corr8", "partmin_corr8", "maxmin_corr8",
    "full_corr9", "partmax_corr9", "partmin_corr9", "maxmin_corr9",
    "full_corr10", "partmax_corr10", "partmin_corr10", "maxmin_corr10",
    "corr_full_partmax", "corr_rank_full_partmax",
    "corr_full_partmin", "corr_rank_full_partmin",
    "corr_partmax_partmin", "corr_rank_partmax_partmin",
    "dtw_full_partmax", "dtw_rank_full_partmin", "dtw_partmax_partmin",
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for suf in SEG_SUF.iter() {
        for b in BASE_NAMES.iter() {
            out.push(format!("{}{}", b, suf));
        }
    }
    out
}

// ============================================================================
// 盘口 CSV 读取（只取 datetime 与 last_prc）
// ============================================================================

/// 列索引（0 基）：symbol,datetime,date,time,exchtime,localtime,last_prc,...
const MKT_COL_DATETIME: usize = 1;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_NCOL: usize = 7;

/// "YYYY-MM-DD HH:MM:SS.mmm" → 当天零点起的微秒数
fn parse_time_us(s: &[u8]) -> i64 {
    if s.len() < 19 || s[10] != b' ' {
        return -1;
    }
    let d = |i: usize| -> i64 { (s[i] - b'0') as i64 };
    let hh = d(11) * 10 + d(12);
    let mi = d(14) * 10 + d(15);
    let ss = d(17) * 10 + d(18);
    let mut us = ((hh * 60 + mi) * 60 + ss) * SEC_US;
    if s.len() > 20 && s[19] == b'.' {
        let mut frac = 0i64;
        let mut scale = 1i64;
        let mut k = 20usize;
        while k < s.len() && s[k].is_ascii_digit() {
            frac = frac * 10 + (s[k] - b'0') as i64;
            scale *= 10;
            k += 1;
        }
        us += frac * SEC_US / scale;
    }
    us
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

/// 读盘口 CSV，返回 (时间分量微秒, last_prc) 列表，保持文件行序。
fn read_snapshots(code: &str, date: i64) -> Option<Vec<(i64, f64)>> {
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

    let mut out: Vec<(i64, f64)> = Vec::with_capacity(content.len() / 400 + 1);
    let mut lines = content.lines();
    lines.next();
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
        let t = parse_time_us(f[MKT_COL_DATETIME]);
        out.push((t, parse_f64_at(f[MKT_COL_LAST_PRC])));
    }
    Some(out)
}

// ============================================================================
// pandas / 库函数语义
// ============================================================================

/// src/statistics/mod.rs 的 max_range_loop(allow_equal=true)
fn max_range_loop(s: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(s.len());
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..s.len() {
        while let Some(&j) = stack.last() {
            if s[j] > s[i] {
                out.push((i - j) as f64);
                break;
            }
            stack.pop();
        }
        if stack.is_empty() {
            out.push((i + 1) as f64);
        }
        stack.push(i);
    }
    out
}

/// src/statistics/mod.rs 的 min_range_loop(allow_equal=true)
fn min_range_loop(s: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(s.len());
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..s.len() {
        while let Some(&j) = stack.last() {
            if s[j] < s[i] {
                out.push((i - j) as f64);
                break;
            }
            stack.pop();
        }
        if stack.is_empty() {
            out.push((i + 1) as f64);
        }
        stack.push(i);
    }
    out
}

fn mean_of(v: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0;
    for &x in v {
        if x.is_finite() {
            n += 1;
            s += x;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

fn std_of(v: &[f64]) -> f64 {
    let m = mean_of(v);
    if !m.is_finite() {
        return f64::NAN;
    }
    let mut n = 0usize;
    let mut ss = 0.0;
    for &x in v {
        if x.is_finite() {
            n += 1;
            let d = x - m;
            ss += d * d;
        }
    }
    if n < 2 {
        return f64::NAN;
    }
    (ss / (n as f64 - 1.0)).sqrt()
}

/// pandas nanops.nanskew
fn skew_of(v: &[f64]) -> f64 {
    let m = mean_of(v);
    if !m.is_finite() {
        return f64::NAN;
    }
    let mut n = 0usize;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in v {
        if x.is_finite() {
            n += 1;
            let d = x - m;
            m2 += d * d;
            m3 += d * d * d;
        }
    }
    if n <= 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    m2 /= nf;
    m3 /= nf;
    let g1 = m3 / m2.powf(1.5);
    g1 * (nf * (nf - 1.0)).sqrt() / (nf - 2.0)
}

fn max_of(v: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &x in v {
        if !x.is_finite() {
            continue;
        }
        if !m.is_finite() || x > m {
            m = x;
        }
    }
    m
}

/// Pearson（成对完整观测）。与 pandas 的 nancorr 在 1e-15 量级一致。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for i in 0..x.len() {
        if x[i].is_finite() && y[i].is_finite() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ys.iter().sum::<f64>() / n as f64;
    let mut vx = 0.0;
    let mut vy = 0.0;
    let mut cxy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        vx += dx * dx;
        vy += dy * dy;
        cxy += dx * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cxy / (vx.sqrt() * vy.sqrt())
}

/// 全体下移 k 行（前 k 行为 NaN），即 pandas 的 s.shift(k)
fn shift(v: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; v.len()];
    for i in k..v.len() {
        out[i] = v[i - k];
    }
    out
}

/// pandas rank(method='average')：升序，并列取平均名次，NaN 保持 NaN
fn rank_average(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| v[i].is_finite()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && v[idx[j]] == v[idx[i]] {
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

/// rp.dtw_distance(a, b, 0)：radius=0 时 Sakoe-Chiba 窗口只剩 j==i，
/// DP 只能沿对角线走，等价于逐点绝对差之和。
fn dtw_diag(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        s += (a[i] - b[i]).abs();
    }
    if a.len() != b.len() {
        return f64::NAN;
    }
    s
}

// ============================================================================
// 单段 61 个值
// ============================================================================

fn sing(full: &[f64]) -> [f64; PER_SEG] {
    let mut res = [f64::NAN; PER_SEG];
    if full.len() <= 10 {
        return res;
    }
    let partmax = max_range_loop(full);
    let partmin = min_range_loop(full);
    let cols: [&[f64]; 3] = [full, &partmax, &partmin];

    for c in 0..3 {
        res[c] = mean_of(cols[c]);
        res[3 + c] = std_of(cols[c]);
        res[6 + c] = skew_of(cols[c]);
        res[9 + c] = max_of(cols[c]);
    }
    for k in 1..=10usize {
        let sh: [Vec<f64>; 3] = [shift(full, k), shift(&partmax, k), shift(&partmin, k)];
        let base = 12 + (k - 1) * 4;
        for c in 0..3 {
            res[base + c] = pearson(cols[c], &sh[c]);
        }
        res[base + 3] = res[base + 1] - res[base + 2];
    }
    res[52] = pearson(full, &partmax);
    res[54] = pearson(full, &partmin);
    res[56] = pearson(&partmax, &partmin);
    let rf = rank_average(full);
    let ra = rank_average(&partmax);
    let rb = rank_average(&partmin);
    res[53] = pearson(&rf, &ra);
    res[55] = pearson(&rf, &rb);
    res[57] = pearson(&ra, &rb);
    res[58] = dtw_diag(full, &partmax);
    res[59] = dtw_diag(full, &partmin);
    res[60] = dtw_diag(&partmax, &partmin);
    res
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let snaps = read_snapshots(code, date)?;
    // 先切 [09:30,14:57]，再丢掉 last_prc==0（NaN != 0 为真，NaN 留下）
    let keep: Vec<(i64, f64)> = snaps
        .into_iter()
        .filter(|&(t, v)| t >= LD_0930 && t <= LD_1457 && v != 0.0)
        .collect();

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for s in 0..N_SEG {
        let vals: Vec<f64> = keep
            .iter()
            .filter(|&&(t, _)| t >= SEG_LO[s] && t <= SEG_HI[s])
            .map(|&(_, v)| v)
            .collect();
        out.extend_from_slice(&sing(&vals));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
