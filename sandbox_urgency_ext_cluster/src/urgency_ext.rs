//! COR-22「迫切交易联动」补充因子：迫切交易时序聚类程度 + 狭义聚类（89 因子）。
//!
//! 设计报告：research_ext_ideas/design_report_cor22_urgency_cluster.md（用户批准）。
//! 口径与初版 urgency_metrics.rs 完全一致：
//!   ratio = (ask_order - bid_order) / (ask_order + bid_order)，sum<=0 → 0
//!   全市场 q5 / q95 / q95_abs 标记 buy / sell / both 三版本迫切交易
//!   read_trade_fast_inner(code, date, false, true, usize::MAX) + retain(flag != 32)
//!   dir = +1 (flag==83) / -1；时间用 session 秒（上午 [0,7200)，下午平移后 [7200,14220)）
//!
//! 因子布局（每股 89 个，顺序与 ext_names() 一致）：
//!   S1 基准量 3    ：buy/sell/both n_urg
//!   S2 原始聚类 9  ：buy/sell/both × (bucket_ent, cv_iei, gap_med)
//!   S3 相对聚类 12 ：buy/sell/both × (xcv_iei, xburst_b, xent, xdens300)  与自身全部成交做差
//!   S4 截面中性 9  ：buy/sell/both × (csz_cv, csz_ent, csz_gapmed)  按 m 十分位分层组内 z
//!   S5 方向结构 4  ：both × (dir_persist_5s, cross_pairs_5s, gap_ratio_bs, cv_diff_bs)
//!   S6 时段结构 9  ：buy/sell/both × (share_open30, share_late30, mass_share_late30)
//!   S7 狭义时序 21 ：buy/sell/both × (km_best_k, km_gap, km_sil, km2_ratio, km2_frac, db_n, db_frac)
//!   S8 截面狭义 22 ：主聚类几何 13 + 多版本几何 6 + GMM 3（详见下）
//!
//! 确定性要求：所有随机源均固定（xorshift 固定种子、分位等距初始化、均匀参照固定种子）。
//! 截断：m<5 → S2/S3/S6/S7 为 NaN；m<10 → S5 为 NaN；当日有效股数<200 → S8 为 NaN。
//! 速度：S7 的 K-Means 输入在 m>1500 时按等间隔抽样至 1500 点（骨架不变，成本可控）。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;

pub const N_FACTORS: usize = 89;
const MIN_M: usize = 5; // S2/S3/S6/S7 门槛
const MIN_M_S5: usize = 10; // S5 门槛
const M_CAP: usize = 1500; // K-Means 输入抽样上限
const N_BUCKETS: usize = 240; // 60 秒桶
const GAP_SIMS: usize = 10; // gap statistic 均匀参照次数
const KM_ITERS: usize = 50; // 1-D K-Means 迭代上限
const KM4_ITERS: usize = 100; // 4-D K-Means 迭代上限
const GMM_K: usize = 16;
const GMM_ITERS: usize = 100;
const CS_MIN_VALID: usize = 200; // 截面聚类最低有效股数
const EPS_DB: f32 = 60.0; // DBSCAN eps（秒）
const EPS5: f32 = 5.0; // S5 方向对的间隔阈值

// ============================================================
// 工具
// ============================================================

/// Howard Hinnant 算法：y/m/d → 距 1970-01-01 的天数。
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// time_us（含 +8h 偏移、下午已平移）→ 连续交易时段秒（本地 09:30 = 0）。
#[inline]
fn session_seconds(date: i64, time_us: i64) -> f32 {
    let y = date / 10000;
    let mm = ((date / 100) % 100) as u32;
    let dd = (date % 100) as u32;
    let t0 = days_from_civil(y, mm, dd) * 86400 + 34200; // 本地 09:30 → +8h 偏移后 09:30 UTC 位置
    (time_us as f64 / 1e6 - t0 as f64) as f32
}

/// 确定性 xorshift64 随机源。
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    #[inline]
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 列出某天全市场股票代码（横截面枚举，有序去重）。
pub fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/transaction");
    let mut set = std::collections::BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

#[inline]
fn nan() -> f32 {
    f32::NAN
}

// ============================================================
// 基础序列统计（S2/S3/S6）
// ============================================================

/// 间隔变异系数 std(tau)/mean(tau)。m<2 或 mean<=0 → None。
fn cv_of(ts: &[f32]) -> Option<f32> {
    let m = ts.len();
    if m < 2 {
        return None;
    }
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for w in ts.windows(2) {
        let d = (w[1] - w[0]) as f64;
        sum += d;
        sumsq += d * d;
    }
    let n = (m - 1) as f64;
    let mean = sum / n;
    if mean <= 0.0 {
        return None;
    }
    let var = (sumsq / n - mean * mean).max(0.0);
    Some((var.sqrt() / mean) as f32)
}

/// 突发指数 B = (cv-1)/(cv+1) ∈ [-1,1]。
fn burst_b_of(cv: Option<f32>) -> Option<f32> {
    cv.map(|c| (c - 1.0) / (c + 1.0))
}

/// 间隔中位数（秒）。
fn median_gap(ts: &[f32]) -> Option<f32> {
    let m = ts.len();
    if m < 2 {
        return None;
    }
    let mut gaps: Vec<f32> = ts.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_unstable_by(|a, b| a.total_cmp(b));
    Some(median_of(&gaps))
}

/// 间隔均值（秒）。逐笔同微秒连拍使中位数约半数股票为 0，均值不受此影响。
fn mean_gap(ts: &[f32]) -> Option<f32> {
    let m = ts.len();
    if m < 2 {
        return None;
    }
    let s: f64 = ts
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .sum();
    Some((s / (m - 1) as f64) as f32)
}

fn median_of(v: &[f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    }
}

/// 60 秒桶归一化熵 ∈ [0,1]，越大越分散。
fn bucket_ent(ts: &[f32]) -> Option<f32> {
    let m = ts.len();
    if m == 0 {
        return None;
    }
    let mut cnt = [0usize; N_BUCKETS];
    for &t in ts {
        let idx = ((t / 60.0) as usize).min(N_BUCKETS - 1);
        cnt[idx] += 1;
    }
    let mf = m as f32;
    let mut ent = 0.0f64;
    for &c in cnt.iter() {
        if c > 0 {
            let p = c as f64 / mf as f64;
            ent -= p * p.ln();
        }
    }
    Some((ent / (N_BUCKETS as f64).ln()) as f32)
}

/// 最密集连续 5 分钟（300 秒滑窗，步长 60 秒）内笔数占比。ts 有序。
fn dens300(ts: &[f32]) -> Option<f32> {
    let m = ts.len();
    if m == 0 {
        return None;
    }
    let mut best = 0usize;
    let mut r = 0usize;
    for l in 0..m {
        let s = ts[l];
        while r < m && ts[r] < s + 300.0 {
            r += 1;
        }
        best = best.max(r - l);
    }
    Some(best as f32 / m as f32)
}

/// 全流 vs 迫切流 的某统计之差。
fn x_of(a: Option<f32>, b: Option<f32>) -> f32 {
    match (a, b) {
        (Some(x), Some(y)) => x - y,
        _ => nan(),
    }
}

// ============================================================
// 1-D K-Means / gap statistic / silhouette / DBSCAN（S7）
// ============================================================

/// 确定性 1-D K-Means（分位等距初始化 + Lloyd）。ts 已排序。
/// 返回 (assign, wss)。空簇保留旧质心。
fn kmeans1d(ts: &[f32], k: usize) -> (Vec<u32>, f64) {
    let m = ts.len();
    let kk = k.min(m).max(1);
    let mut cents: Vec<f64> = (0..kk)
        .map(|i| {
            let idx = (((i as f64) + 0.5) * (m as f64) / (kk as f64)) as usize;
            ts[idx.min(m - 1)] as f64
        })
        .collect();
    let mut assign = vec![0u32; m];
    let mut newc = vec![0f64; kk];
    let mut cnt = vec![0u32; kk];
    for _ in 0..KM_ITERS {
        for i in 0..m {
            let t = ts[i] as f64;
            let mut best = 0usize;
            let mut bd = (t - cents[0]).abs();
            for c in 1..kk {
                let d = (t - cents[c]).abs();
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            assign[i] = best as u32;
        }
        newc.iter_mut().for_each(|v| *v = 0.0);
        cnt.iter_mut().for_each(|v| *v = 0);
        for i in 0..m {
            let a = assign[i] as usize;
            newc[a] += ts[i] as f64;
            cnt[a] += 1;
        }
        let mut moved = 0.0f64;
        for c in 0..kk {
            if cnt[c] > 0 {
                let nc = newc[c] / cnt[c] as f64;
                moved = moved.max((nc - cents[c]).abs());
                cents[c] = nc;
            }
        }
        if moved < 1e-6 {
            break;
        }
    }
    let mut wss = 0.0f64;
    for i in 0..m {
        let d = ts[i] as f64 - cents[assign[i] as usize];
        wss += d * d;
    }
    (assign, wss)
}

/// 1-D gap statistic：best_k = argmax_k gap(k)，gap(k) = E_ref[log W] - log W_obs。
fn gap_statistic(ts: &[f32]) -> (usize, Vec<f32>) {
    let m = ts.len();
    let tmin = ts[0] as f64;
    let tmax = ts[m - 1] as f64;
    let mut obs_wss = vec![0f64; 5];
    let mut assign5: Vec<Vec<u32>> = Vec::new();
    for k in 1..=5usize {
        let (a, w) = kmeans1d(ts, k);
        obs_wss[k - 1] = w;
        assign5.push(a);
    }
    let mut ref_sum = vec![0f64; 5];
    let mut rng = XorShift64::new(20240819);
    for _ in 0..GAP_SIMS {
        let mut sample: Vec<f32> = (0..m)
            .map(|_| (tmin + rng.next() * (tmax - tmin)) as f32)
            .collect();
        sample.sort_unstable_by(|a, b| a.total_cmp(b));
        for k in 1..=5usize {
            let (_, w) = kmeans1d(&sample, k);
            ref_sum[k - 1] += w.max(1e-30).ln();
        }
    }
    let mut gap = vec![0f32; 5];
    let mut best = 1usize;
    for k in 0..5 {
        gap[k] = ((ref_sum[k] / GAP_SIMS as f64) - obs_wss[k].max(1e-30).ln()) as f32;
        if gap[k] > gap[best - 1] {
            best = k + 1;
        }
    }
    let _ = assign5;
    (best, gap)
}

/// 1-D 轮廓系数均值（k=1 → 0）。簇在排序序列上连续，用前缀和 O(m·k)。
fn silhouette_1d(ts: &[f32], k: usize) -> f32 {
    let m = ts.len();
    if k <= 1 || m < 2 {
        return 0.0;
    }
    let (assign, _) = kmeans1d(ts, k);
    // 簇 → 连续段
    let mut seg: Vec<(usize, usize)> = Vec::new(); // (l, r)
    let mut i = 0usize;
    while i < m {
        let a = assign[i];
        let mut j = i;
        while j < m && assign[j] == a {
            j += 1;
        }
        seg.push((i, j));
        i = j;
    }
    // 每簇前缀和（x 与 |x|）
    let mut pre: Vec<f64> = vec![0.0; m + 1];
    for i in 0..m {
        pre[i + 1] = pre[i] + ts[i] as f64;
    }
    let seg_at: Vec<usize> = {
        let mut v = vec![0usize; m];
        for (si, &(l, r)) in seg.iter().enumerate() {
            for p in l..r {
                v[p] = si;
            }
        }
        v
    };
    let mut sil_sum = 0.0f64;
    for i in 0..m {
        let si = seg_at[i];
        let (l, r) = seg[si];
        let x = ts[i] as f64;
        let n_same = (r - l) as f64;
        let mut a_i = 0.0f64;
        if n_same > 1.0 {
            // Σ_{j in [l,r)} |x - x_j|
            let left = x * (i - l) as f64 - (pre[i] - pre[l]);
            let right = (pre[r] - pre[i + 1]) - x * (r - i - 1) as f64;
            a_i = (left + right) / (n_same - 1.0);
        }
        // b_i = min over 其它簇的均值距离
        let mut b_i = f64::INFINITY;
        for (sj, &(l2, r2)) in seg.iter().enumerate() {
            if sj == si {
                continue;
            }
            let n2 = (r2 - l2) as f64;
            let mut sum = 0.0f64;
            // 该簇内点相对 x 的距离和（有序 → 二分分界）
            let cut = ts[l2..r2].partition_point(|&v| (v as f64) < x);
            let pos = l2 + cut;
            let left = x * cut as f64 - (pre[pos] - pre[l2]);
            let right = (pre[r2] - pre[pos]) - x * (r2 - pos) as f64;
            sum = left + right;
            let d = sum / n2;
            if d < b_i {
                b_i = d;
            }
        }
        if b_i.is_finite() {
            let mx = a_i.max(b_i);
            sil_sum += if mx > 0.0 { (b_i - a_i) / mx } else { 0.0 };
        }
    }
    (sil_sum / m as f64) as f32
}

/// 1-D DBSCAN（间隔 ≤ eps 合并，簇需 ≥ min_pts 笔）→ (簇数, 簇内总笔数)。
fn dbscan_runs(ts: &[f32], eps: f32, min_pts: usize) -> (usize, usize) {
    let m = ts.len();
    if m < min_pts {
        return (0, 0);
    }
    let mut n_clusters = 0usize;
    let mut in_cluster = 0usize;
    let mut run_len = 1usize;
    for i in 1..m {
        if ts[i] - ts[i - 1] <= eps {
            run_len += 1;
        } else {
            if run_len >= min_pts {
                n_clusters += 1;
                in_cluster += run_len;
            }
            run_len = 1;
        }
    }
    if run_len >= min_pts {
        n_clusters += 1;
        in_cluster += run_len;
    }
    (n_clusters, in_cluster)
}

/// K-Means 输入抽样：m > M_CAP 时等间隔抽样至 M_CAP（确定性）。
fn sample_for_km(ts: &[f32]) -> Vec<f32> {
    let m = ts.len();
    if m <= M_CAP {
        return ts.to_vec();
    }
    (0..M_CAP)
        .map(|i| ts[(i as f64 * m as f64 / M_CAP as f64) as usize])
        .collect()
}

// ============================================================
// S7：每股三版本的狭义时序聚类（7 指标）
// ============================================================

struct S7Out {
    km_best_k: f32,
    km_gap: f32,
    km_sil: f32,
    km2_ratio: f32,
    km2_frac: f32,
    db_n: f32,
    db_frac: f32,
}

fn s7_of(ts: &[f32]) -> S7Out {
    let m = ts.len();
    let mut out = S7Out {
        km_best_k: nan(),
        km_gap: nan(),
        km_sil: nan(),
        km2_ratio: nan(),
        km2_frac: nan(),
        db_n: nan(),
        db_frac: nan(),
    };
    if m < MIN_M {
        return out;
    }
    // DBSCAN 用全序列
    let (ncl, incl) = dbscan_runs(ts, EPS_DB, 2);
    out.db_n = ncl as f32;
    out.db_frac = incl as f32 / m as f32;
    // K-Means 用抽样序列
    let s = sample_for_km(ts);
    let sm = s.len();
    if sm < 2 {
        return out;
    }
    let (best_k, gap) = gap_statistic(&s);
    out.km_best_k = best_k as f32;
    out.km_gap = gap[best_k - 1];
    out.km_sil = silhouette_1d(&s, best_k);
    // k=2 分离度
    let (a2, _) = kmeans1d(&s, 2);
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut q1 = 0.0f64;
    let mut q2 = 0.0f64;
    for i in 0..sm {
        if a2[i] == 0 {
            c1 += 1;
            q1 += s[i] as f64;
        } else {
            c2 += 1;
            q2 += s[i] as f64;
        }
    }
    let (mut m1, mut m2) = (0.0, 0.0);
    if c1 > 0 {
        m1 = q1 / c1 as f64;
    }
    if c2 > 0 {
        m2 = q2 / c2 as f64;
    }
    for i in 0..sm {
        if a2[i] == 0 {
            s1 += (s[i] as f64 - m1).powi(2);
        } else {
            s2 += (s[i] as f64 - m2).powi(2);
        }
    }
    let sd1 = if c1 > 0 { (s1 / c1 as f64).sqrt() } else { 0.0 };
    let sd2 = if c2 > 0 { (s2 / c2 as f64).sqrt() } else { 0.0 };
    let denom = sd1 + sd2;
    out.km2_ratio = if denom > 0.0 {
        ((m2 - m1).abs() / denom) as f32
    } else {
        nan()
    };
    out.km2_frac = c1.max(c2) as f32 / sm as f32;
    out
}

// ============================================================
// S5：方向结构（both 版）
// ============================================================

struct S5Out {
    dir_persist: f32,
    cross_pairs: f32,
}

fn s5_of(ts: &[f32], dirs: &[i8]) -> S5Out {
    let m = ts.len();
    let mut out = S5Out {
        dir_persist: nan(),
        cross_pairs: nan(),
    };
    if m < MIN_M_S5 {
        return out;
    }
    let mut same = 0usize;
    let mut opp = 0usize;
    let mut near = 0usize;
    for i in 1..m {
        let tau = ts[i] - ts[i - 1];
        if tau <= EPS5 {
            near += 1;
            if dirs[i] == dirs[i - 1] {
                same += 1;
            } else {
                opp += 1;
            }
        }
    }
    out.cross_pairs = opp as f32 / (m - 1) as f32;
    out.dir_persist = if near > 0 {
        same as f32 / near as f32
    } else {
        nan()
    };
    out
}

// ============================================================
// 每股状态与 per-stock 计算
// ============================================================

struct PerStock {
    code: String,
    t_all: Vec<f32>, // 全部成交（排序，session 秒）
    t: [Vec<f32>; 3], // buy / sell / both 迫切交易时间
    mass: [Vec<f32>; 3],
    dirs: [Vec<i8>; 3],
}

/// 三版本各自的基础量：m, bucket_ent, cv_iei, gap_med + 全流对照。
struct StockBase {
    m: [usize; 3],
    ent: [f32; 3],
    cv: [f32; 3],
    gap_med: [f32; 3],
    cv_all: f32,
    b_all: f32,
    ent_all: f32,
    dens_all: f32,
    s7: [S7Out; 3],
    s5: S5Out,
}

fn per_stock_base(ps: &PerStock) -> StockBase {
    let mut base = StockBase {
        m: [0; 3],
        ent: [nan(); 3],
        cv: [nan(); 3],
        gap_med: [nan(); 3],
        cv_all: nan(),
        b_all: nan(),
        ent_all: nan(),
        dens_all: nan(),
        s7: [
            S7Out {
                km_best_k: nan(),
                km_gap: nan(),
                km_sil: nan(),
                km2_ratio: nan(),
                km2_frac: nan(),
                db_n: nan(),
                db_frac: nan(),
            },
            S7Out {
                km_best_k: nan(),
                km_gap: nan(),
                km_sil: nan(),
                km2_ratio: nan(),
                km2_frac: nan(),
                db_n: nan(),
                db_frac: nan(),
            },
            S7Out {
                km_best_k: nan(),
                km_gap: nan(),
                km_sil: nan(),
                km2_ratio: nan(),
                km2_frac: nan(),
                db_n: nan(),
                db_frac: nan(),
            },
        ],
        s5: S5Out {
            dir_persist: nan(),
            cross_pairs: nan(),
        },
    };
    for v in 0..3 {
        base.m[v] = ps.t[v].len();
        if base.m[v] >= MIN_M {
            base.ent[v] = bucket_ent(&ps.t[v]).unwrap_or(nan());
            base.cv[v] = cv_of(&ps.t[v]).unwrap_or(nan());
            base.gap_med[v] = median_gap(&ps.t[v]).unwrap_or(nan());
            base.s7[v] = s7_of(&ps.t[v]);
        }
    }
    // 全流对照
    if ps.t_all.len() >= MIN_M {
        base.cv_all = cv_of(&ps.t_all).unwrap_or(nan());
        base.b_all = burst_b_of(base.cv_all.into()).unwrap_or(nan());
        base.ent_all = bucket_ent(&ps.t_all).unwrap_or(nan());
        base.dens_all = dens300(&ps.t_all).unwrap_or(nan());
    }
    base.s5 = s5_of(&ps.t[2], &ps.dirs[2]);
    base
}

/// per-stock 因子数组（S4/S8 位留 NaN，截面步骤回填）。
fn per_stock_factors(ps: &PerStock, b: &StockBase) -> [f32; N_FACTORS] {
    let mut f = [nan(); N_FACTORS];
    let mut idx = 0usize;
    // S1
    for v in 0..3 {
        f[idx] = b.m[v] as f32;
        idx += 1;
    }
    // S2
    for v in 0..3 {
        f[idx] = b.ent[v];
        idx += 1;
        f[idx] = b.cv[v];
        idx += 1;
        f[idx] = b.gap_med[v];
        idx += 1;
    }
    // S3
    for v in 0..3 {
        let cv = if b.cv[v].is_nan() { None } else { Some(b.cv[v]) };
        let ent = if b.ent[v].is_nan() {
            None
        } else {
            Some(b.ent[v])
        };
        let dens = if b.m[v] >= MIN_M {
            dens300(&ps.t[v])
        } else {
            None
        };
        let b_urg = burst_b_of(cv);
        f[idx] = x_of(cv, Some(b.cv_all).filter(|x| !x.is_nan()));
        idx += 1;
        f[idx] = x_of(b_urg, Some(b.b_all).filter(|x| !x.is_nan()));
        idx += 1;
        f[idx] = x_of(ent, Some(b.ent_all).filter(|x| !x.is_nan()));
        idx += 1;
        f[idx] = x_of(dens, Some(b.dens_all).filter(|x| !x.is_nan()));
        idx += 1;
    }
    // S4 占位（截面回填）
    idx += 9;
    // S5
    f[idx] = b.s5.dir_persist;
    idx += 1;
    f[idx] = b.s5.cross_pairs;
    idx += 1;
    // gap_ratio_bs / cv_diff_bs（买/卖两版均 m>=5）
    // 注：gap_ratio_bs 用均值间隔比（设计原为 med 比；实测 ~50% 股票迫切交易
    // 间隔中位数为 0——同微秒连拍，med 比只剩 47% 覆盖、过不了覆盖硬闸门，
    // 故按设计"阈值按覆盖率校准"口径改为 mean 比，语义不变：买卖两侧间隔不对称）
    f[idx] = ratio_opt(
        mean_gap(&ps.t[0]),
        mean_gap(&ps.t[1]),
        b.m[0] >= MIN_M && b.m[1] >= MIN_M,
    );
    idx += 1;
    f[idx] = diff_opt(b.cv[0], b.cv[1], b.m[0] >= MIN_M && b.m[1] >= MIN_M);
    idx += 1;
    // S6
    for v in 0..3 {
        let m = b.m[v];
        let t = &ps.t[v];
        let mass = &ps.mass[v];
        if m >= MIN_M {
            let mut o = 0usize;
            let mut l = 0usize;
            let mut lm = 0.0f32;
            let mut tm = 0.0f32;
            for i in 0..m {
                if t[i] < 1800.0 {
                    o += 1;
                }
                if t[i] >= 12600.0 {
                    l += 1;
                    lm += mass[i];
                }
                tm += mass[i];
            }
            f[idx] = o as f32 / m as f32;
            f[idx + 1] = l as f32 / m as f32;
            f[idx + 2] = if tm > 0.0 { lm / tm } else { nan() };
        }
        idx += 3;
    }
    // S7
    for v in 0..3 {
        f[idx] = b.s7[v].km_best_k;
        f[idx + 1] = b.s7[v].km_gap;
        f[idx + 2] = b.s7[v].km_sil;
        f[idx + 3] = b.s7[v].km2_ratio;
        f[idx + 4] = b.s7[v].km2_frac;
        f[idx + 5] = b.s7[v].db_n;
        f[idx + 6] = b.s7[v].db_frac;
        idx += 7;
    }
    // S8 占位（截面回填）
    // idx += 22;
    f
}

fn ratio_opt(a: Option<f32>, b: Option<f32>, valid: bool) -> f32 {
    match (a, b) {
        (Some(x), Some(y)) if valid && y > 0.0 => x / y,
        _ => nan(),
    }
}

fn diff_opt(a: f32, b: f32, valid: bool) -> f32 {
    if valid && !a.is_nan() && !b.is_nan() {
        a - b
    } else {
        nan()
    }
}

// ============================================================
// S4：截面分层中性化
// ============================================================

/// 对某版本：按 m 十分位分层，层内 z-score（层内股数 < 20 → NaN）。
fn csz_fill(
    codes: &[String],
    m: &[usize],
    metric: &[f32],
    target: &mut [f32],
) {
    let n = codes.len();
    let mut order: Vec<usize> = (0..n).filter(|&i| m[i] >= MIN_M).collect();
    order.sort_by_key(|&i| m[i]);
    let nv = order.len();
    if nv == 0 {
        return;
    }
    // 分层
    let mut layer = vec![0usize; n];
    for (rank, &i) in order.iter().enumerate() {
        layer[i] = ((rank as f64 * 10.0 / nv as f64) as usize).min(9);
    }
    for l in 0..10 {
        let members: Vec<usize> = (0..n).filter(|&i| m[i] >= MIN_M && layer[i] == l).collect();
        if members.len() < 20 {
            continue;
        }
        let vals: Vec<f64> = members
            .iter()
            .filter_map(|&i| {
                let v = metric[i];
                if v.is_nan() { None } else { Some(v as f64) }
            })
            .collect();
        let nv2 = vals.len();
        if nv2 < 20 {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / nv2 as f64;
        let var = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / nv2 as f64;
        let std = var.sqrt();
        if std < 1e-9 {
            continue;
        }
        for &i in &members {
            let v = metric[i];
            if !v.is_nan() {
                target[i] = ((v as f64 - mean) / std) as f32;
            }
        }
    }
}

// ============================================================
// S8：截面狭义聚类（K-Means ×3 特征集 + GMM + 最近邻）
// ============================================================

type F4 = [f32; 4];

fn zscore4(feats: &mut [F4]) {
    let n = feats.len();
    if n == 0 {
        return;
    }
    for d in 0..4 {
        let valid: Vec<f32> = feats.iter().map(|f| f[d]).filter(|v| !v.is_nan()).collect();
        let nv = valid.len();
        if nv < 2 {
            for f in feats.iter_mut() {
                f[d] = 0.0;
            }
            continue;
        }
        let mean = valid.iter().map(|&v| v as f64).sum::<f64>() / nv as f64;
        let var = valid
            .iter()
            .map(|&v| (v as f64 - mean) * (v as f64 - mean))
            .sum::<f64>()
            / nv as f64;
        let std = var.sqrt();
        for f in feats.iter_mut() {
            if f[d].is_nan() {
                continue;
            }
            f[d] = if std < 1e-9 {
                0.0
            } else {
                ((f[d] as f64 - mean) / std) as f32
            };
        }
    }
}

fn dist4(a: &F4, b: &F4) -> f32 {
    let mut s = 0.0f32;
    for d in 0..4 {
        let t = a[d] - b[d];
        s += t * t;
    }
    s.sqrt()
}

/// 确定性 4-D K-Means（等距取点初始化，Lloyd）。
fn kmeans4d(feats: &[F4], k: usize) -> (Vec<u32>, Vec<F4>) {
    let n = feats.len();
    let kk = k.min(n).max(1);
    let mut cents: Vec<F4> = (0..kk)
        .map(|i| feats[(i * n / kk).min(n - 1)])
        .collect();
    let mut assign = vec![0u32; n];
    let mut newc = vec![[0f32; 4]; kk];
    let mut cnt = vec![0u32; kk];
    for _ in 0..KM4_ITERS {
        for i in 0..n {
            let mut best = 0usize;
            let mut bd = dist4(&feats[i], &cents[0]);
            for c in 1..kk {
                let d = dist4(&feats[i], &cents[c]);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            assign[i] = best as u32;
        }
        newc.iter_mut().for_each(|v| *v = [0.0; 4]);
        cnt.iter_mut().for_each(|v| *v = 0);
        for i in 0..n {
            let a = assign[i] as usize;
            for d in 0..4 {
                newc[a][d] += feats[i][d];
            }
            cnt[a] += 1;
        }
        let mut moved = 0.0f32;
        for c in 0..kk {
            if cnt[c] > 0 {
                let mut nc = [0f32; 4];
                for d in 0..4 {
                    nc[d] = newc[c][d] / cnt[c] as f32;
                }
                moved = moved.max(dist4(&nc, &cents[c]));
                cents[c] = nc;
            }
        }
        if moved < 1e-6 {
            break;
        }
    }
    (assign, cents)
}

/// 截面几何量：dist_to_own, sil, nn1/nn10/nn20, density, margin, local_rank, nn_same_other。
struct CsGeom {
    dist: Vec<f32>,
    sil: Vec<f32>,
    nn1: Vec<f32>,
    nn10: Vec<f32>,
    nn20: Vec<f32>,
    density: Vec<f32>,
    margin: Vec<f32>,
    local_rank: Vec<f32>,
    nn_same_other: Vec<f32>,
    devs: [Vec<f32>; 4],
}

fn cs_geometry(feats: &[F4], k: usize, need_extra: bool) -> CsGeom {
    let n = feats.len();
    let (assign, cents) = kmeans4d(feats, k);
    let mut dist = vec![0f32; n];
    for i in 0..n {
        dist[i] = dist4(&feats[i], &cents[assign[i] as usize]);
    }
    // 成对距离矩阵（n² × 4）
    let dists: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut row = vec![0f32; n];
            for j in 0..n {
                row[j] = dist4(&feats[i], &feats[j]);
            }
            row
        })
        .collect();
    // 轮廓系数
    let sil: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let a = assign[i] as usize;
            let mut s = 0.0f64;
            let mut ca = 0usize;
            for j in 0..n {
                if assign[j] as usize == a && j != i {
                    s += dists[i * n + j] as f64;
                    ca += 1;
                }
            }
            let a_i = if ca > 0 { s / ca as f64 } else { 0.0 };
            let mut b_i = f64::INFINITY;
            for c in 0..k {
                if c == a {
                    continue;
                }
                let mut s2 = 0.0f64;
                let mut cc = 0usize;
                for j in 0..n {
                    if assign[j] as usize == c {
                        s2 += dists[i * n + j] as f64;
                        cc += 1;
                    }
                }
                if cc > 0 {
                    let d = s2 / cc as f64;
                    if d < b_i {
                        b_i = d;
                    }
                }
            }
            if b_i.is_finite() {
                let mx = a_i.max(b_i);
                (if mx > 0.0 { (b_i - a_i) / mx } else { 0.0 }) as f32
            } else {
                0.0
            }
        })
        .collect();
    // 最近邻（按 (距离, 下标) 排序，确定性）
    let neighbors: Vec<(Vec<f32>, Vec<usize>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            idx.sort_unstable_by(|&x, &y| {
                dists[i * n + x]
                    .total_cmp(&dists[i * n + y])
                    .then_with(|| x.cmp(&y))
            });
            let ds: Vec<f32> = idx.iter().map(|&j| dists[i * n + j]).collect();
            (ds, idx)
        })
        .collect();
    let mut nn1 = vec![0f32; n];
    let mut nn10 = vec![0f32; n];
    let mut nn20 = vec![0f32; n];
    for i in 0..n {
        let (ds, _) = &neighbors[i];
        nn1[i] = ds[0];
        nn10[i] = ds[..10.min(ds.len())].iter().sum::<f32>() / 10.min(ds.len()) as f32;
        nn20[i] = ds[..20.min(ds.len())].iter().sum::<f32>() / 20.min(ds.len()) as f32;
    }
    let mut nn10_sorted: Vec<f32> = nn10.clone();
    nn10_sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let r = median_of(&nn10_sorted);
    let density: Vec<f32> = (0..n)
        .map(|i| {
            let mut c = 0usize;
            for j in 0..n {
                if dists[i * n + j] <= r {
                    c += 1;
                }
            }
            c as f32
        })
        .collect();
    // margin：到次近簇质心距离 − 到本簇质心
    let margin: Vec<f32> = (0..n)
        .map(|i| {
            let a = assign[i] as usize;
            let mut bd = f32::INFINITY;
            for c in 0..k {
                if c != a {
                    let d = dist4(&feats[i], &cents[c]);
                    if d < bd {
                        bd = d;
                    }
                }
            }
            if bd.is_finite() {
                bd - dist[i]
            } else {
                nan()
            }
        })
        .collect();
    // local_rank：本簇内 dist 百分位
    let local_rank: Vec<f32> = (0..n)
        .map(|i| {
            let a = assign[i] as usize;
            let mut below = 0usize;
            let mut cnt = 0usize;
            for j in 0..n {
                if assign[j] as usize == a {
                    cnt += 1;
                    if dist[j] < dist[i] {
                        below += 1;
                    }
                }
            }
            if cnt <= 1 {
                0.0
            } else {
                below as f32 / (cnt - 1) as f32
            }
        })
        .collect();
    // nn_same_other：10 近同簇均距 − 10 近异簇均距
    let nn_same_other: Vec<f32> = (0..n)
        .map(|i| {
            let a = assign[i] as usize;
            let (ds, idx) = &neighbors[i];
            let mut same: Vec<f32> = Vec::new();
            let mut other: Vec<f32> = Vec::new();
            for (p, &j) in idx.iter().enumerate() {
                if assign[j] as usize == a {
                    if same.len() < 10 {
                        same.push(ds[p]);
                    }
                } else if other.len() < 10 {
                    other.push(ds[p]);
                }
                if same.len() >= 10 && other.len() >= 10 {
                    break;
                }
            }
            if same.is_empty() || other.is_empty() {
                nan()
            } else {
                let ms = same.iter().sum::<f32>() / same.len() as f32;
                let mo = other.iter().sum::<f32>() / other.len() as f32;
                ms - mo
            }
        })
        .collect();
    // 分量偏离
    let mut devs: [Vec<f32>; 4] = [vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    for i in 0..n {
        for d in 0..4 {
            devs[d][i] = feats[i][d] - cents[assign[i] as usize][d];
        }
    }
    let _ = need_extra;
    CsGeom {
        dist,
        sil,
        nn1,
        nn10,
        nn20,
        density,
        margin,
        local_rank,
        nn_same_other,
        devs,
    }
}

/// GMM（对角协方差，K-Means 初始化，EM 确定性）→ (maxp, ent, margin)。
fn gmm_soft(feats: &[F4], k: usize, init_cents: &[F4]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = feats.len();
    let kk = k.min(n).max(1);
    let mut means: Vec<F4> = init_cents.to_vec();
    let mut covs: Vec<F4> = vec![[1.0; 4]; kk];
    let mut pi = vec![1.0 / kk as f32; kk];
    // 全局方差初始化
    for d in 0..4 {
        let mean = feats.iter().map(|f| f[d] as f64).sum::<f64>() / n as f64;
        let var = feats
            .iter()
            .map(|f| (f[d] as f64 - mean) * (f[d] as f64 - mean))
            .sum::<f64>()
            / n as f64;
        for c in 0..kk {
            covs[c][d] = var.max(1e-6) as f32;
        }
    }
    let mut logl_prev = f64::NEG_INFINITY;
    let mut gamma = vec![vec![0f32; kk]; n];
    for _ in 0..GMM_ITERS {
        // E-step
        let mut logl = 0.0f64;
        for i in 0..n {
            let mut logs = [0f64; GMM_K];
            let mut mx = f64::NEG_INFINITY;
            for c in 0..kk {
                let mut s = pi[c].max(1e-30).ln() as f64;
                for d in 0..4 {
                    let diff = feats[i][d] as f64 - means[c][d] as f64;
                    s += -0.5 * diff * diff / covs[c][d] as f64
                        - 0.5 * (2.0 * std::f64::consts::PI * covs[c][d] as f64).ln();
                }
                logs[c] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f64;
            for c in 0..kk {
                let e = (logs[c] - mx).exp();
                gamma[i][c] = e as f32;
                sum += e;
            }
            for c in 0..kk {
                gamma[i][c] = (gamma[i][c] as f64 / sum) as f32;
            }
            logl += mx + sum.ln();
        }
        if (logl - logl_prev).abs() < 1e-6 * logl.abs().max(1.0) {
            logl_prev = logl;
            break;
        }
        logl_prev = logl;
        // M-step
        let mut nk = vec![0f64; kk];
        for i in 0..n {
            for c in 0..kk {
                nk[c] += gamma[i][c] as f64;
            }
        }
        for c in 0..kk {
            if nk[c] < 1e-30 {
                continue;
            }
            let mut newm = [0f64; 4];
            for i in 0..n {
                for d in 0..4 {
                    newm[d] += gamma[i][c] as f64 * feats[i][d] as f64;
                }
            }
            for d in 0..4 {
                means[c][d] = (newm[d] / nk[c]) as f32;
            }
            let mut newv = [0f64; 4];
            for i in 0..n {
                for d in 0..4 {
                    let diff = feats[i][d] as f64 - means[c][d] as f64;
                    newv[d] += gamma[i][c] as f64 * diff * diff;
                }
            }
            for d in 0..4 {
                covs[c][d] = (newv[d] / nk[c]).max(1e-6) as f32;
            }
            pi[c] = (nk[c] / n as f64) as f32;
        }
    }
    let mut maxp = vec![0f32; n];
    let mut ent = vec![0f32; n];
    let mut margin = vec![0f32; n];
    for i in 0..n {
        let mut g: Vec<f32> = gamma[i].clone();
        g.sort_unstable_by(|a, b| b.total_cmp(a));
        maxp[i] = g[0];
        margin[i] = g[0] - if kk > 1 { g[1] } else { 0.0 };
        let mut e = 0.0f64;
        for c in 0..kk {
            let p = gamma[i][c] as f64;
            if p > 0.0 {
                e -= p * p.ln();
            }
        }
        ent[i] = e as f32;
    }
    (maxp, ent, margin)
}

// ============================================================
// 核心：读全市场 → per-stock → 截面 → (codes, vals)
// ============================================================

pub fn compute_urgency_ext_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date);
    // ① rayon 并行读全市场（过滤撤单）
    let per_trades: Vec<(String, Vec<TradeRecord>)> = codes
        .par_iter()
        .filter_map(|code| {
            let mut trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            trades.retain(|t| t.flag != 32);
            if trades.is_empty() {
                return None;
            }
            trades.sort_unstable_by_key(|t| t.time_us);
            Some((code.clone(), trades))
        })
        .collect();
    if per_trades.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    // ② 全市场分位（与初版相同口径）
    let total: usize = per_trades.iter().map(|(_, t)| t.len()).sum();
    let mut ratio_all: Vec<f32> = Vec::with_capacity(total);
    for (_, trades) in &per_trades {
        for t in trades {
            let sum = (t.ask_order + t.bid_order) as f64;
            let r = if sum > 0.0 {
                ((t.ask_order - t.bid_order) as f64 / sum) as f32
            } else {
                0.0
            };
            ratio_all.push(r);
        }
    }
    let n = ratio_all.len();
    let qidx = |q: f64| ((n as f64 * q) as usize).min(n - 1);
    let mut rs = ratio_all.clone();
    rs.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let q95 = rs[qidx(0.95)];
    let q5 = rs[qidx(0.05)];
    for v in rs.iter_mut() {
        *v = v.abs();
    }
    rs.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let q95_abs = rs[qidx(0.95)];
    drop(rs);
    drop(ratio_all);

    // ③ per-stock 组装 + 计算（保序）
    let stocks: Vec<PerStock> = per_trades
        .into_par_iter()
        .map(|(code, trades)| {
            let mut ps = PerStock {
                code,
                t_all: Vec::with_capacity(trades.len()),
                t: [Vec::new(), Vec::new(), Vec::new()],
                mass: [Vec::new(), Vec::new(), Vec::new()],
                dirs: [Vec::new(), Vec::new(), Vec::new()],
            };
            for t in &trades {
                let ts = session_seconds(date, t.time_us);
                ps.t_all.push(ts);
                let sum = (t.ask_order + t.bid_order) as f64;
                let r = if sum > 0.0 {
                    ((t.ask_order - t.bid_order) as f64 / sum) as f32
                } else {
                    0.0
                };
                let ab = r.abs();
                let d: i8 = if t.flag == 83 { 1 } else { -1 };
                if r < q5 {
                    ps.t[0].push(ts);
                    ps.mass[0].push(ab);
                    ps.dirs[0].push(d);
                }
                if r > q95 {
                    ps.t[1].push(ts);
                    ps.mass[1].push(ab);
                    ps.dirs[1].push(d);
                }
                if ab > q95_abs {
                    ps.t[2].push(ts);
                    ps.mass[2].push(ab);
                    ps.dirs[2].push(d);
                }
            }
            ps
        })
        .collect();

    // ④ per-stock 因子（S4/S8 位 NaN）
    let bases: Vec<StockBase> = stocks
        .par_iter()
        .map(|ps| per_stock_base(ps))
        .collect();
    let mut facs: Vec<[f32; N_FACTORS]> = stocks
        .iter()
        .zip(bases.iter())
        .map(|(ps, b)| per_stock_factors(ps, b))
        .collect();

    // ⑤ S4 截面中性化（回填 idx 24..33）
    for v in 0..3 {
        let m: Vec<usize> = bases.iter().map(|b| b.m[v]).collect();
        let cv: Vec<f32> = bases.iter().map(|b| b.cv[v]).collect();
        let ent: Vec<f32> = bases.iter().map(|b| b.ent[v]).collect();
        let gap: Vec<f32> = bases.iter().map(|b| b.gap_med[v]).collect();
        let mut csz_cv = vec![nan(); facs.len()];
        let mut csz_ent = vec![nan(); facs.len()];
        let mut csz_gap = vec![nan(); facs.len()];
        let codes_v: Vec<String> = stocks.iter().map(|s| s.code.clone()).collect();
        csz_fill(&codes_v, &m, &cv, &mut csz_cv);
        csz_fill(&codes_v, &m, &ent, &mut csz_ent);
        csz_fill(&codes_v, &m, &gap, &mut csz_gap);
        for i in 0..facs.len() {
            facs[i][24 + v * 3] = csz_cv[i];
            facs[i][24 + v * 3 + 1] = csz_ent[i];
            facs[i][24 + v * 3 + 2] = csz_gap[i];
        }
    }

    // ⑥ S8 截面狭义聚类（回填 idx 67..89）
    cs_fill_s8(&stocks, &bases, &mut facs);

    // ⑦ fan-out（保序）
    let mut vals = Vec::with_capacity(facs.len() * N_FACTORS);
    for f in &facs {
        vals.extend(f.iter().copied());
    }
    let codes_out: Vec<String> = stocks.iter().map(|s| s.code.clone()).collect();
    Ok((codes_out, vals))
}

/// S8 回填：特征集 → z → K-Means/GMM → 几何量。
fn cs_fill_s8(stocks: &[PerStock], bases: &[StockBase], facs: &mut [[f32; N_FACTORS]]) {
    let n = stocks.len();
    let s8_base = 67usize;
    // 有效集（both：m>=5 且 ent/cv/gap_med 均有效，避免 NaN 污染聚类）
    let both_ok = |i: usize| {
        bases[i].m[2] >= MIN_M
            && !bases[i].ent[2].is_nan()
            && !bases[i].cv[2].is_nan()
            && !bases[i].gap_med[2].is_nan()
    };
    let valid_both: Vec<usize> = (0..n).filter(|&i| both_ok(i)).collect();
    if valid_both.len() < CS_MIN_VALID {
        return;
    }
    // F_both
    let f_both: Vec<F4> = valid_both
        .iter()
        .map(|&i| {
            [
                bases[i].ent[2],
                bases[i].cv[2],
                bases[i].gap_med[2],
                (bases[i].m[2] as f32).ln_1p(),
            ]
        })
        .collect();
    let mut zb = f_both.clone();
    zscore4(&mut zb);
    let geom = cs_geometry(&zb, GMM_K, true);
    // GMM
    let (_, cents) = kmeans4d(&zb, GMM_K);
    let (gmm_maxp, gmm_ent, gmm_margin) = gmm_soft(&zb, GMM_K, &cents);
    for (k, &i) in valid_both.iter().enumerate() {
        let mut p = s8_base;
        facs[i][p] = geom.dist[k];
        p += 1;
        facs[i][p] = geom.sil[k];
        p += 1;
        facs[i][p] = geom.nn1[k];
        p += 1;
        facs[i][p] = geom.nn10[k];
        p += 1;
        facs[i][p] = geom.nn20[k];
        p += 1;
        facs[i][p] = geom.density[k];
        p += 1;
        facs[i][p] = geom.margin[k];
        p += 1;
        facs[i][p] = geom.local_rank[k];
        p += 1;
        facs[i][p] = geom.nn_same_other[k];
        p += 1;
        facs[i][p] = geom.devs[0][k];
        p += 1;
        facs[i][p] = geom.devs[1][k];
        p += 1;
        facs[i][p] = geom.devs[2][k];
        p += 1;
        facs[i][p] = geom.devs[3][k];
        p += 1;
    }
    // F_buy / F_sell（各跑一次，输出 dist/sil：buy → 80/82，sell → 81/83）
    for (v, off) in [(0usize, 80usize), (1usize, 81usize)] {
        let valid: Vec<usize> = (0..n)
            .filter(|&i| {
                bases[i].m[v] >= MIN_M
                    && !bases[i].ent[v].is_nan()
                    && !bases[i].cv[v].is_nan()
                    && !bases[i].gap_med[v].is_nan()
            })
            .collect();
        if valid.len() < CS_MIN_VALID {
            continue;
        }
        let fv: Vec<F4> = valid
            .iter()
            .map(|&i| {
                [
                    bases[i].ent[v],
                    bases[i].cv[v],
                    bases[i].gap_med[v],
                    (bases[i].m[v] as f32).ln_1p(),
                ]
            })
            .collect();
        let mut z = fv.clone();
        zscore4(&mut z);
        let g = cs_geometry(&z, GMM_K, false);
        for (k, &i) in valid.iter().enumerate() {
            facs[i][off] = g.dist[k];
            facs[i][off + 2] = g.sil[k];
        }
    }
    // 差值（84/85，买卖两侧均有效才计算）
    for i in 0..n {
        let bd = facs[i][80];
        let sd = facs[i][81];
        let bs = facs[i][82];
        let ss = facs[i][83];
        facs[i][84] = if bd.is_nan() || sd.is_nan() { nan() } else { bd - sd };
        facs[i][85] = if bs.is_nan() || ss.is_nan() { nan() } else { bs - ss };
    }
    // GMM 三个（86/87/88）
    for (k, &i) in valid_both.iter().enumerate() {
        facs[i][86] = gmm_maxp[k];
        facs[i][87] = gmm_ent[k];
        facs[i][88] = gmm_margin[k];
    }
}

// ============================================================
// 因子名（唯一真相源，与因子布局严格一致）
// ============================================================

pub fn ext_names() -> Vec<String> {
    let mut names: Vec<String> = Vec::with_capacity(N_FACTORS);
    for v in ["buy", "sell", "both"] {
        names.push(format!("urgency_v1_ext_cluster_{v}_n_urg"));
    }
    for v in ["buy", "sell", "both"] {
        for m in ["bucket_ent", "cv_iei", "gap_med"] {
            names.push(format!("urgency_v1_ext_cluster_{v}_{m}"));
        }
    }
    for v in ["buy", "sell", "both"] {
        for m in ["xcv_iei", "xburst_b", "xent", "xdens300"] {
            names.push(format!("urgency_v1_ext_cluster_{v}_{m}"));
        }
    }
    for v in ["buy", "sell", "both"] {
        for m in ["csz_cv", "csz_ent", "csz_gapmed"] {
            names.push(format!("urgency_v1_ext_cluster_{v}_{m}"));
        }
    }
    for m in ["dir_persist_5s", "cross_pairs_5s", "gap_ratio_bs", "cv_diff_bs"] {
        names.push(format!("urgency_v1_ext_cluster_both_{m}"));
    }
    for v in ["buy", "sell", "both"] {
        for m in ["share_open30", "share_late30", "mass_share_late30"] {
            names.push(format!("urgency_v1_ext_cluster_{v}_{m}"));
        }
    }
    for v in ["buy", "sell", "both"] {
        for m in [
            "km_best_k",
            "km_gap",
            "km_sil",
            "km2_ratio",
            "km2_frac",
            "db_n",
            "db_frac",
        ] {
            names.push(format!("urgency_v1_ext_cluster_{v}_{m}"));
        }
    }
    for m in [
        "cskm_dist",
        "cskm_sil",
        "cskm_nn1",
        "cskm_nn10",
        "cskm_nn20",
        "cskm_density",
        "cskm_margin",
        "cskm_local_rank",
        "cskm_nn_same_other",
        "cskm_ent_dev",
        "cskm_cv_dev",
        "cskm_gap_dev",
        "cskm_n_dev",
    ] {
        names.push(format!("urgency_v1_ext_cluster_both_{m}"));
    }
    names.push("urgency_v1_ext_cluster_buy_cskm_dist".to_string());
    names.push("urgency_v1_ext_cluster_sell_cskm_dist".to_string());
    names.push("urgency_v1_ext_cluster_buy_cskm_sil".to_string());
    names.push("urgency_v1_ext_cluster_sell_cskm_sil".to_string());
    names.push("urgency_v1_ext_cluster_cskm_dist_diff_bs".to_string());
    names.push("urgency_v1_ext_cluster_cskm_sil_diff_bs".to_string());
    for m in ["cskm_gmm_maxp", "cskm_gmm_ent", "cskm_gmm_margin"] {
        names.push(format!("urgency_v1_ext_cluster_both_{m}"));
    }
    names
}

// ============================================================
// PyO3 入口
// ============================================================

#[pyfunction]
pub fn py_urgency_ext(_py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_urgency_ext_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_urgency_ext_names() -> Vec<String> {
    ext_names()
}

/// 单日全市场墙钟计时（rayon 全局线程池），返回秒。
#[pyfunction]
pub fn py_bench_market(_py: Python<'_>, date: i64) -> PyResult<f64> {
    let t0 = std::time::Instant::now();
    let (codes, vals) = compute_urgency_ext_full(date)?;
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "[bench_market {date}] {:.3}s stocks={} cells={}",
        dt,
        codes.len(),
        vals.len()
    );
    Ok(dt)
}

/// 单股 per-stock 计算耗时（阈值已由全市场读出；只计时 per-stock 部分，单线程），返回秒。
#[pyfunction]
pub fn py_bench_stock(_py: Python<'_>, date: i64, code: String) -> PyResult<f64> {
    let mut trades = read_trade_fast_inner(&code, date, false, true, usize::MAX)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))?;
    trades.retain(|t| t.flag != 32);
    trades.sort_unstable_by_key(|t| t.time_us);
    // 全市场阈值（与核心相同口径，但只为拿到 q5/q95/q95_abs）
    let codes = list_codes(date);
    let per_trades: Vec<Vec<TradeRecord>> = codes
        .par_iter()
        .filter_map(|c| {
            let mut tr = read_trade_fast_inner(c, date, false, true, usize::MAX).ok()?;
            tr.retain(|t| t.flag != 32);
            if tr.is_empty() {
                None
            } else {
                Some(tr)
            }
        })
        .collect();
    let total: usize = per_trades.iter().map(|t| t.len()).sum();
    let mut ratio_all: Vec<f32> = Vec::with_capacity(total);
    for tr in &per_trades {
        for t in tr {
            let sum = (t.ask_order + t.bid_order) as f64;
            ratio_all.push(if sum > 0.0 {
                ((t.ask_order - t.bid_order) as f64 / sum) as f32
            } else {
                0.0
            });
        }
    }
    let n = ratio_all.len();
    let qidx = |q: f64| ((n as f64 * q) as usize).min(n - 1);
    let mut rs = ratio_all.clone();
    rs.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let q95 = rs[qidx(0.95)];
    let q5 = rs[qidx(0.05)];
    for v in rs.iter_mut() {
        *v = v.abs();
    }
    rs.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let q95_abs = rs[qidx(0.95)];

    let mut ps = PerStock {
        code: code.clone(),
        t_all: Vec::with_capacity(trades.len()),
        t: [Vec::new(), Vec::new(), Vec::new()],
        mass: [Vec::new(), Vec::new(), Vec::new()],
        dirs: [Vec::new(), Vec::new(), Vec::new()],
    };
    for t in &trades {
        let ts = session_seconds(date, t.time_us);
        ps.t_all.push(ts);
        let sum = (t.ask_order + t.bid_order) as f64;
        let r = if sum > 0.0 {
            ((t.ask_order - t.bid_order) as f64 / sum) as f32
        } else {
            0.0
        };
        let ab = r.abs();
        let d: i8 = if t.flag == 83 { 1 } else { -1 };
        if r < q5 {
            ps.t[0].push(ts);
            ps.mass[0].push(ab);
            ps.dirs[0].push(d);
        }
        if r > q95 {
            ps.t[1].push(ts);
            ps.mass[1].push(ab);
            ps.dirs[1].push(d);
        }
        if ab > q95_abs {
            ps.t[2].push(ts);
            ps.mass[2].push(ab);
            ps.dirs[2].push(d);
        }
    }
    let t0 = std::time::Instant::now();
    let b = per_stock_base(&ps);
    let _ = per_stock_factors(&ps, &b);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "[bench_stock {date} {code}] per-stock {:.6}s (m_buy={} m_sell={} m_both={})",
        dt, b.m[0], b.m[1], b.m[2]
    );
    Ok(dt)
}
