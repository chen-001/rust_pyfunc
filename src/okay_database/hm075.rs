//! hm75「南国路繁华」的 Rust 复现（旧框架 design_whatever_out.TickDataBase，逐股系列）。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm75_南国路繁华/hm75_okay.py
//!   run_factor_out 的实参：get_many=0 / read_full=0 / change_time=1
//!   → read_trade_data_single 返回「一整天逐笔」一张表，窗口由旧代码自己切。
//!   names_single 2248 个；names_save 77 个（其中 5 个带 `_fold`，展开成 unfold 名后
//!   由 run_base 末尾的 dwo.get_abs 生成，本文件不参与）。2248 个名字无重名。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=1)
//!    · with_retreat=1 → 读进来时**不丢** flag==32 的撤单行
//!    · change_time=1 → deal_data 把 exchtime 换成 Timestamp：
//!        to_timedelta(exchtime/1e6, unit='s') + Timestamp('1970-01-01 08:00:00')
//!    · 老框架用的列：date/exchtime/price/volume/turnover/flag/localtime/ask_order/bid_order
//!      （price/localtime 本系列一点没用上）
//! 2) df = dwo.adjust_afternoon(df)：按 exchtime 的钟点保留
//!        [09:30:00, 11:30:00] ∪ [13:00:00, 14:57:00]（两端都含，纳秒精度）
//!    下午那一段整体左移 90 分钟，然后 pd.concat([上午, 下午]).reset_index()
//!    行序 = 上午（文件序）在前、下午（文件序）在后
//! 3) df_retreat = df[df.flag==32]；df = df[df.flag!=32]
//!    df_retreat_other = df[ask_order ∈ df_retreat.ask_order ∪ bid_order ∈ df_retreat.bid_order]
//!    df_retreat = concat([df_retreat_other, df_retreat])
//!    retreat_ask = df_retreat[ask_order != 0].volume.sum()
//!    retreat_bid = df_retreat[bid_order != 0].volume.sum()
//! 4) 在非撤单 df 上按 ask_order / bid_order 分组（pandas groupby 默认按 key 升序）：
//!      ask_dura = (group.exchtime.last() - group.exchtime.first()).total_seconds()
//!      ask_num  = (num.last() - num.first() + 1) - group.volume.count()   # num = 0..n-1 位置号
//!      ask_dura = ask_dura[ask_dura > 0]；ask_num = ask_num[ask_num > 0]（各自独立过滤）
//!      half 档：x[x >= x.quantile(0.5)]（线性插值分位，两端都取）
//! 5) one(ask_dura, bid_dura, ask_num, bid_num) 产出 1082 个值：
//!      res1..res9 各 4 个（36）
//!      res10 = 4 块 × 261（ask / bid / ask-bid 差 / ask-bid 差的绝对值）
//!      res12 = retreat_hold_ask_ratio / retreat_hold_bid_ratio（2）
//!    full 档 + half 档各一次 → 2164
//! 6) res11：rp.analyze_long_orders(exchtime_ns, order, volume[, 0.5]) 各 6 个返回值，
//!    取前 3 个向量 + 后 3 个标量，拼成 12 列的 DataFrame，
//!    再过 dwo.get_features_factors(..., with_abs=False, with_corr=False) → 72，
//!    再拼上 12 个总比值 → 84。2164 + 84 = 2248。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、必须照抄的 pandas 语义
//! ════════════════════════════════════════════════════════════════════
//! · exchtime → 纳秒：pandas 走 conversion.pyx 的 cast_from_unit_vectorized（unit='s'→'ns'）：
//!       base = (int64)values[i];  frac = values[i] - base
//!       frac = np.round(frac, 9)              # rint(frac*1e9)/1e9，四舍六入五成双
//!       out[i] = (int64)(base*1e9) + (int64)(frac*1e9)
//!   不是简单的「微秒 × 1000」。实测四条对账数据（20250303/20250304/20180601/20231020）
//!   上本实现与 pd.to_timedelta 逐位相同（0 处不一致）。
//! · Timedelta.total_seconds() = _value / 1e9，_value 是整数纳秒差 → dura 逐位精确。
//!   dura 要做「> 0」「>= 分位点」的硬掩码、还要做 rank，差 1 ulp 就能翻面，所以这一条必须对。
//! · groupby().volume.sum()：本系列 volume 被 pandas 读成 int64，求和是整数精确和；
//!   这里按 f64 累加，值域远小于 2^53，逐位相同。
//! · groupby().turnover.sum()：float64 走 groupby.pyx 的 group_sum，**Kahan 补偿求和**，
//!   按行序累加。照抄。
//! · Series.sum()/mean()/std()/skew()/kurt()：pandas 走 numpy 的成对求和（PW_BLOCKSIZE=128、
//!   8 路展开），NaN 填 0 后求和、按有效观测数计数。照抄（hm073.rs 里那套已验证的实现）。
//! · Series.quantile：np.percentile(method='linear')，虚拟下标 (n-1)*q，numpy 的 _lerp 插值。
//! · Series.corr(Series) → nanops.nancorr → np.corrcoef（两遍法，Welford 的替代在这里差 1 ulp，
//!   但 res1 那四个值不做掩码也不做 rank，1 ulp 不影响 1e-4 验收）。
//! · rp.pandas_series_rank：average、升序、NaN 保持，并列判据 |a-b| < 1e-15（series_rank.rs）。
//! · rp.ols_residuals：带截距，ndarray 的 .dot() 算 X'X / X'y 再走 solve_linear_system3。
//!   残差要拿去做 rank，必须逐位对齐，不能自己写顺序求和。
//! · rp.corrwith(df, df.shift(1))：statistics/mod.rs 的 calculate_correlation_optimized
//!   （先求均值、再按 32 个一批分块累加）。
//! · rp.fast_correlation_matrix_v2_df(df, max_workers=1)：fast_correlation_v2.rs，
//!   均值走 Kahan，偏差和按 8 路展开顺序累加。
//! · rp.trend_2d(df, 0)：非有限值丢掉、位置用 1 基、样本 <2 或任一侧方差为 0 时给 0.0。
//! · dwo.get_features_factors(df, with_abs=False)：6 个统计量（mean/std/skew/kurt/autocorr1/
//!   trend）+ 上三角成对相关，统计量在前、相关块在后，且是「统计量优先」排列
//!   （先 18 个 mean，再 18 个 std，……）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、旧代码里的两处「照抄不误」
//! ════════════════════════════════════════════════════════════════════
//! · 第 176 行 `yisnum_resi_bid=pd.Series(rp.ols_residuals(bid_twins_num…, bid_twins_dura…))`
//!   与第 174 行 `yisdura_resi_bid` **参数完全相同**（旧代码的笔误）。
//!   所以 bid 侧的 yisnum_resi_bid 数值上等于 yisdura_resi_bid，只是挂了另一个 index。
//!   这里照抄（否则 bid_df 的两列会与旧代码不同）。
//! · groupby 把 ask_order==0 / bid_order==0 也当成一个正常分组参与，
//!   只有在算 retreat_ask/retreat_bid 时才用 `!= 0` 过滤。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、已知不可复现的部分（与实现无关，旧代码自身随机）
//! ════════════════════════════════════════════════════════════════════
//! rp.analyze_long_orders 内部用 HashMap 遍历订单（long_order_analysis.rs 第 129/154 行），
//! std 的 HashMap 每次 new 的哈希种子都不同，所以：
//!   · 三个比值向量的**顺序**每次进程都变 → res11 里 12 列的 autocorr1 / trend 跟着变；
//!   · top_ratio=0.5 时 filter_long_orders 按 time_span 降序稳定排序后取 ceil(n/2)，
//!     并列的那批订单谁被选中也随 HashMap 顺序变 → 部分列的 mean/std/skew/kurt 也会变。
//! 实测：同一份 hm75_okay.py 在同一个进程里连跑三次，2248 个值里有 34 个互相不同
//! （全部落在 res11 的 2164..2247 区间）。本文件按「订单号升序」这一确定性顺序复现，
//! 结果是这条随机分布里的一支；那 34 个下标都不在 names_save 里（已逐项核对）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、其它偏差
//! ════════════════════════════════════════════════════════════════════
//! · volume / turnover 用 Rust 的正确舍入解析（不是 pandas 的 precise_xstrtod）。
//!   这两列只参与求和、比值、均值、相关，不做「精确等于 0」「并列顺序」「阈值比较」，
//!   1 ulp 的解析差不可能翻过 1e-4 的验收线。exchtime 走整数前缀，本来就是精确值。
//! · 名字与 groupby 的输出顺序无关：names() 里的顺序完全按旧代码 names_list 的拼接顺序。

use std::collections::{BTreeMap, HashSet};
use std::fs;

use ndarray::{Array1, Array2};

// ============================================================================
// 逐笔 CSV 的列索引（0 基，与 /ssd_data/stock 下的表头一致）
// ============================================================================
const COL_EXCHTIME: usize = 4;
const COL_VOLUME: usize = 8;
const COL_TURNOVER: usize = 9;
const COL_FLAG: usize = 10;
const COL_ASK_ORDER: usize = 13;
const COL_BID_ORDER: usize = 14;
const MAX_COLS: usize = 16;

// ============================================================================
// 时间常量（纳秒）
// ============================================================================
const NS_PER_SEC: i64 = 1_000_000_000;
const DAY_NS: i64 = 86_400 * NS_PER_SEC;
const OFFSET_8H_NS: i64 = 8 * 3600 * NS_PER_SEC;
const AFTERNOON_SHIFT_NS: i64 = 90 * 60 * NS_PER_SEC;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * NS_PER_SEC;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * NS_PER_SEC;
const LD_1300: i64 = 13 * 3600 * NS_PER_SEC;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * NS_PER_SEC;

// ============================================================================
// 与 pandas 有关的数学工具（与 hm073.rs 同一套，已在那里对过 1588 个值）
// ============================================================================

#[inline]
fn is_ok(v: f64) -> bool {
    !v.is_nan()
}

/// numpy 的成对求和（PW_BLOCKSIZE=128、8 路展开）。
fn pairwise_sum(a: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &x in a.iter() {
            res += x;
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        while i + 8 <= n {
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

/// pandas nanops 里的 _zero_out_fperr：|v| < 1e-14 归零
#[inline]
fn zero_out_fperr(v: f64) -> f64 {
    if v.abs() < 1e-14 {
        0.0
    } else {
        v
    }
}

/// 把 NaN 换成 0 的一份拷贝 + 有效观测数（pandas 的 skipna 路径都是这么干的）
fn filled(v: &[f64]) -> (Vec<f64>, usize) {
    let mut out = Vec::with_capacity(v.len());
    let mut cnt = 0usize;
    for &x in v.iter() {
        if x.is_nan() {
            out.push(0.0);
        } else {
            out.push(x);
            cnt += 1;
        }
    }
    (out, cnt)
}

/// Series.mean()（skipna）
fn nan_mean(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt == 0 {
        return f64::NAN;
    }
    pairwise_sum(&f) / cnt as f64
}

/// Series.sum()（skipna，min_count=0：全 NaN 给 0）
fn nan_sum(v: &[f64]) -> f64 {
    let (f, _) = filled(v);
    pairwise_sum(&f)
}

/// Series.std(ddof=1)
fn nan_std(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt <= 1 {
        return f64::NAN;
    }
    let avg = pairwise_sum(&f) / cnt as f64;
    let mut sqr: Vec<f64> = Vec::with_capacity(f.len());
    for (i, &x) in f.iter().enumerate() {
        if v[i].is_nan() {
            sqr.push(0.0);
        } else {
            sqr.push((avg - x) * (avg - x));
        }
    }
    (pairwise_sum(&sqr) / (cnt - 1) as f64).sqrt()
}

/// Series.skew()（bias=False，修正 Fisher-Pearson）
fn nan_skew(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt < 3 {
        return f64::NAN;
    }
    let mean = pairwise_sum(&f) / cnt as f64;
    let mut a2: Vec<f64> = Vec::with_capacity(f.len());
    let mut a3: Vec<f64> = Vec::with_capacity(f.len());
    for (i, &x) in f.iter().enumerate() {
        if v[i].is_nan() {
            a2.push(0.0);
            a3.push(0.0);
        } else {
            let d = x - mean;
            a2.push(d * d);
            a3.push(d * d * d);
        }
    }
    let m2 = zero_out_fperr(pairwise_sum(&a2));
    let m3 = zero_out_fperr(pairwise_sum(&a3));
    if m2 == 0.0 {
        return 0.0;
    }
    let c = cnt as f64;
    (c * (c - 1.0).sqrt() / (c - 2.0)) * (m3 / m2.powf(1.5))
}

/// Series.kurt()（bias=False，超额峰度）
fn nan_kurt(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt < 4 {
        return f64::NAN;
    }
    let mean = pairwise_sum(&f) / cnt as f64;
    let mut a2: Vec<f64> = Vec::with_capacity(f.len());
    let mut a4: Vec<f64> = Vec::with_capacity(f.len());
    for (i, &x) in f.iter().enumerate() {
        if v[i].is_nan() {
            a2.push(0.0);
            a4.push(0.0);
        } else {
            let d = x - mean;
            let d2 = d * d;
            a2.push(d2);
            a4.push(d2 * d2);
        }
    }
    let m2 = pairwise_sum(&a2);
    let m4 = pairwise_sum(&a4);
    let c = cnt as f64;
    let adj = 3.0 * (c - 1.0) * (c - 1.0) / ((c - 2.0) * (c - 3.0));
    let numerator = zero_out_fperr(c * (c + 1.0) * (c - 1.0) * m4);
    let denominator = zero_out_fperr((c - 2.0) * (c - 3.0) * m2 * m2);
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator - adj
}

/// Series.quantile(q)：pandas 走 np.percentile(..., method='linear')。
/// 虚拟下标 (n-1)*q；插值照 numpy 的 _lerp（t>=0.5 走 b-(b-a)*(1-t)）。
fn nan_quantile(v: &[f64], q: f64) -> f64 {
    let mut ok: Vec<f64> = v.iter().copied().filter(|&x| is_ok(x)).collect();
    if ok.is_empty() {
        return f64::NAN;
    }
    ok.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = ok.len();
    let virt = (n as f64 - 1.0) * q;
    let prev_f = virt.floor();
    let mut pi = prev_f as i64;
    let mut ni = pi + 1;
    if virt >= n as f64 - 1.0 {
        pi = -1;
        ni = -1;
    }
    if virt < 0.0 {
        pi = 0;
        ni = 0;
    }
    let idx = |i: i64| -> usize {
        if i < 0 {
            (n as i64 + i) as usize
        } else {
            i as usize
        }
    };
    let pv = ok[idx(pi)];
    let nv = ok[idx(ni)];
    let gamma = virt - prev_f;
    let diff = nv - pv;
    let mut res = pv + diff * gamma;
    if gamma >= 0.5 {
        res = nv - diff * (1.0 - gamma);
    }
    res
}

/// pandas 的 Welford 在线 Pearson（Series.corr / DataFrame.corr 走这条）。
/// divpr == 0 时给 NaN。
fn pearson_welford(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs = 0.0f64;
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len().min(y.len()) {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let vx = x[i];
        let vy = y[i];
        nobs += 1.0;
        let dx = vx - mx;
        let dy = vy - my;
        mx += 1.0 / nobs * dx;
        my += 1.0 / nobs * dy;
        sxx += (vx - mx) * dx;
        syy += (vy - my) * dy;
        sxy += (vx - mx) * dy;
    }
    if nobs < 1.0 {
        return f64::NAN;
    }
    let divisor = (sxx * syy).sqrt();
    if divisor != 0.0 {
        sxy / divisor
    } else {
        f64::NAN
    }
}

/// rp.corrwith 那条：先求均值、再按 32 个一批分块累加（statistics/mod.rs）。
fn pearson_corrwith(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut count = 0usize;
    let (mut sum_x, mut sum_y) = (0.0f64, 0.0f64);
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        pairs.push((x[i], y[i]));
        sum_x += x[i];
        sum_y += y[i];
        count += 1;
    }
    if count < 2 {
        return f64::NAN;
    }
    let mean_x = sum_x / count as f64;
    let mean_y = sum_y / count as f64;
    let (mut cov, mut var_x, mut var_y) = (0.0f64, 0.0f64, 0.0f64);
    for chunk in pairs.chunks(32) {
        let (mut lc, mut lx, mut ly) = (0.0f64, 0.0f64, 0.0f64);
        for &(xi, yi) in chunk.iter() {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            lc += dx * dy;
            lx += dx * dx;
            ly += dy * dy;
        }
        cov += lc;
        var_x += lx;
        var_y += ly;
    }
    if var_x.abs() < f64::EPSILON || var_y.abs() < f64::EPSILON {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Kahan 补偿求和（fast_correlation_v2 里算列均值用的那个，也给 groupby.sum 用）
fn kahan_sum(v: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &x in v.iter() {
        let y = x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// rp.fast_correlation_matrix_v2_df(df, max_workers=1)[i,j]
/// （clean 分支 8 路展开顺序累加；含 NaN 的分支也照抄了，以防万一）
fn pearson_fast(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut px: Vec<f64> = Vec::with_capacity(n);
    let mut py: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() {
            px.push(x[i]);
        }
        if !y[i].is_nan() {
            py.push(y[i]);
        }
    }
    if px.len() == n && py.len() == n {
        if n < 2 {
            return f64::NAN;
        }
        let mi = kahan_sum(&px) / n as f64;
        let mj = kahan_sum(&py) / n as f64;
        let (mut cov, mut vi, mut vj) = (0.0f64, 0.0f64, 0.0f64);
        let chunks = n / 8;
        let rem = n % 8;
        for c in 0..chunks {
            let base = c * 8;
            for k in 0..8 {
                let idx = base + k;
                let a = px[idx] - mi;
                let b = py[idx] - mj;
                cov += a * b;
                vi += a * a;
                vj += b * b;
            }
        }
        for k in 0..rem {
            let idx = chunks * 8 + k;
            let a = px[idx] - mi;
            let b = py[idx] - mj;
            cov += a * b;
            vi += a * a;
            vj += b * b;
        }
        if vi.abs() < f64::EPSILON || vj.abs() < f64::EPSILON {
            return f64::NAN;
        }
        return cov / (vi.sqrt() * vj.sqrt());
    }
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            pairs.push((x[i], y[i]));
        }
    }
    if pairs.len() < 2 {
        return f64::NAN;
    }
    let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let mi = kahan_sum(&xs) / pairs.len() as f64;
    let mj = kahan_sum(&ys) / pairs.len() as f64;
    let (mut cov, mut vi, mut vj) = (0.0f64, 0.0f64, 0.0f64);
    for &(a, b) in pairs.iter() {
        let da = a - mi;
        let db = b - mj;
        cov += da * db;
        vi += da * da;
        vj += db * db;
    }
    if vi.abs() < f64::EPSILON || vj.abs() < f64::EPSILON {
        return f64::NAN;
    }
    cov / (vi.sqrt() * vj.sqrt())
}

/// rp.trend_2d 的一维版本：值与 1 基位置的 Pearson；非有限值丢掉；
/// 样本 < 2 或任一侧方差为 0 给 0.0（不是 NaN）。
fn trend_1d(v: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::new();
    let mut idx: Vec<f64> = Vec::new();
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            values.push(x);
            idx.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x = values.iter().sum::<f64>() / n as f64;
    let mean_y = idx.iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = idx[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// rp.pandas_series_rank(x)：average、升序、NaN 保持（相等判据 |a-b| < 1e-15）
fn rank_avg(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (i, &x) in v.iter().enumerate() {
        if !x.is_nan() {
            pairs.push((x, i));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < pairs.len() {
        let cur = pairs[i].0;
        let mut j = i;
        while j < pairs.len() && (pairs[j].0 - cur).abs() < 1e-15 {
            j += 1;
        }
        let avg = ((i + 1) + j) as f64 / 2.0;
        for k in i..j {
            out[pairs[k].1] = avg;
        }
        i = j;
    }
    out
}

/// rp.ols_residuals(X=[x], y)：带截距的一元回归残差（用 ndarray 的 .dot() + solve_linear_system3）。
fn ols_residual(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len());
    let mut out = vec![f64::NAN; n];
    let mut idx: Vec<usize> = Vec::new();
    for i in 0..n {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        idx.push(i);
    }
    if idx.is_empty() {
        return out;
    }
    let m = idx.len();
    let mut design = Array2::<f64>::ones((m, 2));
    let mut yv = Array1::<f64>::zeros(m);
    for (row, &i) in idx.iter().enumerate() {
        design[[row, 1]] = x[i];
        yv[row] = y[i];
    }
    let xt_x = design.t().dot(&design);
    let xt_y = design.t().dot(&yv);
    let coeff = crate::statistics::solve_linear_system3(&xt_x.view(), &xt_y.view());
    let y_pred = design.dot(&coeff);
    for (row, &i) in idx.iter().enumerate() {
        out[i] = y[i] - y_pred[row];
    }
    out
}

// ============================================================================
// dwo.get_features_factors(df, with_abs=False[, with_corr])
// ============================================================================

/// 6 个统计量，顺序即旧代码 res 列表（with_abs=False 时没有 *_abs 那一档）
const STATS6: [&str; 6] = ["mean", "std", "skew", "kurt", "autocorr1", "trend"];

fn stat6(kind: usize, v: &[f64]) -> f64 {
    match kind {
        0 => nan_mean(v),
        1 => nan_std(v),
        2 => nan_skew(v),
        3 => nan_kurt(v),
        4 => {
            if v.len() < 2 {
                f64::NAN
            } else {
                pearson_corrwith(&v[1..], &v[..v.len() - 1])
            }
        }
        5 => trend_1d(v),
        _ => unreachable!(),
    }
}

fn feat6_vals(cols: &[&[f64]], with_corr: bool) -> Vec<f64> {
    let k = cols.len();
    let mut out: Vec<f64> = Vec::new();
    for kind in 0..6 {
        for c in cols.iter() {
            out.push(stat6(kind, c));
        }
    }
    if with_corr {
        for i in 0..k {
            for j in (i + 1)..k {
                out.push(pearson_fast(cols[i], cols[j]));
            }
        }
    }
    out
}

fn feat6_names(cols: &[&str], with_corr: bool) -> Vec<String> {
    let k = cols.len();
    let mut out: Vec<String> = Vec::new();
    for s in STATS6.iter() {
        for c in cols.iter() {
            out.push(format!("{}_{}", c, s));
        }
    }
    if with_corr {
        for i in 0..k {
            for j in (i + 1)..k {
                out.push(format!("{}_corr_{}", cols[i], cols[j]));
            }
        }
    }
    out
}

// ============================================================================
// rp.analyze_long_orders（纯 Rust 版，见文件头「已知不可复现的部分」）
// ============================================================================

#[derive(Clone)]
struct LOrder {
    order_id: i64,
    first_idx: usize,
    last_idx: usize,
    total_volume: f64,
    time_span: i64,
    occurrence_count: usize,
    ratio: f64,
}

struct LInfo {
    first_idx: usize,
    last_idx: usize,
    total_volume: f64,
    times: Vec<i64>,
}

fn other_orders_volume(start: usize, end: usize, target: i64, order: &[i64], vol: &[f64]) -> f64 {
    let mut other = 0.0f64;
    for i in start..=end {
        if order[i] != target {
            other += vol[i];
        }
    }
    other
}

fn filter_long(orders: &[LOrder], top_ratio: f64, kind: u8) -> Vec<LOrder> {
    if orders.is_empty() || top_ratio >= 1.0 {
        return orders.to_vec();
    }
    if top_ratio <= 0.0 {
        return Vec::new();
    }
    let mut sorted = orders.to_vec();
    match kind {
        1 => sorted.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count)),
        _ => sorted.sort_by(|a, b| b.time_span.cmp(&a.time_span)),
    }
    let keep = (orders.len() as f64 * top_ratio).ceil() as usize;
    sorted.into_iter().take(keep).collect()
}

/// 返回 (时间漫长比值, 次数漫长比值, 两者都漫长比值, 三个总比值)
#[allow(clippy::type_complexity)]
fn analyze_long_orders(
    t: &[i64],
    order: &[i64],
    vol: &[f64],
    top_ratio: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64) {
    if t.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), 0.0, 0.0, 0.0);
    }
    let mut map: BTreeMap<i64, LInfo> = BTreeMap::new();
    for i in 0..order.len() {
        let e = map.entry(order[i]).or_insert(LInfo {
            first_idx: i,
            last_idx: i,
            total_volume: 0.0,
            times: Vec::new(),
        });
        e.last_idx = i;
        e.total_volume += vol[i];
        e.times.push(t[i]);
    }

    let mut time_long: Vec<LOrder> = Vec::new();
    let mut count_long: Vec<LOrder> = Vec::new();
    let mut both_long: Vec<LOrder> = Vec::new();

    for (oid, info) in map.iter() {
        if info.first_idx == info.last_idx {
            continue;
        }
        let is_time_long = {
            let mut uniq: Vec<i64> = info.times.clone();
            uniq.sort_unstable();
            uniq.dedup();
            uniq.len() > 1
        };
        let is_count_long = info.last_idx - info.first_idx > info.times.len() - 1;
        if !(is_time_long || is_count_long) {
            continue;
        }
        let other = other_orders_volume(info.first_idx, info.last_idx, *oid, order, vol);
        let ratio = if info.total_volume > 0.0 {
            other / info.total_volume
        } else {
            0.0
        };
        let mut mx = i64::MIN;
        let mut mn = i64::MAX;
        for &x in info.times.iter() {
            if x > mx {
                mx = x;
            }
            if x < mn {
                mn = x;
            }
        }
        let lo = LOrder {
            order_id: *oid,
            first_idx: info.first_idx,
            last_idx: info.last_idx,
            total_volume: info.total_volume,
            time_span: mx - mn,
            occurrence_count: info.times.len(),
            ratio,
        };
        if is_time_long {
            time_long.push(lo.clone());
        }
        if is_count_long {
            count_long.push(lo.clone());
        }
        if is_time_long && is_count_long {
            both_long.push(lo);
        }
    }

    let final_time = filter_long(&time_long, top_ratio, 0);
    let final_count = filter_long(&count_long, top_ratio, 1);
    let final_both = filter_long(&both_long, top_ratio, 2);

    let mut time_ratios: Vec<f64> = Vec::with_capacity(final_time.len());
    let mut count_ratios: Vec<f64> = Vec::with_capacity(final_count.len());
    let mut both_ratios: Vec<f64> = Vec::with_capacity(final_both.len());

    let mut t_other = 0.0f64;
    let mut t_ord = 0.0f64;
    for o in final_time.iter() {
        time_ratios.push(o.ratio);
        t_other += other_orders_volume(o.first_idx, o.last_idx, o.order_id, order, vol);
        t_ord += o.total_volume;
    }
    let mut c_other = 0.0f64;
    let mut c_ord = 0.0f64;
    for o in final_count.iter() {
        count_ratios.push(o.ratio);
        c_other += other_orders_volume(o.first_idx, o.last_idx, o.order_id, order, vol);
        c_ord += o.total_volume;
    }
    let mut b_other = 0.0f64;
    let mut b_ord = 0.0f64;
    for o in final_both.iter() {
        both_ratios.push(o.ratio);
        b_other += other_orders_volume(o.first_idx, o.last_idx, o.order_id, order, vol);
        b_ord += o.total_volume;
    }

    let time_total = if t_ord > 0.0 { t_other / t_ord } else { 0.0 };
    let count_total = if c_ord > 0.0 { c_other / c_ord } else { 0.0 };
    let both_total = if b_ord > 0.0 { b_other / b_ord } else { 0.0 };

    (
        time_ratios,
        count_ratios,
        both_ratios,
        time_total,
        count_total,
        both_total,
    )
}

// ============================================================================
// 逐笔 CSV 读取
// ============================================================================

/// exchtime 是「带 .0 的整数串」，整数前缀就是微秒数。
#[inline]
fn int_prefix(raw: &[u8]) -> i64 {
    let mut i = 0usize;
    let mut neg = false;
    if i < raw.len() && (raw[i] == b'-' || raw[i] == b'+') {
        neg = raw[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < raw.len() && raw[i].is_ascii_digit() {
        v = v * 10 + (raw[i] - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

fn parse_num(raw: &[u8]) -> f64 {
    if raw.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(raw)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// pandas conversion.pyx::cast_from_unit_vectorized 的标量版（unit='s' → 'ns'）：
///   base = (int64)x;  frac = x - base;  frac = np.round(frac, 9)
///   out = (int64)(base*1e9) + (int64)(frac*1e9)
fn sec_float_to_ns(x: f64) -> i64 {
    if x.is_nan() {
        return i64::MIN;
    }
    let base = x as i64;
    let frac = x - base as f64;
    let frac_r = (frac * 1e9).round_ties_even() / 1e9;
    base * 1_000_000_000 + (frac_r * 1e9) as i64
}

#[derive(Clone, Copy)]
struct Raw {
    t: i64,
    vol: f64,
    turn: f64,
    flag: f64,
    ask: i64,
    bid: i64,
}

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

/// 复刻到 adjust_afternoon 为止：读出 [09:30:00, 11:30:00] ∪ [13:00:00, 14:57:00] 的行，
/// 下午整体左移 90 分钟，行序 = 上午（文件序）在前、下午（文件序）在后。
fn read_day(code: &str, date: i64) -> Option<Vec<Raw>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in crate::data_paths::level2_roots() {
        let p = root
            .join(date.to_string())
            .join("transaction")
            .join(&filename);
        if let Ok(bytes) = fs::read(&p) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    let mut morning: Vec<Raw> = Vec::new();
    let mut afternoon: Vec<Raw> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 {
            continue;
        }
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if line.is_empty() {
            continue;
        }
        let mut f: [&[u8]; MAX_COLS] = [&[][..]; MAX_COLS];
        let nf = split_into(line, &mut f);
        if nf <= COL_BID_ORDER {
            continue;
        }
        if f[COL_EXCHTIME].is_empty() {
            continue;
        }
        let ex_us = int_prefix(f[COL_EXCHTIME]);
        let x = ex_us as f64 / 1_000_000.0;
        let ns = sec_float_to_ns(x);
        if ns == i64::MIN {
            continue;
        }
        let t = ns + OFFSET_8H_NS;
        let tod = t.rem_euclid(DAY_NS);
        let row = Raw {
            t,
            vol: parse_num(f[COL_VOLUME]),
            turn: parse_num(f[COL_TURNOVER]),
            flag: parse_num(f[COL_FLAG]),
            ask: int_prefix(f[COL_ASK_ORDER]),
            bid: int_prefix(f[COL_BID_ORDER]),
        };
        if (LD_0930..=LD_1130).contains(&tod) {
            morning.push(row);
        } else if (LD_1300..=LD_1457).contains(&tod) {
            afternoon.push(Raw {
                t: t - AFTERNOON_SHIFT_NS,
                ..row
            });
        }
    }

    morning.append(&mut afternoon);
    if morning.is_empty() {
        return None;
    }
    Some(morning)
}

// ============================================================================
// 分组与订单级序列
// ============================================================================

/// pandas groupby(key).volume.sum() / .turnover.sum()。
/// volume 是整数和（f64 累加在 2^53 内逐位精确），turnover 走 Kahan（groupby.pyx 的 group_sum）。
/// 返回按 key 升序排列的三条。
fn group_sums(key: &[i64], vol: &[f64], turn: &[f64]) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let n = key.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by_key(|&i| key[i]); // 稳定排序 → 组内保持行序
    let mut gk: Vec<i64> = Vec::new();
    let mut gv: Vec<f64> = Vec::new();
    let mut gt: Vec<f64> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let k = key[idx[start]];
        let mut end = start;
        let mut sv = 0.0f64;
        let mut s = 0.0f64;
        let mut comp = 0.0f64;
        while end < n && key[idx[end]] == k {
            let i = idx[end];
            sv += vol[i];
            let val = turn[i];
            if !val.is_nan() {
                let y = val - comp;
                let t = s + y;
                comp = (t - s) - y;
                s = t;
            }
            end += 1;
        }
        gk.push(k);
        gv.push(sv);
        gt.push(s);
        start = end;
    }
    (gk, gv, gt)
}

/// 组级 dura / num：
///   dura = (该组最后一条 exchtime − 第一条 exchtime) / 1e9
///   num  = (num.last − num.first + 1) − 该组 volume 的非 NaN 计数
/// 各自独立过滤 > 0，返回 (keys, dura, num_of_filtered_keys)
fn order_series(key: &[i64], t: &[i64], vol: &[f64]) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let n = key.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by_key(|&i| key[i]);
    let mut k_all: Vec<i64> = Vec::new();
    let mut d_all: Vec<f64> = Vec::new();
    let mut m_all: Vec<f64> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let k = key[idx[start]];
        let mut end = start;
        let mut cnt = 0usize;
        while end < n && key[idx[end]] == k {
            if !vol[idx[end]].is_nan() {
                cnt += 1;
            }
            end += 1;
        }
        let fi = idx[start];
        let li = idx[end - 1];
        let dura = (t[li] - t[fi]) as f64 / 1e9;
        let num = (li - fi + 1) as i64 - cnt as i64;
        k_all.push(k);
        d_all.push(dura);
        m_all.push(num as f64);
        start = end;
    }
    (k_all, d_all, m_all)
}

/// 按 sel 里的 key 顺序（升序）从 src 里挑出值；缺 key 给 NaN（理论上不会缺）。
fn pick_by_keys(src_k: &[i64], src_v: &[f64], sel: &[i64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(sel.len());
    for &k in sel {
        match src_k.binary_search(&k) {
            Ok(i) => out.push(src_v[i]),
            Err(_) => out.push(f64::NAN),
        }
    }
    out
}

// ============================================================================
// 一天的中间结果
// ============================================================================

struct Day {
    t: Vec<i64>,
    vol: Vec<f64>,
    ask: Vec<i64>,
    bid: Vec<i64>,
    // ask 侧 groupby 结果（全部组，按 key 升序）
    ask_gk: Vec<i64>,
    ask_volfrac: Vec<f64>, // groupby.volume.sum()/df.volume.sum()
    ask_vwap: Vec<f64>,    // 偏离度：(turnover.sum()/volfrac)/vwap_allday - 1
    bid_gk: Vec<i64>,
    bid_volfrac: Vec<f64>,
    bid_vwap: Vec<f64>,
    retreat_ask: f64,
    retreat_bid: f64,
}

/// 一天的四个订单级序列（已做 >0 过滤）
struct Series4 {
    ak: Vec<i64>,
    adv: Vec<f64>,
    ank: Vec<i64>,
    anv: Vec<f64>,
    bk: Vec<i64>,
    bdv: Vec<f64>,
    bnk: Vec<i64>,
    bnv: Vec<f64>,
}

/// 旧代码 one() 的复现，返回 1082 个值。
fn one(d: &Day, s: &Series4) -> Option<Vec<f64>> {
    let akset: HashSet<i64> = s.ak.iter().copied().collect();
    let anset: HashSet<i64> = s.ank.iter().copied().collect();
    let bkset: HashSet<i64> = s.bk.iter().copied().collect();
    let bnset: HashSet<i64> = s.bnk.iter().copied().collect();

    // ask_twins_dura = ask_dura[ask_dura.index.isin(ask_num.index)]
    let mut atk: Vec<i64> = Vec::new();
    let mut atdv: Vec<f64> = Vec::new();
    for i in 0..s.ak.len() {
        if anset.contains(&s.ak[i]) {
            atk.push(s.ak[i]);
            atdv.push(s.adv[i]);
        }
    }
    let mut atnk: Vec<i64> = Vec::new();
    let mut atnv: Vec<f64> = Vec::new();
    for i in 0..s.ank.len() {
        if akset.contains(&s.ank[i]) {
            atnk.push(s.ank[i]);
            atnv.push(s.anv[i]);
        }
    }
    let mut btk: Vec<i64> = Vec::new();
    let mut btdv: Vec<f64> = Vec::new();
    for i in 0..s.bk.len() {
        if bnset.contains(&s.bk[i]) {
            btk.push(s.bk[i]);
            btdv.push(s.bdv[i]);
        }
    }
    let mut btnk: Vec<i64> = Vec::new();
    let mut btnv: Vec<f64> = Vec::new();
    for i in 0..s.bnk.len() {
        if bkset.contains(&s.bnk[i]) {
            btnk.push(s.bnk[i]);
            btnv.push(s.bnv[i]);
        }
    }

    // hold_ask = df[df.ask_order.isin(ask_twins_dura.index)].volume.sum()
    let atkset: HashSet<i64> = atk.iter().copied().collect();
    let btset: HashSet<i64> = btk.iter().copied().collect();
    let mut hv: Vec<f64> = Vec::new();
    for i in 0..d.ask.len() {
        if atkset.contains(&d.ask[i]) {
            hv.push(d.vol[i]);
        }
    }
    let hold_ask = pairwise_sum(&hv);
    let mut hv2: Vec<f64> = Vec::new();
    for i in 0..d.bid.len() {
        if btset.contains(&d.bid[i]) {
            hv2.push(d.vol[i]);
        }
    }
    let hold_bid = pairwise_sum(&hv2);

    let retreat_hold_ask_ratio = if hold_ask > 0.0 {
        d.retreat_ask / hold_ask
    } else {
        f64::NAN
    };
    let retreat_hold_bid_ratio = if hold_bid > 0.0 {
        d.retreat_bid / hold_bid
    } else {
        f64::NAN
    };

    // 1. 相关性
    let ask_corr_dura_num = pearson_welford(&atdv, &atnv);
    let bid_corr_dura_num = pearson_welford(&btdv, &btnv);
    let corr_diff = ask_corr_dura_num - bid_corr_dura_num;

    // 2/3. 共同占比
    let ask_dura_count = atdv.len() as f64 / s.ak.len() as f64;
    let bid_dura_count = btdv.len() as f64 / s.bk.len() as f64;
    let dura_count_diff = ask_dura_count - bid_dura_count;
    let ask_num_count = atnv.len() as f64 / s.ank.len() as f64;
    let bid_num_count = btnv.len() as f64 / s.bnk.len() as f64;
    let num_count_diff = ask_num_count - bid_num_count;

    // 4/5/6：ask_volume / ask_vwap 的子集和
    let a_twins_vol = pick_by_keys(&d.ask_gk, &d.ask_volfrac, &atk);
    let a_dura_vol = pick_by_keys(&d.ask_gk, &d.ask_volfrac, &s.ak);
    let a_num_vol = pick_by_keys(&d.ask_gk, &d.ask_volfrac, &s.ank);
    let b_twins_vol = pick_by_keys(&d.bid_gk, &d.bid_volfrac, &btk);
    let b_dura_vol = pick_by_keys(&d.bid_gk, &d.bid_volfrac, &s.bk);
    let b_num_vol = pick_by_keys(&d.bid_gk, &d.bid_volfrac, &s.bnk);

    let ask_twins_volume_on_num_pos = nan_sum(&a_twins_vol) / nan_sum(&a_dura_vol);
    let bid_twins_volume_on_num_pos = nan_sum(&b_twins_vol) / nan_sum(&b_dura_vol);
    let twins_volume_diff = ask_twins_volume_on_num_pos - bid_twins_volume_on_num_pos;
    let ask_volume_on_dura_pos = nan_sum(&a_dura_vol);
    let bid_volume_on_dura_pos = nan_sum(&b_dura_vol);
    let volume_on_dura_diff = ask_volume_on_dura_pos - bid_volume_on_dura_pos;
    let ask_volume_on_num_pos = nan_sum(&a_num_vol);
    let bid_volume_on_num_pos = nan_sum(&b_num_vol);
    let volume_on_num_diff = ask_volume_on_num_pos - bid_volume_on_num_pos;

    // 7/8/9：vwap 偏离度的子集均值
    let a_twins_vwap = pick_by_keys(&d.ask_gk, &d.ask_vwap, &atk);
    let a_dura_vwap = pick_by_keys(&d.ask_gk, &d.ask_vwap, &s.ak);
    let a_num_vwap = pick_by_keys(&d.ask_gk, &d.ask_vwap, &s.ank);
    let b_twins_vwap = pick_by_keys(&d.bid_gk, &d.bid_vwap, &btk);
    let b_dura_vwap = pick_by_keys(&d.bid_gk, &d.bid_vwap, &s.bk);
    let b_num_vwap = pick_by_keys(&d.bid_gk, &d.bid_vwap, &s.bnk);

    let ask_twins_vwap_mean = nan_mean(&a_twins_vwap);
    let bid_twins_vwap_mean = nan_mean(&b_twins_vwap);
    let twins_vwap_mean_diff = ask_twins_vwap_mean - bid_twins_vwap_mean;
    let ask_vwap_on_dura_pos_mean = nan_mean(&a_dura_vwap);
    let bid_vwap_on_dura_pos_mean = nan_mean(&b_dura_vwap);
    let vwap_on_dura_mean_diff = ask_vwap_on_dura_pos_mean - bid_vwap_on_dura_pos_mean;
    let ask_vwap_on_num_pos_mean = nan_mean(&a_num_vwap);
    let bid_vwap_on_num_pos_mean = nan_mean(&b_num_vwap);
    let vwap_on_num_mean_diff = ask_vwap_on_num_pos_mean - bid_vwap_on_num_pos_mean;

    let mut out: Vec<f64> = Vec::with_capacity(1082);
    out.push(ask_corr_dura_num);
    out.push(bid_corr_dura_num);
    out.push(corr_diff);
    out.push(corr_diff.abs());
    out.push(ask_dura_count);
    out.push(bid_dura_count);
    out.push(dura_count_diff);
    out.push(dura_count_diff.abs());
    out.push(ask_num_count);
    out.push(bid_num_count);
    out.push(num_count_diff);
    out.push(num_count_diff.abs());
    out.push(ask_twins_volume_on_num_pos);
    out.push(bid_twins_volume_on_num_pos);
    out.push(twins_volume_diff);
    out.push(twins_volume_diff.abs());
    out.push(ask_volume_on_dura_pos);
    out.push(bid_volume_on_dura_pos);
    out.push(volume_on_dura_diff);
    out.push(volume_on_dura_diff.abs());
    out.push(ask_volume_on_num_pos);
    out.push(bid_volume_on_num_pos);
    out.push(volume_on_num_diff);
    out.push(volume_on_num_diff.abs());
    out.push(ask_twins_vwap_mean);
    out.push(bid_twins_vwap_mean);
    out.push(twins_vwap_mean_diff);
    out.push(twins_vwap_mean_diff.abs());
    out.push(ask_vwap_on_dura_pos_mean);
    out.push(bid_vwap_on_dura_pos_mean);
    out.push(vwap_on_dura_mean_diff);
    out.push(vwap_on_dura_mean_diff.abs());
    out.push(ask_vwap_on_num_pos_mean);
    out.push(bid_vwap_on_num_pos_mean);
    out.push(vwap_on_num_mean_diff);
    out.push(vwap_on_num_mean_diff.abs());

    // ---- 第 10 块：ask/bid 各自 18 列的 features ----
    // 旧代码第 176 行 `yisnum_resi_bid` 的实参与第 174 行 `yisdura_resi_bid` 完全相同
    // （笔误），所以 bid 侧那一列其实是 yisdura 的副本 → dup_yisnum=true 照抄。
    let ask_block = side_block(&atdv, &atnv, &a_twins_vol, &a_twins_vwap, false);
    let bid_block = side_block(&btdv, &btnv, &b_twins_vol, &b_twins_vwap, true);
    let nfeat = ask_block.len();
    for &v in ask_block.iter() {
        out.push(v);
    }
    for &v in bid_block.iter() {
        out.push(v);
    }
    for i in 0..nfeat {
        out.push(ask_block[i] - bid_block[i]);
    }
    for i in 0..nfeat {
        out.push((ask_block[i] - bid_block[i]).abs());
    }

    out.push(retreat_hold_ask_ratio);
    out.push(retreat_hold_bid_ratio);
    Some(out)
}

/// 一侧的 18 列 + dwo.get_features_factors(..., with_abs=False) → 261 个值。
/// `dup_yisnum` 复刻旧代码第 176 行的笔误（bid 侧的 yisnum_resi 与 yisdura_resi 同参数）。
fn side_block(
    dura: &[f64],
    num: &[f64],
    volume: &[f64],
    vwap: &[f64],
    dup_yisnum: bool,
) -> Vec<f64> {
    let m = dura.len();
    let dura_rank = rank_avg(dura);
    let num_rank = rank_avg(num);
    let rank_diff: Vec<f64> = (0..m).map(|i| dura_rank[i] - num_rank[i]).collect();
    let rank_abs: Vec<f64> = rank_diff.iter().map(|x| x.abs()).collect();

    let yisdura_resi = ols_residual(num, dura);
    let yisnum_resi = if dup_yisnum {
        ols_residual(num, dura)
    } else {
        ols_residual(dura, num)
    };
    let yisdura_rank = rank_avg(&yisdura_resi);
    let yisnum_rank = rank_avg(&yisnum_resi);

    let rd_yisdura: Vec<f64> = (0..m).map(|i| yisdura_rank[i] - dura_rank[i]).collect();
    let rd_yisdura_abs: Vec<f64> = rd_yisdura.iter().map(|x| x.abs()).collect();
    let rd_yisnum: Vec<f64> = (0..m).map(|i| yisnum_rank[i] - num_rank[i]).collect();
    let rd_yisnum_abs: Vec<f64> = rd_yisnum.iter().map(|x| x.abs()).collect();
    let rd_twins: Vec<f64> = (0..m).map(|i| yisdura_rank[i] - yisnum_rank[i]).collect();
    let rd_twins_abs: Vec<f64> = rd_twins.iter().map(|x| x.abs()).collect();

    let cols: [&[f64]; 18] = [
        dura,
        num,
        volume,
        vwap,
        &dura_rank,
        &num_rank,
        &rank_diff,
        &rank_abs,
        &yisdura_resi,
        &yisdura_rank,
        &yisnum_resi,
        &yisnum_rank,
        &rd_yisdura,
        &rd_yisdura_abs,
        &rd_yisnum,
        &rd_yisnum_abs,
        &rd_twins,
        &rd_twins_abs,
    ];
    feat6_vals(&cols, true)
}

// ============================================================================
// 名称清单
// ============================================================================

/// 静态块 names1..names9（每个 4 项），顺序即旧代码 res1..res9
const HEAD_NAMES: [&str; 36] = [
    "ask_corr_dura_num",
    "bid_corr_dura_num",
    "corr_diff",
    "corr_diff_abs",
    "ask_dura_count_when_num_pos",
    "bid_dura_count_when_num_pos",
    "dura_count_diff",
    "dura_count_diff_abs",
    "ask_num_count_when_dura_pos",
    "bid_num_count_when_dura_pos",
    "num_count_diff",
    "num_count_diff_abs",
    "ask_twins_volume_on_num_pos",
    "bid_twins_volume_on_num_pos",
    "twins_volume_diff",
    "twins_volume_diff_abs",
    "ask_volume_on_dura_pos",
    "bid_volume_on_dura_pos",
    "volume_on_dura_diff",
    "volume_on_dura_diff_abs",
    "ask_volume_on_num_pos",
    "bid_volume_on_num_pos",
    "volume_on_num_diff",
    "volume_on_num_diff_abs",
    "ask_twins_vwap_mean",
    "bid_twins_vwap_mean",
    "twins_vwap_mean_diff",
    "twins_vwap_mean_diff_abs",
    "ask_vwap_on_dura_pos_mean",
    "bid_vwap_on_dura_pos_mean",
    "vwap_on_dura_mean_diff",
    "vwap_on_dura_mean_diff_abs",
    "ask_vwap_on_num_pos_mean",
    "bid_vwap_on_num_pos_mean",
    "vwap_on_num_mean_diff",
    "vwap_on_num_mean_diff_abs",
];

const ASK_COLS: [&str; 18] = [
    "dura",
    "num",
    "volume",
    "vwap",
    "ask_dura_rank",
    "ask_num_rank",
    "ask_rank_diff",
    "ask_rank_abs",
    "yisdura_resi_ask",
    "yisdura_resi_ask_rank",
    "yisnum_resi_ask",
    "yisnum_resi_ask_rank",
    "ask_rank_diff_resi_yisdura",
    "ask_rank_diff_resi_yisdura_abs",
    "ask_rank_diff_resi_yisnum",
    "ask_rank_diff_resi_yisnum_abs",
    "ask_rank_diff_resi_twins",
    "ask_rank_diff_resi_twins_abs",
];

const BID_COLS: [&str; 18] = [
    "dura",
    "num",
    "volume",
    "vwap",
    "bid_dura_rank",
    "bid_num_rank",
    "bid_rank_diff",
    "bid_rank_abs",
    "yisdura_resi_bid",
    "yisdura_resi_bid_rank",
    "yisnum_resi_bid",
    "yisnum_resi_bid_rank",
    "bid_rank_diff_resi_yisdura",
    "bid_rank_diff_resi_yisdura_abs",
    "bid_rank_diff_resi_yisnum",
    "bid_rank_diff_resi_yisnum_abs",
    "bid_rank_diff_resi_twins",
    "bid_rank_diff_resi_twins_abs",
];

/// res11 那 12 列的列名（顺序即旧代码 DataFrame 的构造顺序）
const R11_COLS: [&str; 12] = [
    "dura_ask_ratios_full",
    "count_ask_ratios_full",
    "twins_ask_ratios_full",
    "dura_bid_ratios_full",
    "count_bid_ratios_full",
    "twins_bid_ratios_full",
    "dura_ask_ratios_half",
    "count_ask_ratios_half",
    "twins_ask_ratios_half",
    "dura_bid_ratios_half",
    "count_bid_ratios_half",
    "twins_bid_ratios_half",
];

/// res11 末尾 12 个总比值
const R11_SCALARS: [&str; 12] = [
    "dura_ask_ratio_full",
    "dura_bid_ratio_full",
    "count_ask_ratio_full",
    "count_bid_ratio_full",
    "twins_ask_ratio_full",
    "twins_bid_ratio_full",
    "dura_ask_ratio_half",
    "dura_bid_ratio_half",
    "count_ask_ratio_half",
    "count_bid_ratio_half",
    "twins_ask_ratio_half",
    "twins_bid_ratio_half",
];

pub const N_OUT: usize = 2 * 1082 + 84; // 2248

pub fn names() -> Vec<String> {
    let n_per_one = HEAD_NAMES.len() + 4 * feat6_names(&ASK_COLS, true).len() + 2;
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    let ask_feat = feat6_names(&ASK_COLS, true);
    let bid_feat = feat6_names(&BID_COLS, true);
    for prefix in ["full_", "half_"] {
        for n in HEAD_NAMES.iter() {
            out.push(format!("{}{}", prefix, n));
        }
        for n in ask_feat.iter() {
            out.push(format!("{}ask_v_{}", prefix, n));
        }
        for n in bid_feat.iter() {
            out.push(format!("{}bid_v_{}", prefix, n));
        }
        for n in ask_feat.iter() {
            out.push(format!("{}ask_bid_diff_v_{}", prefix, n));
        }
        for n in ask_feat.iter() {
            out.push(format!("{}ask_bid_diff_abs_v_{}", prefix, n));
        }
        out.push(format!("{}retreat_hold_ask_ratio", prefix));
        out.push(format!("{}retreat_hold_bid_ratio", prefix));
        debug_assert_eq!(out.len() % n_per_one, 0);
    }
    out.extend(feat6_names(&R11_COLS, false));
    for s in R11_SCALARS.iter() {
        out.push(s.to_string());
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// 单股单日
// ============================================================================

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let raws = read_day(code, date)?;

    // ---- 撤单拆分 ----
    let mut main_idx: Vec<usize> = Vec::with_capacity(raws.len());
    let mut ret_idx: Vec<usize> = Vec::new();
    for (i, r) in raws.iter().enumerate() {
        if r.flag == 32.0 {
            ret_idx.push(i);
        } else {
            main_idx.push(i);
        }
    }
    if main_idx.is_empty() {
        return None;
    }

    let ask_ret: HashSet<i64> = ret_idx.iter().map(|&i| raws[i].ask).collect();
    let bid_ret: HashSet<i64> = ret_idx.iter().map(|&i| raws[i].bid).collect();

    // df_retreat_other 在前、df_retreat 在后；各自再按 ask_order!=0 / bid_order!=0 取 volume
    let mut ra_v: Vec<f64> = Vec::new();
    let mut rb_v: Vec<f64> = Vec::new();
    for pass in 0..2 {
        let idxs: &Vec<usize> = if pass == 0 { &main_idx } else { &ret_idx };
        for &i in idxs.iter() {
            if pass == 0 && !(ask_ret.contains(&raws[i].ask) || bid_ret.contains(&raws[i].bid)) {
                continue;
            }
            let r = raws[i];
            if r.ask != 0 {
                ra_v.push(r.vol);
            }
            if r.bid != 0 {
                rb_v.push(r.vol);
            }
        }
    }
    let retreat_ask = pairwise_sum(&ra_v);
    let retreat_bid = pairwise_sum(&rb_v);

    // ---- 主表 ----
    let n = main_idx.len();
    let mut t: Vec<i64> = Vec::with_capacity(n);
    let mut vol: Vec<f64> = Vec::with_capacity(n);
    let mut turn: Vec<f64> = Vec::with_capacity(n);
    let mut ask: Vec<i64> = Vec::with_capacity(n);
    let mut bid: Vec<i64> = Vec::with_capacity(n);
    for &i in main_idx.iter() {
        let r = raws[i];
        t.push(r.t);
        vol.push(r.vol);
        turn.push(r.turn);
        ask.push(r.ask);
        bid.push(r.bid);
    }

    let total_vol = pairwise_sum(&vol);
    let total_turn = pairwise_sum(&turn);
    let vwap_allday = total_turn / total_vol;

    let (ask_gk, ask_gvol, ask_gturn) = group_sums(&ask, &vol, &turn);
    let (bid_gk, bid_gvol, bid_gturn) = group_sums(&bid, &vol, &turn);

    let ask_volfrac: Vec<f64> = ask_gvol.iter().map(|&x| x / total_vol).collect();
    let bid_volfrac: Vec<f64> = bid_gvol.iter().map(|&x| x / total_vol).collect();
    let ask_vwap: Vec<f64> = (0..ask_gk.len())
        .map(|i| (ask_gturn[i] / ask_volfrac[i]) / vwap_allday - 1.0)
        .collect();
    let bid_vwap: Vec<f64> = (0..bid_gk.len())
        .map(|i| (bid_gturn[i] / bid_volfrac[i]) / vwap_allday - 1.0)
        .collect();

    let d = Day {
        t,
        vol,
        ask,
        bid,
        ask_gk,
        ask_volfrac,
        ask_vwap,
        bid_gk,
        bid_volfrac,
        bid_vwap,
        retreat_ask,
        retreat_bid,
    };

    // ---- 订单级序列（旧代码：dura 与 num 各自独立过滤 > 0）----
    let (ak_all, ad_all, an_all) = order_series(&d.ask, &d.t, &d.vol);
    let (bk_all, bd_all, bn_all) = order_series(&d.bid, &d.t, &d.vol);
    let (ak, adv) = pos_filter(&ak_all, &ad_all);
    let (ank, anv2) = pos_filter(&ak_all, &an_all);
    let (bk, bdv) = pos_filter(&bk_all, &bd_all);
    let (bnk, bnv2) = pos_filter(&bk_all, &bn_all);

    // 旧代码里 4 个分母是 shape[0]，为 0 会 ZeroDivisionError（被 except 吞成 None）
    if ak.is_empty() || bk.is_empty() || ank.is_empty() || bnk.is_empty() {
        return None;
    }

    let full = Series4 {
        ak: ak.clone(),
        adv: adv.clone(),
        ank: ank.clone(),
        anv: anv2.clone(),
        bk: bk.clone(),
        bdv: bdv.clone(),
        bnk: bnk.clone(),
        bnv: bnv2.clone(),
    };
    let res_full = one(&d, &full)?;

    let qa = nan_quantile(&adv, 0.5);
    let qb = nan_quantile(&bdv, 0.5);
    let (hak, hadv) = ge_filter(&ak, &adv, qa);
    let (hbk, hbdv) = ge_filter(&bk, &bdv, qb);
    let (hank, hanv) = ge_filter(&ank, &anv2, nan_quantile(&anv2, 0.5));
    let (hbnk, hbnv) = ge_filter(&bnk, &bnv2, nan_quantile(&bnv2, 0.5));
    let half = Series4 {
        ak: hak,
        adv: hadv,
        ank: hank,
        anv: hanv,
        bk: hbk,
        bdv: hbdv,
        bnk: hbnk,
        bnv: hbnv,
    };
    let res_half = one(&d, &half)?;

    // ---- res11 ----
    let (ta, ca, ba, tta, cta, bta) = analyze_long_orders(&d.t, &d.ask, &d.vol, 1.0);
    let (tb, cb, bb, ttb, ctb, btb) = analyze_long_orders(&d.t, &d.bid, &d.vol, 1.0);
    let (tc, cc, bc, ttc, ctc, btc) = analyze_long_orders(&d.t, &d.ask, &d.vol, 0.5);
    let (td, cd, bd, ttd, ctd, btd) = analyze_long_orders(&d.t, &d.bid, &d.vol, 0.5);

    let cols12: [Vec<f64>; 12] = [ta, ca, ba, tb, cb, bb, tc, cc, bc, td, cd, bd];
    let maxn = cols12.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut mat: Vec<Vec<f64>> = Vec::with_capacity(12);
    for c in cols12.iter() {
        let mut v = c.clone();
        v.resize(maxn, f64::NAN);
        mat.push(v);
    }
    let refs: Vec<&[f64]> = mat.iter().map(|v| v.as_slice()).collect();
    let mut res11 = feat6_vals(&refs, false);
    res11.push(tta);
    res11.push(ttb);
    res11.push(cta);
    res11.push(ctb);
    res11.push(bta);
    res11.push(btb);
    res11.push(ttc);
    res11.push(ttd);
    res11.push(ctc);
    res11.push(ctd);
    res11.push(btc);
    res11.push(btd);

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for v in res_full.iter().chain(res_half.iter()).chain(res11.iter()) {
        out.push(*v as f32);
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// > 0 过滤（旧代码 `x[x > 0]`；NaN 自然落选）
fn pos_filter(keys: &[i64], vals: &[f64]) -> (Vec<i64>, Vec<f64>) {
    let mut k = Vec::new();
    let mut v = Vec::new();
    for i in 0..keys.len() {
        if vals[i] > 0.0 {
            k.push(keys[i]);
            v.push(vals[i]);
        }
    }
    (k, v)
}

fn ge_filter(keys: &[i64], vals: &[f64], q: f64) -> (Vec<i64>, Vec<f64>) {
    let mut k = Vec::new();
    let mut v = Vec::new();
    for i in 0..keys.len() {
        if vals[i] >= q {
            k.push(keys[i]);
            v.push(vals[i]);
        }
    }
    (k, v)
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
