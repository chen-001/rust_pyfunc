//! hm22「跨股票连结」——旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm22_跨股票连结/hm22_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   但因子函数自己调的是 read_trade_data_full_market(lazy=0)，
//!   也就是**一天全市场**的逐笔成交合成一张表，由它自己切窗口。
//!
//! ════════════════════════════════════════════════════════════════════
//! 为什么这个系列只能做横截面（compute_cross），不能逐股算
//! ════════════════════════════════════════════════════════════════════
//! `sing()` 里对每只股票算出一条「成交量档位 1..9 上 volume_ratio 之和」的 9 点曲线，
//! 再对这个 9×N 的表按**列**求相关矩阵 df.corr()。也就是说
//!   · trend       = corr(档位序号, 该股票曲线)
//!   · corr_skew   = 该股票那一列相关值（N+1 个，含与 'one' 档位列的相关）的偏度
//!   · corr_median = 同一列的中位数
//! 三个量都要用到全市场所有股票的曲线，逐股单独算是算不出来的（没有横截面就没有
//! 那一列相关值），所以走 kind='cross_section' 的 pipeline_groups + compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_full_market(lazy=0).sort('exchtime')
//!    ！！注意 with_retreat 取的是函数默认值 1，也就是**不丢 flag==32 的撤单行**。
//!      撤单行的 volume 照样进 volume_ratio 的分母（它是 pl.col('volume').sum().over('code')），
//!      只是 flag 既不是 66 也不是 83，所以不进任何档位求和。本文件因此用
//!      read_trade_fast_inner(code, date, true, ...) 把撤单行读进来，一个都不筛掉。
//!    这一条和「撤单过滤自己按数值筛」的通用要求不冲突：这里的正确答案是「不筛」。
//!
//! 2) 切窗口（旧代码用 pl.col('exchtime').dt.time() 比 pl.time(...)，
//!    exchtime 是微秒精度的 Datetime，比较是**微秒级**的，不是整秒）：
//!      all       : 09:30:00 <= t <= 14:57:00     （两端都含）
//!      open30    : 09:30:00 <  t <  10:00:00     （两端都不含）
//!      morning   : 09:30:00 <= t <  11:30:00
//!      afternoon : 13:00:00 <= t <  14:57:00
//!      close30   : 14:30:00 <= t <  14:57:00
//!    子窗口是从已经切好的 df 上再切的，所以和 all 的边界无关。
//! 3) sing(df)（每个时段单独算一遍，volume_ratio 的分母也是按时段单独求的）：
//!      a) volume_ratio = volume / volume.sum().over('code')   —— 用**未取整**的 volume
//!      b) volume = volume.round(0)                            —— 只用于后面的档位筛选
//!      c) for num in 100,200,...,900:
//!           子集 = volume == num
//!           再按 flag==66 / flag==83 分两路
//!           各自 group_by('code').sum(volume_ratio) → 每个 code 一个值
//!           pivot(index='one'(=num/100), on='code') → 1 行 × K 列
//!         9 个档位上下拼成 9 行 × K 列的表（缺的格子是 null），66 一路、83 一路
//!      d) 对每一路：
//!           先丢掉「9 个档位里有 null」的列，再丢掉「9 个值全 0」的列
//!           corr = df.corr()   （列两两 Pearson；底层是 numpy.corrcoef）
//!           trend       = corr 里 'one' 那一行、丢掉 'one' 列
//!           corr_skew   = corr 丢掉 'one' 列之后，每列（N+1 个值：与 'one' 的相关 + 与所有股票的相关）的偏度
//!           corr_median = 同一列的中位数
//!           → 三个 pandas Series，index 是股票代码（不含 'one'）
//!      e) 输出 6 个：trend66-trend83、rank 差、skew66-skew83、rank 差、median66-median83、rank 差
//!         rank 是 pandas Series.rank()（升序、并列取平均、NaN 保留），两条 Series 各自先排名再相减，
//!         对齐用的是 index 的并集 —— 只在一边出现的股票算出来是 NaN。
//!
//! names（names_single）：一个时段 6 个，五个时段顺序 all / open30 / morning / afternoon / close30，共 30 个。
//! names_save（10 个）：all_trend_rank, morning_corr_skew_rank, all_corr_skew_rank, morning_trend_rank,
//!   morning_corr_median, morning_corr_median_rank, afternoon_corr_median, afternoon_trend_rank,
//!   all_corr_median_rank, afternoon_corr_skew
//!
//! ════════════════════════════════════════════════════════════════════
//! 输出排布（横截面系列的约定，别搞反）
//! ════════════════════════════════════════════════════════════════════
//! compute_cross 返回 (codes, vals)，vals 是**股票优先**：
//! 每只股票连续 30 个值（五个时段各 6 个，时段顺序 all/open30/morning/afternoon/close30），
//! 第 i 只股票的值在 vals[i*30 .. i*30+30]，与 worker 侧
//! `vals.chunks(expected_len).zip(codes.iter())` 的分块方式一致。
//! expected_len = names().len() = 30。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度取舍（为什么全程必须是 f32、而且必须顺序累加）
//! ════════════════════════════════════════════════════════════════════
//! 实测（不是猜的）：polars 对 Float32 的求和——`group_by(...).agg(sum())` 和
//! `sum().over('code')`——都是**朴素的 f32 顺序累加**，既不是树形归约，也不是先升到 f64 再求和。
//! 两种规模的累加误差都不小，和 f64 精确值差得远：
//!   · 000001@20250303 全时段总成交量：f32 顺序和 145724432，f64 精确和 145756667
//!     （相对差 2.2e-4）；polars 的 over('code') 给的就是 145724432。
//!   · 000568@20250303「100 股 83 方向」那个档位组（18260 笔）：f32 顺序和 0.100027360022，
//!     f64 精确和 0.100033375193，polars 给的是 0.100027360022 —— 逐位相同。
//! 这两处只要有一处对不上，volume_ratio 就整列偏 1e-4 量级；而最终输出是
//! 「两个相关系数之差」（量级常常只有 1e-3），这点偏差会被放大成 1e-3 相对误差，过不了验收。
//! 所以这里**照抄 polars 的算法**：分母（每只股票每个时段的总成交量）与档位和都用 f32 顺序累加，
//! 顺序 = 文件里的行序（实测文件本身就是 exchtime 升序，与旧代码 sort('exchtime') 后同序）。
//! 这样两边的 volume_ratio 逐位相同、档位和逐位相同，后面的相关矩阵就只差 f64 舍入（~1e-16）。
//!
//! volume 用 f32 读取就够了：股数是整数，f32 能精确表示（>2^24 股的单笔极罕见，
//! 而且那种行也落不进 100~900 的档位筛选）。
//!
//! 精度取舍的第二个点：polars 的 skew 默认是 bias=True，也就是总体偏度 m3/m2^1.5
//! （m2、m3 都除以 n）；median 把 NaN 当最大值排在最后，偶数个取中间两个的平均。
//! 这两条都是实测 polars 1.30 的行为，不是猜的。
//!
//! 精度取舍的第三个点（这一步是最后一块拼图）：相关矩阵必须逐位复刻 numpy.corrcoef，
//! 三个细节都不能省，任何一个对不上都会让「两个相关值是否严格相等」变样，
//! 进而让 rank 类因子差半个名次（相对误差 1e-2 量级）：
//!   a) 求均值用 numpy 的 pairwise 求和顺序（9 个元素先并成 ((0+1)+(2+3))+((4+5)+(6+7)) 再加第 9 个）；
//!   b) 内积用**融合乘加**：numpy 的 X @ X.T 走 BLAS dgemm，
//!      scipy-openblas 在这台机器上编的是 Haswell 内核，9 维内积是顺序 FMA 累加；
//!   c) 相关值用两次除法 (c/d_i)/d_j，不能合并成一次 —— 正是这个不对称让
//!      C(i,j) 与 C(j,i) 在浮点上不再严格相等，旧代码才没有凭空多出并列。
//! 实测：a+b+c 全做之后，子集上 300 只股票 × 30 个因子与旧代码的最大相对误差是 1.6e-11。

use rayon::prelude::*;

use crate::data_paths;
use crate::fast_csv_reader::read_trade_fast_inner;

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

const N_WINDOWS: usize = 5;
const N_LEVELS: usize = 9;
const PER_WINDOW: usize = 6;
pub const N_OUT: usize = PER_WINDOW * N_WINDOWS; // 30

/// 五个时段：(下界, 下界含否, 上界, 上界含否)，单位是当天零点起的微秒。
const WINDOWS: [(i64, bool, i64, bool); N_WINDOWS] = [
    (LD_0930, true, LD_1457, true),    // all
    (LD_0930, false, LD_1000, false),  // open30
    (LD_0930, true, LD_1130, false),   // morning
    (LD_1300, true, LD_1457, false),   // afternoon
    (LD_1430, true, LD_1457, false),   // close30
];

const WINDOW_TAGS: [&str; N_WINDOWS] = ["all", "open30", "morning", "afternoon", "close30"];
const METRIC_NAMES: [&str; PER_WINDOW] = [
    "trend",
    "trend_rank",
    "corr_skew",
    "corr_skew_rank",
    "corr_median",
    "corr_median_rank",
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for tag in WINDOW_TAGS.iter() {
        for m in METRIC_NAMES.iter() {
            out.push(format!("{}_{}", tag, m));
        }
    }
    out
}

/// 每只股票在五个时段上的聚合量。
///
/// `total[w]` 是该时段全部逐笔（含撤单行）的成交量之和，用作 volume_ratio 的分母；
/// `sum66/sum83[w][l]` 是「成交量档位 l（=100*(l+1) 股）且 flag 分别为 66/83」那些行的
/// volume_ratio 之和；`has*` 区分「没有这样的行」和「和为 0」——
/// 旧代码里前者在 pivot 之后是 null，会让这一列被整列丢掉。
///
/// `total` 和档位和都用 f32 顺序累加，理由见文件头的精度取舍一节
/// （要和 polars 的 f32 顺序累加逐位对齐）。
#[derive(Clone, Copy)]
struct StockAgg {
    total: [f32; N_WINDOWS],
    sum66: [[f32; N_LEVELS]; N_WINDOWS],
    sum83: [[f32; N_LEVELS]; N_WINDOWS],
    has66: [[bool; N_LEVELS]; N_WINDOWS],
    has83: [[bool; N_LEVELS]; N_WINDOWS],
}

impl StockAgg {
    fn new() -> Self {
        StockAgg {
            total: [0.0; N_WINDOWS],
            sum66: [[0.0; N_LEVELS]; N_WINDOWS],
            sum83: [[0.0; N_LEVELS]; N_WINDOWS],
            has66: [[false; N_LEVELS]; N_WINDOWS],
            has83: [[false; N_LEVELS]; N_WINDOWS],
        }
    }
}

/// 读一只股票当天的逐笔，按五个时段聚合。
///
/// 读不到文件、或当天的成交为空时返回 None（上游会补 NaN，保证因子个数恒定）。
fn aggregate_stock(code: &str, date: i64) -> Option<StockAgg> {
    // with_retreat=true：撤单行必须保留（旧代码 read_trade_data_full_market 的 with_retreat 默认是 1）。
    // with_afternoon_adjust=false：旧代码没有调 adjust_afternoon，窗口用的是真实钟点。
    let trades = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;
    if trades.is_empty() {
        return None;
    }
    let mut agg = StockAgg::new();
    // 第一遍：算每个时段的总成交量（volume_ratio 的分母）。
    // 必须 f32 顺序累加：polars 的 sum().over('code') 就是这么算的（实测见文件头）。
    for r in trades.iter() {
        let t = r.time_us.rem_euclid(DAY_US);
        for w in 0..N_WINDOWS {
            let (lo, lo_incl, hi, hi_incl) = WINDOWS[w];
            let ok_lo = if lo_incl { t >= lo } else { t > lo };
            let ok_hi = if hi_incl { t <= hi } else { t < hi };
            if ok_lo && ok_hi {
                agg.total[w] += r.volume;
            }
        }
    }
    // 第二遍：按成交量档位与 flag 累加 volume_ratio。
    // 同样 f32 顺序累加（与 polars 的 group_by.sum() 逐位对齐）。
    for r in trades.iter() {
        let flag = r.flag;
        if flag != 66 && flag != 83 {
            continue;
        }
        let rv = r.volume.round();
        let lv = (rv / 100.0) as i32;
        if !(1..=9).contains(&lv) || rv != (lv as f32) * 100.0 {
            continue;
        }
        let l = (lv - 1) as usize;
        let t = r.time_us.rem_euclid(DAY_US);
        for w in 0..N_WINDOWS {
            let (lo, lo_incl, hi, hi_incl) = WINDOWS[w];
            let ok_lo = if lo_incl { t >= lo } else { t > lo };
            let ok_hi = if hi_incl { t <= hi } else { t < hi };
            if !(ok_lo && ok_hi) {
                continue;
            }
            // 旧代码是 Float32 / Float32 的除法，这里照做。
            let ratio = r.volume / agg.total[w];
            if flag == 66 {
                agg.sum66[w][l] += ratio;
                agg.has66[w][l] = true;
            } else {
                agg.sum83[w][l] += ratio;
                agg.has83[w][l] = true;
            }
        }
    }
    Some(agg)
}

/// 某一路上「保留下来的股票列」：9 个档位都有值，且不是 9 个值全 0。
fn retained(has: &[bool; N_LEVELS], sum: &[f32; N_LEVELS]) -> bool {
    if !has.iter().all(|&b| b) {
        return false;
    }
    sum.iter().any(|&v| v != 0.0)
}

/// 总体偏度 m3/m2^1.5（polars `skew()` 默认 bias=True 的算法）。
/// 任一元素是 NaN → NaN；m2 == 0 → NaN（0/0）。
fn skew_bias(x: &[f64]) -> f64 {
    if x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let m: f64 = x.iter().sum::<f64>() / n;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &v in x {
        let d = v - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    if !(m2 > 0.0) {
        return f64::NAN;
    }
    m3 / (m2.sqrt() * m2)
}

/// 全序比较：NaN 排在最大值之后（polars 的 median 就是这么排的，实测 [0.5,NaN,0.5,1.0] → 0.75）。
fn cmp_total(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// polars 的 median：把 NaN 当最大值排在最后，偶数个取中间两个的平均。
fn median_total(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    buf.select_nth_unstable_by(mid, cmp_total);
    let upper = buf[mid];
    if n % 2 == 1 {
        return upper;
    }
    // 偶数：还要下半区（含 mid 之前）的最大值；用 select_nth 把 mid-1 放到位再取它
    let lower = {
        let (left, _) = buf.split_at_mut(mid);
        let k = mid - 1;
        left.select_nth_unstable_by(k, cmp_total);
        left[k]
    };
    (lower + upper) / 2.0
}

/// pandas `Series.rank()`：升序、并列取平均名次、NaN 保留为 NaN（不参与排名）。
/// 返回与输入等长的名次数组。
fn rank_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| !values[i].is_nan()).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

/// numpy 对 9 个元素做 mean(axis=1) 时的 pairwise 求和顺序（loops_utils.h 的 n<=128 分支）：
/// 前 8 个进 8 个累加器，按 ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)) 合并，最后再加第 9 个。
/// 逐位复刻是为了让「相关矩阵对称元是否严格相等」这件事和旧代码一致 ——
/// 排名（rank）对严格并列很敏感，差半个名次就是 1e-2 的相对误差。
#[inline]
fn np_mean9(x: &[f64; N_LEVELS]) -> f64 {
    let res = ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
    (res + x[8]) / N_LEVELS as f64
}

/// 两列的中心化内积（numpy 里是 X @ X.T，9 维内积）。
///
/// 必须用 mul_add（融合乘加，FMA）：numpy 的 X @ X.T 走 BLAS dgemm，
/// scipy-openblas 在这台机器上编的是 Haswell 内核，9 维内积是**顺序 FMA**累加出来的
/// （vfmadd231pd 一个累加器走完 k）。非融合的 a*b 再相加会差 1 个 ULP，
/// 而 1 个 ULP 就足以改变「两个相关值是否严格相等」，进而让 rank 差半个名次（相对误差 1e-2）。
/// 实测：加了 FMA 之后，子集上 300 只股票 × 30 个因子与旧代码的最大相对误差从 5.9e-2 降到 1.6e-11。
#[inline]
fn dot9(a: &[f64; N_LEVELS], b: &[f64; N_LEVELS]) -> f64 {
    let mut s = 0.0f64;
    for l in 0..N_LEVELS {
        s = a[l].mul_add(b[l], s);
    }
    s
}

/// 一路（flag 固定）在某个时段上的横截面结果：trend / corr_skew / corr_median
/// 各自的原值和名次，长度 = 保留股票的个数。
struct BranchStats {
    idx: Vec<usize>,
    trend: Vec<f64>,
    skew: Vec<f64>,
    median: Vec<f64>,
    trend_rank: Vec<f64>,
    skew_rank: Vec<f64>,
    median_rank: Vec<f64>,
}

/// 算一路的横截面统计。
///
/// `curves` 是保留下来的股票曲线（每只 9 个档位的 volume_ratio 之和），
/// 内部前面补一列 'one'（值 1..9），得到旧代码那张 9 行 × (K+1) 列的表。
/// 相关矩阵逐位复刻 numpy.corrcoef（polars 的 DataFrame.corr 就是调它）：
///     avg = np.mean(X, axis=1)                       // 9 个元素的 pairwise 求和
///     X  -= avg[:, None]
///     c   = X @ X.T ;  c *= 1/(n-1)                  // n=9 → *0.125（2 的幂，精确）
///     d   = sqrt(diag(c))
///     C   = (c / d[:, None]) / d[None, :]            // 两次除法，不合并成一次
///     np.clip(C, -1, 1)
/// 两次除法这一步不能省：它让 C 的第 (i,j) 与第 (j,i) 个元素在浮点上不再严格相等
/// （旧代码里正是靠这个才没有出现假并列），合并成一次除法会凭空造出严格并列，
/// 进而让 rank 差半个名次。
/// 输出的三个量都是「股票」那一侧（长度 K），顺序与 `idx` 一致。
fn branch_stats(curves: &[[f64; N_LEVELS]], idx: Vec<usize>) -> BranchStats {
    let k = curves.len();
    let n = k + 1;
    // 中心化后的矩阵（行 = 'one' + 各股票，列 = 9 个档位）
    let mut xc = vec![[0.0f64; N_LEVELS]; n];
    {
        let mut row = [0.0f64; N_LEVELS];
        for (l, slot) in row.iter_mut().enumerate() {
            *slot = (l + 1) as f64;
        }
        let m = np_mean9(&row);
        for l in 0..N_LEVELS {
            xc[0][l] = row[l] - m;
        }
    }
    for i in 0..k {
        let m = np_mean9(&curves[i]);
        for l in 0..N_LEVELS {
            xc[i + 1][l] = curves[i][l] - m;
        }
    }
    // d_i = sqrt(c_ii)，c_ii = (Σ x²) * 0.125
    let d: Vec<f64> = (0..n)
        .map(|i| (dot9(&xc[i], &xc[i]) * 0.125).sqrt())
        .collect();

    let mut trend = vec![f64::NAN; k];
    let mut skew = vec![f64::NAN; k];
    let mut median = vec![f64::NAN; k];
    // 逐列并行：每列 = 相关矩阵的一列（K+1 个值），trend 取与 'one'（行 0）的那个。
    let cols: Vec<(f64, f64, f64)> = (1..n)
        .into_par_iter()
        .map(|j| {
            let dj = d[j];
            let mut col = vec![f64::NAN; n];
            for i in 0..n {
                let cij = dot9(&xc[i], &xc[j]) * 0.125;
                col[i] = ((cij / d[i]) / dj).clamp(-1.0, 1.0);
            }
            let mut buf = col.clone();
            (col[0], skew_bias(&col), median_total(&mut buf))
        })
        .collect();
    for (t, (a, b, c)) in cols.into_iter().enumerate() {
        trend[t] = a;
        skew[t] = b;
        median[t] = c;
    }

    let trend_rank = rank_average(&trend);
    let skew_rank = rank_average(&skew);
    let median_rank = rank_average(&median);
    BranchStats {
        idx,
        trend,
        skew,
        median,
        trend_rank,
        skew_rank,
        median_rank,
    }
}

/// 横截面入口：一次算一天全市场，返回 (股票代码, 扁平因子值)。
///
/// 值按股票分块排列（每只股票连续 30 个），与 `dispatch_cross` 的
/// `vals.chunks(expected_len).zip(codes)` 对齐。
pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    let codes = data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return None;
    }
    // 逐股并行读；读不到的整条 NaN。
    let aggs: Vec<Option<StockAgg>> = codes
        .par_iter()
        .map(|code| aggregate_stock(code, date))
        .collect();

    let n_codes = codes.len();
    let mut out = vec![f64::NAN; n_codes * N_OUT];

    for w in 0..N_WINDOWS {
        // 两路各挑出保留的列：9 个档位都有值、且不是全 0。
        let mut idx66 = Vec::new();
        let mut curves66 = Vec::new();
        let mut idx83 = Vec::new();
        let mut curves83 = Vec::new();
        for (i, agg) in aggs.iter().enumerate() {
            let Some(a) = agg else { continue };
            if retained(&a.has66[w], &a.sum66[w]) {
                let mut c = [0.0f64; N_LEVELS];
                for (l, slot) in c.iter_mut().enumerate() {
                    *slot = a.sum66[w][l] as f64;
                }
                idx66.push(i);
                curves66.push(c);
            }
            if retained(&a.has83[w], &a.sum83[w]) {
                let mut c = [0.0f64; N_LEVELS];
                for (l, slot) in c.iter_mut().enumerate() {
                    *slot = a.sum83[w][l] as f64;
                }
                idx83.push(i);
                curves83.push(c);
            }
        }
        let b66 = branch_stats(&curves66, idx66);
        let b83 = branch_stats(&curves83, idx83);

        // 把两路按股票对齐（旧代码是 pandas 按 index 并集对齐，只在一边出现的算 NaN）。
        let mut slot66 = vec![usize::MAX; n_codes];
        for (p, &i) in b66.idx.iter().enumerate() {
            slot66[i] = p;
        }
        let mut slot83 = vec![usize::MAX; n_codes];
        for (p, &i) in b83.idx.iter().enumerate() {
            slot83[i] = p;
        }
        for i in 0..n_codes {
            let a = slot66[i];
            let b = slot83[i];
            let base = i * N_OUT + w * PER_WINDOW;
            let pick = |v: &Vec<f64>, s: usize| -> f64 {
                if s == usize::MAX {
                    f64::NAN
                } else {
                    v[s]
                }
            };
            out[base] = pick(&b66.trend, a) - pick(&b83.trend, b);
            out[base + 1] = pick(&b66.trend_rank, a) - pick(&b83.trend_rank, b);
            out[base + 2] = pick(&b66.skew, a) - pick(&b83.skew, b);
            out[base + 3] = pick(&b66.skew_rank, a) - pick(&b83.skew_rank, b);
            out[base + 4] = pick(&b66.median, a) - pick(&b83.median, b);
            out[base + 5] = pick(&b66.median_rank, a) - pick(&b83.median_rank, b);
        }
    }

    let flat: Vec<f32> = out.into_iter().map(|v| v as f32).collect();
    debug_assert_eq!(flat.len(), n_codes * N_OUT);
    Some((codes, flat))
}

/// hm22 是横截面系列，没有逐股实现。
pub fn compute_stock(_code: &str, _date: i64) -> Option<Vec<f32>> {
    None
}
