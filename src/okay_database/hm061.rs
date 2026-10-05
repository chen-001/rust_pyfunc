//! hm61「来去均衡」——旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm61_来去均衡/hm61_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   逐股系列（check_my_kind == "level2"），只用盘口快照。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) asks, bids = read_market_data_single_price_vol_pair()
//!    · read_market_data_single()（get_many=0/read_full=0）把读进来的整张盘口表原样返回，
//!      只做 deal_data（change_time=1，把 exchtime 从 epoch 微秒换成 Timestamp）。
//!      这个系列完全不用索引/时间，只用 exchtime 排序，所以时间换算不影响结果。
//!    · `df = df[df.last_prc != 0]` —— 开盘前 last_prc=0 的快照先扔掉。
//!    · 把 10 档 ask_prc 熔成长表（每行 10 条，number 从 'ask_prcN' 取第 7 位之后 → 1..10），
//!      再把 10 档 ask_vol 熔成长表按列拼上去；bids 同理。
//!      `asks = asks[asks.price != 0]`（价格为空档位的整条丢掉，**成交量不筛**）。
//!    · 最后 reset_index().sort_values(['exchtime','number'])。
//!      实测这个排序其实不影响结果：后面全是 groupby 聚合与分位数，都与行序无关。
//! 2) 三个价格区间：
//!      asks_inner80 = asks[price 落在 asks[number==1].price 的 10%~90% 分位之间]
//!      bids_inner80 = bids[price 落在 **bids** 自己的 number==1 价格的 10%~90% 分位之间]
//!      inner50 用 25%~75% 分位。！！买和卖各自用自己的分位数，不能混用（混用会全错）。
//!      分位数是 pandas 的线性插值口径：位置 (n-1)q，正好落在整点上就取那个值。
//! 3) sing(asks, bids)（全部 / inner80 / inner50 各算一遍，共 3 遍 151 个值）：
//!      a) asks = asks[vol != 0]; bids = bids[vol != 0]
//!      b) price_both = set(asks.price) & set(bids.price)
//!      c) 按 price 分组求 vol 的 max / min / median（groupby 默认 skipna、按 price 升序）
//!         再把不在 price_both 里的价格整条丢掉（顺序仍是 price 升序）
//!      d) 十条序列（两条一组相减，后五条取绝对值）：
//!         amax_bmax, amin_bmin, amax_bmin, amin_bmax, amid_bmid 及各自的 abs 版
//!      e) 每条序列 11 个统计量：mean, median, std(ddof=1), skew, kurt, min, max, sum,
//!         autocorr(1), autocorr(2), autocorr(3)     → 110 个
//!      f) 再 41 个两两相关系数（顺序见 CORR_PAIRS；注意第 3 组的
//!         `amax_bmin_corr_amid_bmid_abs` 在旧代码里**写了两遍**，是原样复刻）  → 41 个
//! 4) 输出 = sing(全部) + sing(inner80) + sing(inner50) = 151 × 3 = 453 个值。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、pandas 统计量的口径（实测，不是猜的）
//! ════════════════════════════════════════════════════════════════════
//! · mean / min / max：跳过 NaN；全 NaN → NaN（sum 例外，全 NaN 时 pandas 给 0.0）
//! · median：跳过 NaN，偶数个取中间两个的平均
//! · std：ddof=1，样本 < 2 → NaN
//! · **skew 不是总体偏度**：pandas 给的是调整后的 Fisher-Pearson G1
//!     g1 = (n·√(n-1)/(n-2)) · (m3/m2^1.5)，其中 m2=Σ(x-mean)²、m3=Σ(x-mean)³ 都**不除以 n**；
//!     n<3 → NaN；m2 恰好为 0 → 0。|m2| 或 |m3| < 1e-14 时按 0 处理（_zero_out_fperr）
//! · **kurt 也不是总体峰度**：无偏超额峰度 G2
//!     [n(n+1)(n-1)m4] / [(n-2)(n-3)m2²] − 3(n-1)²/((n-2)(n-3))，n<4 → NaN，分母 0 → 0
//! · autocorr(lag) = corr(x, x.shift(lag))：成对完整观测（任一 NaN 丢掉这一对），
//!     样本 < 2 或任一侧方差为 0 → NaN。
//!     ！！`Series.autocorr` 在 `lag >= len(self)` 时会**抛 ValueError**，
//!     旧代码的 try/except 会吞掉它并让 get_factor 返回 None → 该股该日 453 个值全 NaN。
//!     所以这里只要有一组公共价格数 ≤ 3，整条就返回 None。
//! · 两两相关系数：pandas 先按索引对齐、mask = 两侧都非 NaN，再 np.corrcoef。
//!   两个序列长度相同（同一批公共价格），所以就是按位置成对。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、输出与命名
//! ════════════════════════════════════════════════════════════════════
//! names() 的 453 个名字按 names_single 原样生成，**与旧代码一样带一个重名**
//! （第一块里的 amax_bmin_corr_amid_bmid_abs 出现两次）。FactorDict 用的是 Vec<String>、
//! 不去重，所以重名不会打架；要落盘的 names_save 里不包含这个名字。

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;

// ── 盘口 CSV 列下标（0 基，与文件头一致） ──
// symbol(0) datetime(1) date(2) time(3) exchtime(4) localtime(5) last_prc(6) ...
// ask_prc1(21) ask_vol1(22) bid_prc1(23) bid_vol1(24) ask_prc2(25) ...
const COL_EXCHTIME: usize = 4;
const COL_LAST_PRC: usize = 6;
const COL_ASK_PRC1: usize = 21;

const N_LEVELS: usize = 10;
const N_SERIES: usize = 10;
const N_STATS: usize = 11;
const N_CORR: usize = 41;
const PER_BLOCK: usize = N_SERIES * N_STATS + N_CORR; // 151
const N_BLOCKS: usize = 3;
pub const N_OUT: usize = PER_BLOCK * N_BLOCKS; // 453

/// 十条序列的标签（顺序 = sing() 里 res1 的拼接顺序）
const SERIES_TAGS: [&str; N_SERIES] = [
    "amax_bmax",
    "amin_bmin",
    "amax_bmin",
    "amin_bmax",
    "amid_bmid",
    "amax_bmax_abs",
    "amin_bmin_abs",
    "amax_bmin_abs",
    "amin_bmax_abs",
    "amid_bmid_abs",
];

const STAT_TAGS: [&str; N_STATS] = [
    "mean", "median", "std", "skew", "kurt", "min", "max", "sum", "autocorr1", "autocorr2",
    "autocorr3",
];

/// 41 组相关系数（下标对应 SERIES_TAGS 的位置），顺序与旧代码 res2 完全一致。
/// 第 3 组最后一个 (2, 9) 是旧代码里的重复项，保留。
const CORR_PAIRS: [(usize, usize); N_CORR] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 3), (2, 4), (2, 9), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (5, 6), (5, 7), (5, 8), (5, 9),
    (6, 7), (6, 8), (6, 9),
    (7, 8), (7, 9),
    (8, 9),
];

const BLOCK_TAGS: [&str; N_BLOCKS] = ["全部", "价格中间80perc", "价格中间50perc"];

/// names() 的 453 个名字按 names_single 生成，唯一的例外是**重名**：
/// 旧 names_single 里 `amax_bmin_corr_amid_bmid_abs_全部` 出现了两次（第一块里第 129、134 位，
/// 旧代码写重复的一组相关），这两列的值本来也完全相同。列式 store 的组合校验不允许重名
/// （实测报「组合 store 因子名重复: amax_bmin_corr_amid_bmid_abs_全部」），
/// 所以这里给**第二次出现**的那一列加后缀 `_dup2`：名字变了、值不变、列数仍是 453。
/// 这一列不在 names_save 里，对入库名单没有影响。
pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    let mut seen: HashSet<String> = HashSet::with_capacity(N_OUT);
    fn push(out: &mut Vec<String>, seen: &mut HashSet<String>, name: String) {
        if seen.insert(name.clone()) {
            out.push(name);
            return;
        }
        let mut k = 2usize;
        loop {
            let cand = format!("{}_dup{}", name, k);
            if seen.insert(cand.clone()) {
                out.push(cand);
                return;
            }
            k += 1;
        }
    }
    for block in BLOCK_TAGS.iter() {
        for tag in SERIES_TAGS.iter() {
            for st in STAT_TAGS.iter() {
                push(&mut out, &mut seen, format!("{}_{}_{}", tag, st, block));
            }
        }
        for (i, j) in CORR_PAIRS.iter() {
            push(
                &mut out,
                &mut seen,
                format!("{}_corr_{}_{}", SERIES_TAGS[*i], SERIES_TAGS[*j], block),
            );
        }
    }
    out
}

// ============================================================================
// CSV 读取
// ============================================================================

#[inline]
fn split_into<'a>(bytes: &'a [u8], buf: &mut [&'a [u8]; 64]) -> usize {
    let mut start = 0usize;
    let mut n = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b',' {
            if n < 64 {
                buf[n] = &bytes[start..i];
            }
            n += 1;
            start = i + 1;
        }
    }
    if n < 64 {
        buf[n] = &bytes[start..];
    }
    (n + 1).min(64)
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

/// 一条挂单：(时间, 档位, 价格, 成交量)
#[derive(Clone, Copy)]
struct Entry {
    ex: f64,
    num: f64,
    price: f64,
    vol: f64,
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

/// 读一只股票当天的盘口，返回 (ask 挂单, bid 挂单)，按 (exchtime, 档位) 升序。
/// 两次筛选与旧代码一致：先 `last_prc != 0` 丢快照，再按档位丢 price == 0 的格子。
fn read_market(code: &str, date: i64) -> Option<(Vec<Entry>, Vec<Entry>)> {
    let path = market_path(code, date)?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut asks: Vec<Entry> = Vec::new();
    let mut bids: Vec<Entry> = Vec::new();
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; 64] = [&[][..]; 64];
        let nf = split_into(bytes, &mut f);
        if nf <= COL_ASK_PRC1 + 39 {
            continue;
        }
        let last = parse_f64(f[COL_LAST_PRC]);
        if last == 0.0 {
            continue;
        }
        let ex = parse_f64(f[COL_EXCHTIME]);
        for k in 0..N_LEVELS {
            let base = COL_ASK_PRC1 + k * 4;
            let ap = parse_f64(f[base]);
            if ap != 0.0 {
                asks.push(Entry { ex, num: (k + 1) as f64, price: ap, vol: parse_f64(f[base + 1]) });
            }
            let bp = parse_f64(f[base + 2]);
            if bp != 0.0 {
                bids.push(Entry { ex, num: (k + 1) as f64, price: bp, vol: parse_f64(f[base + 3]) });
            }
        }
    }
    let key = |a: &Entry, b: &Entry| {
        a.ex.partial_cmp(&b.ex)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.num.partial_cmp(&b.num).unwrap_or(std::cmp::Ordering::Equal))
    };
    asks.sort_by(key);
    bids.sort_by(key);
    Some((asks, bids))
}

// ============================================================================
// pandas 口径的统计量
// ============================================================================

/// pandas `Series.quantile(q)`：默认线性插值，位置 (n-1)q。
fn quantile_linear(vals: &mut Vec<f64>, q: f64) -> f64 {
    vals.retain(|v| !v.is_nan());
    let n = vals.len();
    if n == 0 {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = (n - 1) as f64 * q;
    let lo = pos.floor();
    let hi = pos.ceil();
    let li = lo as usize;
    let hi_i = hi as usize;
    if li == hi_i {
        vals[li]
    } else {
        vals[li] + (vals[hi_i] - vals[li]) * (pos - lo)
    }
}

/// pandas 的 `_zero_out_fperr`：|x| < 1e-14 当 0 处理。
#[inline]
fn zero_out(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        0.0
    } else {
        x
    }
}

/// pandas `Series.skew()`：调整后的 Fisher-Pearson G1。
fn p_skew(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 3.0 {
        return f64::NAN;
    }
    let mean: f64 = v.iter().sum::<f64>() / n;
    let (mut m2, mut m3) = (0.0f64, 0.0f64);
    for &x in v {
        let d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    let m2 = zero_out(m2);
    let m3 = zero_out(m3);
    if m2 == 0.0 {
        return 0.0;
    }
    (n * (n - 1.0).sqrt() / (n - 2.0)) * (m3 / m2.powf(1.5))
}

/// pandas `Series.kurt()`：无偏超额峰度 G2。
fn p_kurt(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 4.0 {
        return f64::NAN;
    }
    let mean: f64 = v.iter().sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0f64, 0.0f64);
    for &x in v {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    let mut num = n * (n + 1.0) * (n - 1.0) * m4;
    let mut den = (n - 2.0) * (n - 3.0) * m2 * m2;
    num = zero_out(num);
    den = zero_out(den);
    if den == 0.0 {
        return 0.0;
    }
    let adj = 3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));
    num / den - adj
}

/// 成对完整观测的 Pearson 相关（pandas `Series.corr` 口径）：
/// 任一侧 NaN 丢掉这一对；有效样本 < 2 或任一侧方差为 0 → NaN。
fn pearson_pair(a: &[f64], b: &[f64]) -> f64 {
    let mut n = 0.0f64;
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        let (vx, vy) = (a[i], b[i]);
        n += 1.0;
        let dx = vx - mx;
        let dy = vy - my;
        mx += dx / n;
        my += dy / n;
        sxx += (vx - mx) * dx;
        syy += (vy - my) * dy;
        sxy += (vx - mx) * dy;
    }
    if n < 2.0 {
        return f64::NAN;
    }
    let d = (sxx * syy).sqrt();
    if d == 0.0 {
        f64::NAN
    } else {
        sxy / d
    }
}

/// 一条序列的 11 个统计量（顺序见 STAT_TAGS）。空序列在这里不会出现（上游已判长度）。
fn stats11(x: &[f64]) -> [f64; N_STATS] {
    let mut out = [f64::NAN; N_STATS];
    let v: Vec<f64> = x.iter().copied().filter(|z| !z.is_nan()).collect();
    let n = v.len();
    out[7] = v.iter().sum::<f64>(); // sum：全 NaN 时 pandas 给 0.0
    if n == 0 {
        return out;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    out[0] = mean;
    let mut sorted = v.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out[1] = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    if n >= 2 {
        let mut ss = 0.0f64;
        for &z in v.iter() {
            ss += (z - mean) * (z - mean);
        }
        out[2] = (ss / (n - 1) as f64).sqrt();
    }
    out[3] = p_skew(&v);
    out[4] = p_kurt(&v);
    out[5] = sorted[0];
    out[6] = sorted[n - 1];
    // autocorr(lag) = corr(x, x.shift(lag))：按 (x[t], x[t-lag]) 成对，NaN 成对丢掉
    for lag in 1..=3usize {
        if lag >= x.len() {
            continue; // 上游已保证 len > 3
        }
        let mut a = Vec::with_capacity(x.len() - lag);
        let mut b = Vec::with_capacity(x.len() - lag);
        for t in lag..x.len() {
            a.push(x[t]);
            b.push(x[t - lag]);
        }
        out[7 + lag] = pearson_pair(&a, &b);
    }
    out
}

// ============================================================================
// sing()
// ============================================================================

/// 按价格分组算 max / min / median（pandas groupby 口径：按 price 升序、skipna）。
/// 输入是已经过 `vol != 0` 的 (price, vol)。返回 (价格升序的价格, max, min, median)。
fn group_by_price(e: &[Entry]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..e.len()).collect();
    idx.sort_by(|&a, &b| {
        e[a].price
            .partial_cmp(&e[b].price)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut prices = Vec::new();
    let (mut mx, mut mn, mut md) = (Vec::new(), Vec::new(), Vec::new());
    let mut i = 0usize;
    while i < idx.len() {
        let p = e[idx[i]].price;
        let mut j = i;
        let mut vals: Vec<f64> = Vec::new();
        while j < idx.len() && e[idx[j]].price == p {
            let v = e[idx[j]].vol;
            if !v.is_nan() {
                vals.push(v);
            }
            j += 1;
        }
        prices.push(p);
        if vals.is_empty() {
            mx.push(f64::NAN);
            mn.push(f64::NAN);
            md.push(f64::NAN);
        } else {
            let mut s = vals.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            mx.push(s[s.len() - 1]);
            mn.push(s[0]);
            let n = s.len();
            md.push(if n % 2 == 1 {
                s[n / 2]
            } else {
                (s[n / 2 - 1] + s[n / 2]) / 2.0
            });
        }
        i = j;
    }
    (prices, mx, mn, md)
}

/// 一段 sing()：返回 151 个值。
///
/// ！！这里**不能**对「公共价格数 ≤ 3」提前退出：pandas 2.3.3 的 `Series.autocorr`
/// 已经不做 `lag >= len(self)` 的长度检查，短序列只会算出一串 NaN
/// （实测 000001@20250303 的价格中间50perc 就只有 3 个公共价格，kurt/autocorr2/autocorr3 是 NaN，
///  其余照常给值）。公共价格为 0 时也一样：sum 是 0.0，其余 NaN。
fn sing(asks: &[Entry], bids: &[Entry]) -> Vec<f64> {
    let av: Vec<Entry> = asks.iter().copied().filter(|e| e.vol != 0.0).collect();
    let bv: Vec<Entry> = bids.iter().copied().filter(|e| e.vol != 0.0).collect();
    let (ap, amax, amin, amid) = group_by_price(&av);
    let (bp, bmax, bmin, bmid) = group_by_price(&bv);

    // price_both = set(asks.price) & set(bids.price)，顺序沿用 asks 的升序
    let mut common: Vec<usize> = Vec::new();
    let mut bidx: Vec<usize> = Vec::new();
    {
        let mut p = 0usize;
        for i in 0..ap.len() {
            while p < bp.len() && bp[p] < ap[i] {
                p += 1;
            }
            if p < bp.len() && bp[p] == ap[i] {
                common.push(i);
                bidx.push(p);
            }
        }
    }
    let n = common.len();
    let mut s: [Vec<f64>; N_SERIES] = std::array::from_fn(|_| Vec::new());
    let mut base: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
    for k in 0..5 {
        let mut v = Vec::with_capacity(n);
        for t in 0..n {
            let i = common[t];
            let j = bidx[t];
            v.push(match k {
                0 => amax[i] - bmax[j],
                1 => amin[i] - bmin[j],
                2 => amax[i] - bmin[j],
                3 => amin[i] - bmax[j],
                _ => amid[i] - bmid[j],
            });
        }
        base[k] = v;
    }
    for k in 0..5 {
        s[k] = base[k].clone();
    }
    for k in 0..5 {
        s[k + 5] = base[k].iter().map(|v| v.abs()).collect();
    }

    let mut out: Vec<f64> = Vec::with_capacity(PER_BLOCK);
    for ser in s.iter() {
        out.extend_from_slice(&stats11(ser));
    }
    for (i, j) in CORR_PAIRS.iter() {
        out.push(pearson_pair(&s[*i], &s[*j]));
    }
    debug_assert_eq!(out.len(), PER_BLOCK);
    out
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 一只股票的一天：算出 453 个因子值。算不出来返回 None（上游补整条 NaN）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let (asks, bids) = read_market(code, date)?;

    // 三个价格区间的分位数：买、卖各用自己 number==1 的价格（不可混用！）
    let mut lvl1a: Vec<f64> = asks.iter().filter(|e| e.num == 1.0).map(|e| e.price).collect();
    let mut lvl1b: Vec<f64> = bids.iter().filter(|e| e.num == 1.0).map(|e| e.price).collect();
    let a10 = quantile_linear(&mut lvl1a, 0.10);
    let a90 = quantile_linear(&mut lvl1a, 0.90);
    let a25 = quantile_linear(&mut lvl1a, 0.25);
    let a75 = quantile_linear(&mut lvl1a, 0.75);
    let b10 = quantile_linear(&mut lvl1b, 0.10);
    let b90 = quantile_linear(&mut lvl1b, 0.90);
    let b25 = quantile_linear(&mut lvl1b, 0.25);
    let b75 = quantile_linear(&mut lvl1b, 0.75);

    let in_range = |e: &Entry, lo: f64, hi: f64| e.price >= lo && e.price <= hi;
    let sel = |list: &[Entry], lo: f64, hi: f64| -> Vec<Entry> {
        list.iter().copied().filter(|e| in_range(e, lo, hi)).collect()
    };

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    out.extend(sing(&asks, &bids));
    out.extend(sing(&sel(&asks, a10, a90), &sel(&bids, b10, b90)));
    out.extend(sing(&sel(&asks, a25, a75), &sel(&bids, b25, b75)));
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm61 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
