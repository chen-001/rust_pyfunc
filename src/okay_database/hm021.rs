//! hm21「阁中帝子今何在」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm21_阁中帝子今何在/hm21_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔 + 一整天的盘口」各自一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) market = read_market_data_single()          → 全天盘口快照
//!    trades = read_trade_data_single(with_retreat=0) → 丢掉 flag==32 的撤单行
//! 2) 两张表各自 set_index('exchtime').between_time('09:30:00','14:57:00').sort_index()
//!    market 只留 [ask_prc1, bid_prc1, ask_vol1, bid_vol1]
//! 3) market = market.assign(label=1); market.label = market.label.cumsum()
//!    → 按时间排序后的行号，从 1 开始；max_label_here = 最后一行号 = 快照行数 M
//! 4) trades = trades[['volume','flag']]
//!    df = concat([market, trades]).sort_index()   ← 先 market 行、再 trades 行，再按时间排
//!    df.label = df.label.bfill()
//!    → 每笔成交拿到「排序后位于它后面的第一个快照」的 label（包含时刻相同的那个）
//!      排在最后一个快照之后的成交 label 为 NaN，groupby 时被丢掉
//! 5) market = market.set_index('label')
//!    market_ask = market[market.ask_prc1 < market.ask_prc1.shift(1)].ask_vol1
//!    market_bid = market[market.bid_prc1 > market.bid_prc1.shift(1)].bid_vol1
//!    → 卖一价相对上一行下降 / 买一价相对上一行上升 的那些 label 的档位量
//!      （shift 是位置上的上一行，第一行的上一行是 NaN，比较为 False，不计入）
//! 6) trades_66 = trades[trades.flag==66].groupby('label').volume.sum()
//!    trades_83 = trades[trades.flag==83].groupby('label').volume.sum()
//! 7) ask_66 = (market_ask / (trades_66_ask + market_ask)).dropna()
//!    bid_83 = (market_bid / (trades_83_bid + market_bid)).dropna()
//!    → trades_66_ask 是 trades_66 里 label 也在 market_ask 中的那些；
//!      Series 相加按 label 对齐，只在一边出现的 label 结果是 NaN，dropna 丢掉。
//!      所以 ask_66 的 label 集合 = market_ask.index ∩ trades_66.index，按 label 升序。
//! 8) 两边元素个数都 >15 才算，否则 19 个值全是 NaN。
//!    然后 19 个输出（顺序 = names_single = 本文件 names()）：
//!      #0~#2   |ask_66.autocorr(k) - bid_83.autocorr(k)|          k=1,2,3
//!      #3      |ask_66 的 (label[75%位] - label[25%位]) - bid_83 同量|   ← label 序号差
//!      #4~#14  把 ask_66 按**值**升序排（sort_values().reset_index()），
//!              label 列除以 max_label_here 之后取
//!              max/min/mean/std(样本)/sum/skew(修正)/kurt(修正)/
//!              q75/q25/median/iqr(=q75-q25)，
//!              再和 bid_83 的同名统计量做 |差|
//!      #15~#17 |排序后 label 序列的 autocorr(1..3) 之差|
//!      #18     |排序后 label 序列的 (75%位 - 25%位) 之差|
//!
//! names_single（19 个，本文件 names() 必须逐个对上）：
//!   corr1, corr2, corr3, distance, max_label, min_label, mean_label, std_label,
//!   sum_label, skew_label, kurt_label, q75_label, q25_label, median_label, iqr_label,
//!   corr1_label, corr2_label, corr3_label, distance_label
//! names_save（旧 okay 文件里要存进 base 层的 5 个）：
//!   q75_label, max_label, min_label, mean_label, sum_label
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、sort_index 的关键语义：单调时根本不排序
//! ════════════════════════════════════════════════════════════════════
//! pandas 的 DataFrame.sort_index() 走 get_indexer_indexer，里面有一句提前返回：
//!     elif (np.all(ascending) and target.is_monotonic_increasing) ...: return None
//! 也就是**索引已经单调不减时原序返回，一次比较都不做**。
//! 这一条对本系列是决定性的：逐笔和盘口从文件里读出来本来就是按时间递增的，
//! 所以那两次 sort_index 是恒等置换；如果照 numpy quicksort 去重排同时刻的行，
//! 实测 000001@20250303 的 ask_66 里会有一个 label 的量从 18 变成 22（见下）。
//! 只有 concat([market, trades]).sort_index() 那一次，索引是「先全部快照、再全部成交」，
//! 不是单调的，才真正走 numpy 的 quicksort（aquicksort_datetime，见 np_argsort_quicksort）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、一处已知的不可复现项（4 个输出，已实测，写在这里免得以后重复排查）
//! ════════════════════════════════════════════════════════════════════
//! 上表 #15~#18 这 4 个输出依赖 `ask_66.sort_values()` / `bid_83.sort_values()` 的
//! **并列顺序**：ask_66 的值是「档位量 / (该 label 的 66 号成交量 + 档位量)」，
//! 大量取到相同的有理数（600519@20250304 的 828 个值里只有 88 个不同的值）。
//! pandas 的 Series.sort_values() 走 nargsort → ndarray.argsort(kind='quicksort')，
//! numpy 2.x 对 float64 分派到 x86-simd-sort 的 AVX-512 argsort（本机有 avx512f/bw/vl/dq），
//! 并列顺序由 8 通道位网络 / 带压缩存储的分区决定，既不是稳定序，也不是任何可描述的规则；
//! datetime64 才走标量 aquicksort（hm013 复刻的是那一条），float64 走的不是它。
//! 而且这个顺序与机器相关：换成只有 AVX2 的机器，numpy 会走另一套 avx2_argsort，
//! 旧框架自己在这 4 个值上跨机器就不可复现。
//! 本文件的做法：按值做**稳定排序**（值相同时保持原来的 label 升序），写死在
//! `value_order` 里。实测 4 组对照的最大相对差（旧值 vs 稳定排序）：
//!   000001@20250303  #15 0.2363/0.2692  #16 0.0080/0.0743  #17 0.1202/0.0836  #18 0.5864/0.1278
//!   600519@20250304  #15 0.0379/0.0519  #16 0.0165/0.0740  #17 0.0191/0.0719  #18 0.3673/1.3090
//!   000002@20180601  #15 0.0613/0.0558  #16 0.0176/0.0191  #17 0.1020/0.1087  #18 0.6538/1.0862
//!   300750@20231020  #15 0.0174/0.0524  #16 0.0635/0.0361  #17 0.0117/0.0045  #18 0.4091/0.2211
//!   最大相对差 0.909（000001 的 #18）。
//! 为什么不影响因子库：这 4 个都不在 names_save 里；
//! base 层只存 q75_label / max_label / min_label / mean_label / sum_label（#11/#4/#5/#6/#8），
//! 都是对 label 列的顺序无关统计量（顺序变了 max/min/mean/std/sum/skew/kurt/分位数都不变），
//! gap5 入选（q75_label_min_smooth_5、mean_label_mean_smooth_10）与
//! gap1 入选（max_label_min_smooth_5、mean_label_mean_smooth_20、min_label_min_smooth_10、
//! sum_label_max_smooth_5）也全部只由这 5 个加工而来，所以入库的因子值一个都不受影响。
//! 同一类问题在 hm018.rs 里也有记录（float argsort 的并列）。
//!
//! 对照验证（lib/okay_database/verify_series.py --series hm021）：
//!   规范要求的 4 组 20250303:000001 / 20250304:600519 / 20180601:000002 / 20231020:300750
//!   → 每组 19 个值里正好只有上面这 4 个位置不一致，其余 15 个最大相对差 4.4e-8（= f32 输出精度）；
//!   另外补测 7 组（20180601:000001、20231020:600519、20250303:300750、20200713:000001、
//!   20250303:688981、20190102:002415、20240102:300059）也是同样结论，前 15 个最大相对差 4.9e-8。
//!   20160201:000002 逐笔文件不存在，旧代码返回 None、本实现同样返回 None（两边都落 NaN），一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、精度：逐笔与盘口都按 f64 自己解析 CSV
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤按规范自己 filter flag != 32（不靠 read_trade_fast_inner 的 with_retreat：
//!   库里那条判断按字节比 "32"，2025 年的明细写的是 "32.0"，过滤不生效）。
//! · 盘口只用到 ask_prc1/bid_prc1 做**大小比较**，f32 也能应付（相邻价位差 0.01，
//!   远大于 f32 在千元价位上的分辨率）；但 ask_vol1/bid_vol1 在 #0~#14 里参与
//!   「档位量 / (成交量 + 档位量)」这种比值，而且最后是拿两个比值相减（#0~#3 是
//!   相关系数之差、#4~#14 是统计量之差），差值量级可能只有 1e-4 上下，
//!   f32 输入会把相对误差放大到无法满足 1e-4 的门槛。
//!   所以逐笔与盘口都在本文件里按 f64 解析 CSV（列索引与 fast_csv_reader 的常量一致），
//!   中间全部用 f64 算，只在返回前一次性降到 f32。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、窗口用微秒闭区间
//! ════════════════════════════════════════════════════════════════════
//! 旧代码的 exchtime 是 pandas Timestamp（deal_data 里 exchtime/1e6 秒 + 1970-01-01 08:00），
//! between_time 两端都含、精确到微秒，不能按整秒截断（14:57:00.3 必须落在窗口外）。
//! 这里用当日微秒数做闭区间判断：pandas 那条路径把微秒转 ns 时会有不到 0.4 微秒的
//! 取整漂移，但它是单调映射、不同微秒值间隔 1000ns 不会被压到一起，
//! 所以排序次序、并列关系、以及和窗口边界的比较结果都与逐微秒判断一致。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;

/// 盘口 CSV 的列索引（0 基）：ask_prc1=21, ask_vol1=22, bid_prc1=23, bid_vol1=24
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_ASK_PRC1: usize = 21;
const MKT_COL_ASK_VOL1: usize = 22;
const MKT_COL_BID_PRC1: usize = 23;
const MKT_COL_BID_VOL1: usize = 24;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

pub const N_OUT: usize = 19;

/// 每个时段的元素个数下限：旧代码是 shape[0] > 15
const MIN_LEN: usize = 15;

pub fn names() -> Vec<String> {
    [
        "corr1",
        "corr2",
        "corr3",
        "distance",
        "max_label",
        "min_label",
        "mean_label",
        "std_label",
        "sum_label",
        "skew_label",
        "kurt_label",
        "q75_label",
        "q25_label",
        "median_label",
        "iqr_label",
        "corr1_label",
        "corr2_label",
        "corr3_label",
        "distance_label",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[inline]
fn day_us(t: i64) -> i64 {
    t.rem_euclid(DAY_US)
}

/// between_time('09:30:00','14:57:00')：两端都含，微秒精度
#[inline]
fn in_span(t: i64) -> bool {
    let d = day_us(t);
    d >= LD_0930 && d <= LD_1457
}

// ============================================================================
// CSV 的 f64 读取
// ============================================================================

/// 一条逐笔成交（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    volume: f64,
    flag: f64,
}

/// 一条盘口快照（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct MktRow {
    time_us: i64,
    ask_prc1: f64,
    bid_prc1: f64,
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

/// 定位某个日期的逐笔 / 盘口文件（kind 取 "transaction" 或 "market_data"）
fn level2_file(code: &str, date: i64, kind: &str) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_{}.csv", code, date, kind);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join(kind).join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 拆出一行的第 `want` 个字段（0 基）。找不到给空切片。
#[inline]
fn field(line: &[u8], want: usize) -> &[u8] {
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col == want {
                return &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col == want {
        return &line[start..];
    }
    &[]
}

/// 读 transaction CSV（f64）。行集与 pandas 读整张表一致（不过滤 exchtime==0 的行，
/// 它会落在窗口外被 between_time 丢掉，保留不影响结果）。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let path = level2_file(code, date, "transaction")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let b = line.as_bytes();
        out.push(TradeRow {
            time_us: parse_i64_at(field(b, TRD_COL_EXCHTIME)) + 8 * 3600 * SEC_US,
            volume: parse_f64_at(field(b, TRD_COL_VOLUME)),
            flag: parse_f64_at(field(b, TRD_COL_FLAG)),
        });
    }
    Some(out)
}

/// 读 market_data CSV（f64）。行集与 pandas 读整张表的行集一致。
fn read_market_f64(code: &str, date: i64) -> Option<Vec<MktRow>> {
    let path = level2_file(code, date, "market_data")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<MktRow> = Vec::with_capacity(content.len() / 200 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let b = line.as_bytes();
        out.push(MktRow {
            time_us: parse_i64_at(field(b, MKT_COL_EXCHTIME)) + 8 * 3600 * SEC_US,
            ask_prc1: parse_f64_at(field(b, MKT_COL_ASK_PRC1)),
            ask_vol1: parse_f64_at(field(b, MKT_COL_ASK_VOL1)),
            bid_prc1: parse_f64_at(field(b, MKT_COL_BID_PRC1)),
            bid_vol1: parse_f64_at(field(b, MKT_COL_BID_VOL1)),
        });
    }
    Some(out)
}

// ============================================================================
// numpy aquicksort_datetime 的逐行复刻（只有 concat 那一次排序会用到）
// ============================================================================

#[inline]
fn msb(mut u: u64) -> i32 {
    let mut d = 0i32;
    while u > 1 {
        u >>= 1;
        d += 1;
    }
    d
}

/// numpy aheapsort_（索引版堆排序），只在与 quicksort 深度耗尽时才会用到
fn aheapsort(v: &[i64], idx: &mut [usize], lo: usize, n: usize) {
    if n < 2 {
        return;
    }
    let mut n = n;
    let mut l = n >> 1;
    while l > 0 {
        let tmp = idx[lo + l - 1];
        let mut i = l;
        let mut j = l << 1;
        while j <= n {
            if j < n && v[idx[lo + j - 1]] < v[idx[lo + j]] {
                j += 1;
            }
            if v[tmp] < v[idx[lo + j - 1]] {
                idx[lo + i - 1] = idx[lo + j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        idx[lo + i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = idx[lo + n - 1];
        idx[lo + n - 1] = idx[lo];
        n -= 1;
        let mut i = 1usize;
        let mut j = 2usize;
        while j <= n {
            if j < n && v[idx[lo + j - 1]] < v[idx[lo + j]] {
                j += 1;
            }
            if v[tmp] < v[idx[lo + j - 1]] {
                idx[lo + i - 1] = idx[lo + j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        idx[lo + i - 1] = tmp;
    }
}

/// numpy `aquicksort_<datetime_tag>`（npysort/quicksort.cpp）的逐行复刻。
///
/// 平局顺序必须与它一致：pandas 单键 sort_values / sort_index 落在这个不稳定分区上，
/// 同一时刻的成交与快照谁在前决定了成交取哪个 label。
fn np_argsort_quicksort(v: &[i64]) -> Vec<usize> {
    let num = v.len();
    let mut idx: Vec<usize> = (0..num).collect();
    if num <= 1 {
        return idx;
    }
    const SMALL_QUICKSORT: usize = 15;

    let mut stack: Vec<(usize, usize, i32)> = Vec::with_capacity(64);
    let mut pl = 0usize;
    let mut pr = num - 1;
    let mut cdepth: i32 = msb(num as u64) * 2;

    loop {
        if cdepth < 0 {
            aheapsort(v, &mut idx, pl, pr - pl + 1);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if v[idx[pm]] < v[idx[pl]] {
                    idx.swap(pm, pl);
                }
                if v[idx[pr]] < v[idx[pm]] {
                    idx.swap(pr, pm);
                }
                if v[idx[pm]] < v[idx[pl]] {
                    idx.swap(pm, pl);
                }
                let vp = v[idx[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                idx.swap(pm, pj);
                loop {
                    pi += 1;
                    while v[idx[pi]] < vp {
                        pi += 1;
                    }
                    pj -= 1;
                    while vp < v[idx[pj]] {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    idx.swap(pi, pj);
                }
                let pk = pr - 1;
                idx.swap(pi, pk);
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth - 1));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth - 1));
                    pl = pi + 1;
                }
                cdepth -= 1;
            }
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = idx[pi];
                let vp = v[vi];
                let mut pj = pi;
                let mut pk = pi - 1;
                while pj > pl && vp < v[idx[pk]] {
                    idx[pj] = idx[pk];
                    pj -= 1;
                    pk -= 1;
                }
                idx[pj] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            Some((npl, npr, d)) => {
                pl = npl;
                pr = npr;
                cdepth = d;
            }
            None => break,
        }
    }
    idx
}

/// pandas DataFrame.sort_index() 的置换。
///
/// get_indexer_indexer 里有一句提前返回：索引**已经单调不减**时原序返回、不排序。
/// 逐笔与盘口从文件读出来本来就有序，所以那两次是恒等置换；
/// 只有 concat 之后的那一次索引不单调，才真正走 numpy quicksort。
fn sort_index_perm(times: &[i64]) -> Vec<usize> {
    let monotonic = times.windows(2).all(|w| w[0] <= w[1]);
    if monotonic {
        (0..times.len()).collect()
    } else {
        np_argsort_quicksort(times)
    }
}

// ============================================================================
// 统计量（对齐 pandas / numpy）
// ============================================================================

/// np.corrcoef(a, b)[0, 1]：两遍算法（先减均值，再点积 / (n-1)），
/// 与 pandas Series.corr / autocorr 的 nancorr 一致（min_periods 默认 1）。
fn np_corrcoef(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 || n != b.len() {
        return f64::NAN;
    }
    let nf = n as f64;
    let ma = a.iter().sum::<f64>() / nf;
    let mb = b.iter().sum::<f64>() / nf;
    let (mut cxx, mut cyy, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = a[i] - ma;
        let dy = b[i] - mb;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    let f = nf - 1.0;
    let vxy = cxy / f;
    let d = (cxx / f).sqrt() * (cyy / f).sqrt();
    if d != 0.0 {
        vxy / d
    } else {
        f64::NAN
    }
}

/// Series.autocorr(lag) = self.corr(self.shift(lag))，shift 出的 NaN 对会被丢掉，
/// 等价于 pearson(x[0..n-lag], x[lag..n])。
fn autocorr(s: &[f64], lag: usize) -> f64 {
    let n = s.len();
    if lag >= n {
        return f64::NAN;
    }
    np_corrcoef(&s[..n - lag], &s[lag..])
}

/// Series.std()：样本标准差（ddof=1）
fn std_s(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = v.iter().sum::<f64>() / n as f64;
    let ss = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
    (ss / (n as f64 - 1.0)).sqrt()
}

/// Series.skew()（bias=False）：pandas nanops.nanskew
///   result = (n * sqrt(n-1) / (n-2)) * (m3 / m2^1.5)，m2/m3 是二阶、三阶中心矩之和
///   m2 归零时给 0，n < 3 给 NaN
fn skew_s(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = v.iter().sum::<f64>() / nf;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for x in v.iter() {
        let d = x - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    (nf * (nf - 1.0).sqrt() / (nf - 2.0)) * (m3 / m2.powf(1.5))
}

/// Series.kurt()（bias=False）：pandas nanops.nankurt
///   result = n(n+1)(n-1)m4 / ((n-2)(n-3)m2^2) - 3(n-1)^2/((n-2)(n-3))
///   n < 4 给 NaN，分母归零给 0
fn kurt_s(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = v.iter().sum::<f64>() / nf;
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for x in v.iter() {
        let d = x - m;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    let mut num = nf * (nf + 1.0) * (nf - 1.0) * m4;
    let mut den = (nf - 2.0) * (nf - 3.0) * m2 * m2;
    // pandas 的 _zero_out_fperr：绝对值 < 1e-14 视为 0
    if num.abs() < 1e-14 {
        num = 0.0;
    }
    if den.abs() < 1e-14 {
        den = 0.0;
    }
    if den == 0.0 {
        return 0.0;
    }
    let adj = 3.0 * (nf - 1.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0));
    num / den - adj
}

/// numpy np.percentile(method='linear') 的内插：t >= 0.5 时走 b - (b-a)*(1-t)，
/// 与 numpy `_lerp` 的分支一致。入参必须是已经升序排好的序列。
fn quantile_linear(sorted_v: &[f64], q: f64) -> f64 {
    let n = sorted_v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted_v[0];
    }
    let idx = (n as f64 - 1.0) * q;
    let lo = idx.floor();
    let hi = idx.ceil();
    if lo == hi {
        return sorted_v[lo as usize];
    }
    let a = sorted_v[lo as usize];
    let b = sorted_v[hi as usize];
    let t = idx - lo;
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

/// 一个比值序列（label 升序）的 14 个顺序无关/相关统计量，外加 label 序号差。
/// 顺序与旧代码 #3~#18 一致：
///   [0] label 序号差（排序前），[1..11] max..iqr（按值排序后），
///   [12..14] 按值排序后 label 序列的 autocorr(1..3)，[15] 排序后 label 的序号差
fn series_stats(labels: &[i64], values: &[f64], max_label: f64) -> [f64; 16] {
    let n = values.len();
    let mut out = [f64::NAN; 16];
    // label 序号差（用排序前的 label 升序序列）
    out[0] = (labels[(n as f64 * 0.75) as usize] - labels[(n as f64 * 0.25) as usize]) as f64;

    // 按值升序排（稳定：值相同时保持 label 升序），再把 label 除以 max_label
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        values[x]
            .partial_cmp(&values[y])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let lab: Vec<f64> = order
        .iter()
        .map(|&i| labels[i] as f64 / max_label)
        .collect();

    let mut sorted = lab.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &x in lab.iter() {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
        sum += x;
    }
    out[1] = hi; // max
    out[2] = lo; // min
    let mean = sum / n as f64;
    out[3] = mean;
    out[4] = std_s(&lab);
    out[5] = sum;
    out[6] = skew_s(&lab);
    out[7] = kurt_s(&lab);
    let q75 = quantile_linear(&sorted, 0.75);
    let q25 = quantile_linear(&sorted, 0.25);
    out[8] = q75;
    out[9] = q25;
    out[10] = quantile_linear(&sorted, 0.5);
    out[11] = q75 - q25;
    out[12] = autocorr(&lab, 1);
    out[13] = autocorr(&lab, 2);
    out[14] = autocorr(&lab, 3);
    out[15] = lab[(n as f64 * 0.75) as usize] - lab[(n as f64 * 0.25) as usize];
    out
}

fn metrics(ask_66: &[(i64, f64)], bid_83: &[(i64, f64)], max_label: f64) -> [f64; N_OUT] {
    let mut out = [f64::NAN; N_OUT];
    if ask_66.len() <= MIN_LEN || bid_83.len() <= MIN_LEN {
        return out;
    }
    let av: Vec<f64> = ask_66.iter().map(|x| x.1).collect();
    let bv: Vec<f64> = bid_83.iter().map(|x| x.1).collect();
    for k in 1..=3usize {
        out[k - 1] = (autocorr(&av, k) - autocorr(&bv, k)).abs();
    }
    let al: Vec<i64> = ask_66.iter().map(|x| x.0).collect();
    let bl: Vec<i64> = bid_83.iter().map(|x| x.0).collect();
    let sa = series_stats(&al, &av, max_label);
    let sb = series_stats(&bl, &bv, max_label);
    // #3：排序前的 label 序号差
    out[3] = (sa[0] - sb[0]).abs();
    // #4~#14：max/min/mean/std/sum/skew/kurt/q75/q25/median/iqr（sa[1..12]）
    for i in 0..11 {
        out[4 + i] = (sa[1 + i] - sb[1 + i]).abs();
    }
    // #15~#17：排序后 label 序列的 autocorr
    for i in 0..3 {
        out[15 + i] = (sa[12 + i] - sb[12 + i]).abs();
    }
    // #18：排序后 label 的序号差
    out[18] = (sa[15] - sb[15]).abs();
    out
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades_all = read_trade_f64(code, date)?;
    let market_all = read_market_f64(code, date)?;

    // with_retreat=0：丢掉 flag==32 的撤单行。自己按数值过滤（"32.0" 也要能过滤掉）；
    // 空 flag 解析成 NaN，NaN != 32 为真，与 pandas 的 data[data["flag"] != 32] 一致。
    // 窗口过滤在读取之后，与旧代码的 between_time 顺序一致（两者都是先撤单后切窗口，
    // 结果集合等价）。
    let t: Vec<TradeRow> = trades_all
        .into_iter()
        .filter(|r| r.flag != 32.0 && in_span(r.time_us))
        .collect();
    let m: Vec<MktRow> = market_all
        .into_iter()
        .filter(|r| in_span(r.time_us))
        .collect();
    if t.is_empty() || m.is_empty() {
        return None;
    }

    // sort_index()：单调时是恒等置换（见文件头第二节）
    let m_times: Vec<i64> = m.iter().map(|r| r.time_us).collect();
    let m_perm = sort_index_perm(&m_times);
    let m: Vec<MktRow> = m_perm.iter().map(|&i| m[i]).collect();
    let t_times: Vec<i64> = t.iter().map(|r| r.time_us).collect();
    let t_perm = sort_index_perm(&t_times);
    let t: Vec<TradeRow> = t_perm.iter().map(|&i| t[i]).collect();

    let m_cnt = m.len();
    let t_cnt = t.len();

    // concat([market, trades]).sort_index() 然后 bfill：
    // 排序后往前扫，成交拿到它后面第一个快照的 label（没有则是 NaN）
    let mut times: Vec<i64> = Vec::with_capacity(m_cnt + t_cnt);
    times.extend(m.iter().map(|r| r.time_us));
    times.extend(t.iter().map(|r| r.time_us));
    let perm = sort_index_perm(&times);
    let mut labels: Vec<i64> = vec![-1; t_cnt]; // -1 代表 NaN
    let mut cur: i64 = -1;
    for &i in perm.iter().rev() {
        if i < m_cnt {
            cur = i as i64 + 1;
        } else {
            labels[i - m_cnt] = cur;
        }
    }

    // market_ask / market_bid：卖一价下降 / 买一价上升的那些行（label 从 1 开始）
    let mut mkt_ask = vec![f64::NAN; m_cnt + 1];
    let mut mkt_bid = vec![f64::NAN; m_cnt + 1];
    for k in 1..m_cnt {
        if m[k].ask_prc1 < m[k - 1].ask_prc1 {
            mkt_ask[k + 1] = m[k].ask_vol1;
        }
        if m[k].bid_prc1 > m[k - 1].bid_prc1 {
            mkt_bid[k + 1] = m[k].bid_vol1;
        }
    }

    // 按 label 汇总 66 号买 / 83 号卖的成交量
    let mut v66: HashMap<i64, f64> = HashMap::new();
    let mut v83: HashMap<i64, f64> = HashMap::new();
    for i in 0..t_cnt {
        let lab = labels[i];
        if lab < 0 {
            continue;
        }
        let f = t[i].flag;
        if f == 66.0 {
            *v66.entry(lab).or_insert(0.0) += t[i].volume;
        } else if f == 83.0 {
            *v83.entry(lab).or_insert(0.0) += t[i].volume;
        }
    }

    // ask_66 = market_ask / (trades_66 + market_ask)，按 label 升序，两边都有的 label 才留
    let mut ask_66: Vec<(i64, f64)> = Vec::new();
    for lab in 1..=m_cnt {
        let av = mkt_ask[lab];
        if av.is_nan() {
            continue;
        }
        if let Some(&s) = v66.get(&(lab as i64)) {
            ask_66.push((lab as i64, av / (s + av)));
        }
    }
    let mut bid_83: Vec<(i64, f64)> = Vec::new();
    for lab in 1..=m_cnt {
        let bv = mkt_bid[lab];
        if bv.is_nan() {
            continue;
        }
        if let Some(&s) = v83.get(&(lab as i64)) {
            bid_83.push((lab as i64, bv / (s + bv)));
        }
    }

    let out = metrics(&ask_66, &bid_83, m_cnt as f64);
    Some(out.iter().map(|&v| v as f32).collect())
}

/// hm21 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
