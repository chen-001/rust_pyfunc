//! hm039 边城新生 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm39_边城新生/hm39_okay.py
//! get_many=0 / read_full=0 / change_time=1，逐笔与盘口各自一整张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、时间轴（最容易错的一步）
//! ════════════════════════════════════════════════════════════════════
//! 旧代码 new_time()：把 exchtime 当索引，切 [09:30:00,11:30:00] 与 [13:00:00,14:57:00]
//! 两段，**下午段整体减 91 分钟**，然后 concat（上午段在前）。于是所有时刻都落在
//! 09:30:00 ~ 13:26:00 这个「平移后的钟点」上，逐笔和盘口用同一套平移。
//! 后面所有窗口（silent/open30/morning/afternoon/close30）都是在这个平移钟点上切的。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、每个 (param, future) 的算式
//! ════════════════════════════════════════════════════════════════════
//! 1) price_max = rp.max_range_loop(price, allow_equal=False)
//!    price_min = rp.min_range_loop(price, allow_equal=False)   （在**平移后**的整段上算）
//!    这两个是「到左边最近一个价格 >= / <= 当前价的行距」，整数列。
//! 2) 候选 = 该列 > 该列.rolling_future("{future}s").max()
//!    rolling_future 不含当前点，窗口是 (i, i+window]；NaN 比较为 False。
//!    注意它的滑动窗口实现假定索引单调，而这里的索引（上午段接下午段）**不单调**，
//!    库里的实现会复用上一轮的 window_end，复刻时必须照抄那段逻辑，不能换成正确的窗口最大值。
//! 3) 候选按该列**降序**排，取前 param 行。
//!    ！！这里的降序并列顺序由 numpy 决定（见第五节），不是稳定排序。
//! 4) 六个时段各切一刀，得到 trade_max（来自 price_max 的候选）与 trade_min。
//! 5) 每个时段：asks 侧用 trade_max 的行、bids 侧用 trade_min 的行。
//!    对每一行 (t1, p1)：before = 该侧盘口里 exchtime∈[t1-f, t1] 且 price==p1 的 vol 均值；
//!                       after  = 同侧 [t1, t1+f] 的 vol 均值；ratio = (after-before)/before。
//!    ！！f 只作用于 asks 侧：asks 用传入的 future_minute（1 或 3），
//!      bids 侧**写死 3 分钟**，与传入值无关（旧代码就这么写的）。
//!    再分正负求和：正 ratio 求和、负 ratio 求和（NaN 两边都不进）。空选择 sum 记 0.0。
//! 6) 若该时段的 trade_max 或 trade_min 为空 → 该时段 4 个值全 NaN（f=1、f=3 都是）。
//!
//! 每段 8 个值 = [asks_pos, bids_pos, asks_neg, bids_neg]（f=1 四个）+
//!              [asks_pos, bids_pos, asks_neg, bids_neg]（f=3 四个）
//! sing(future) = allday + silent + open30 + morning + afternoon + close30，共 48 个
//! one(param)   = sing(30) + sing(60)，共 96 个
//! 最终值        = one(20) + one(50) + one(100)，共 288 个
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、names_single（本文件 names() 必须逐个对上，共 288 个）
//! ════════════════════════════════════════════════════════════════════
//! 旧代码的 names 是按 [param50,param100] × [future30,future60,future180] × 六个时段
//! × [minute1,minute3] × 四个指标 拼出来的；而**值**是按 [param20,50,100] ×
//! [future30,60] 算出来的。两边口径本来就不一致（future180 那一档没有对应的值，
//! param20 那一档没有对应的名字），这是旧代码的原样，names() 照抄名字顺序、
//! compute_stock 照抄值顺序，位置一一对应即可。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、读取上的两个坑（与 hm013 一致）
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤按规范自己 filter flag != 32（不能靠 read_trade_fast_inner 的 with_retreat：
//!   库里那条判断按字节比 "32"，2025 年的明细写的是 "32.0"，过滤不生效）。
//! · 盘口先 df[df.last_prc != 0]，再把 10 档拆成长表、丢掉 price==0 的档；NaN 价不算 0，
//!   留在表里也永远匹配不上（这里用 !(p == 0.0) 表达）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、降序排序的并列顺序：逐行复刻 numpy 的 x86-simd-sort argsort
//! ════════════════════════════════════════════════════════════════════
//! `trade[mask].reset_index().sort_values(col, ascending=False).head(param)` 里的
//! sort_values 是单键 → pandas nargsort → np.argsort(kind="quicksort")。
//! numpy 2.x 在支持 AVX512 的机器上对 int64 走 **x86-simd-sort**（不是 npysort 的
//! introsort），并列顺序由它的「8 通道位元排序网络 + 分区」决定。
//! 实测：改用一个稳定排序，000002@20180601 的 288 个值里有 8 个相对误差到 12%
//! （因为 100 个名额里挤进了并列组里不同的那一行）。所以这里把
//! numpy/_core/src/npysort/x86-simd-sort/src/{xss-common-argsort.h,xss-network-keyvaluesort.hpp,
//! xss-reg-networks.hpp} 的 int64 路径用标量模拟 8 通道的方式逐行搬了过来。
//! 复刻结果已和 numpy 2.3.5（本机 AVX512_SKX）对拍：48 个真实候选数组（4 组验收数据、
//! 尺寸 138~510）的 argsort 结果**完全一致**。
//! 唯一的近似：x86-simd-sort 在迭代次数耗尽时会退回 std::sort（libstdc++ 的 introsort），
//! 这里用「按值稳定排序」代替。实测这 48 个数组一次都没触发。

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;

// ============================================================================
// 时间常量
// ============================================================================

const SEC_NS: i64 = 1_000_000_000;
const MIN_NS: i64 = 60 * SEC_NS;
const DAY_NS: i64 = 86_400 * SEC_NS;
/// 下午段整体减 91 分钟（旧代码 df2.index - 91min）
const SHIFT_NS: i64 = 91 * MIN_NS;
/// 北京时间偏移：exchtime 是 epoch 微秒，加 8 小时才是当天钟点
const BJ_OFFSET_NS: i64 = 8 * 3600 * SEC_NS;

const fn hm(h: i64, m: i64) -> i64 {
    (h * 3600 + m * 60) * SEC_NS
}

/// 平移后钟点上的原始切分区间（闭区间，微秒级精度）
const MORNING_LO: i64 = hm(9, 30);
const MORNING_HI: i64 = hm(11, 30);
const AFTERNOON_LO: i64 = hm(13, 0);
const AFTERNOON_HI: i64 = hm(14, 57);

/// 六个时段（平移后的钟点闭区间），顺序必须与旧代码 res_* 的拼接顺序一致
const SESSIONS: [(i64, i64); 6] = [
    (0, DAY_NS),            // allday（旧代码不切窗口，全部行）
    (hm(10, 5), hm(12, 51)), // silent
    (hm(9, 30), hm(10, 0)),  // open30
    (hm(9, 30), hm(11, 30)), // morning
    (hm(11, 59), hm(13, 27)), // afternoon
    (hm(13, 0), hm(13, 27)),  // close30
];

/// 三个 param 与两个 future 的取值顺序（值顺序 [20,50,100] × [30,60]）
const PARAMS: [usize; 3] = [20, 50, 100];
const FUTURES: [i64; 2] = [30, 60];

/// 四个指标的顺序：asks_pos, bids_pos, asks_neg, bids_neg
const PER_SESSION: usize = 8;
pub const N_OUT: usize = 3 * 2 * 6 * PER_SESSION; // 288

const METRIC_NAMES: [&str; 4] = ["asks_pos_sum", "bids_pos_sum", "asks_neg_sum", "bids_neg_sum"];
const MINUTE_NAMES: [&str; 2] = ["minute1", "minute3"];
const SESSION_NAMES: [&str; 6] = ["allday", "silent", "open30", "morning", "afternoon", "close30"];
const FUTURE_NAMES: [&str; 3] = ["future30", "future60", "future180"];
const PARAM_NAMES: [&str; 2] = ["param50", "param100"];

/// 旧代码 `names` 变量原样：param（外层，两个）× future（三个）× 时段 × 分钟 × 指标
pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for p in PARAM_NAMES.iter() {
        for f in FUTURE_NAMES.iter() {
            for s in SESSION_NAMES.iter() {
                for m in MINUTE_NAMES.iter() {
                    for metric in METRIC_NAMES.iter() {
                        out.push(format!("{}_{}_{}_{}_{}", metric, m, s, f, p));
                    }
                }
            }
        }
    }
    out
}

// ============================================================================
// CSV 读取（f64 精度，列索引与 fast_csv_reader 的常量一致）
// ============================================================================

const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_ASK_PRC_BASE: usize = 21; // ask_prc1=21, ask_vol1=22, bid_prc1=23, bid_vol1=24, ...
const MKT_NCOL: usize = 61;

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

/// 把一行 CSV 拆成最多 n 个字段
fn split_fields<'a>(line: &'a [u8], f: &mut [&'a [u8]]) {
    let n = f.len();
    for x in f.iter_mut() {
        *x = &line[..0];
    }
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < n {
                f[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < n {
        f[col] = &line[start..];
    }
}

fn find_file(code: &str, date: i64, kind: &str, suffix: &str) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_{}.csv", code, date, suffix);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join(kind).join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 当天钟点（纳秒，含 8 小时偏移，落在 [0, DAY_NS) 内）
#[inline]
fn clock_ns(exchtime_us: i64) -> i64 {
    (exchtime_us * 1000 + BJ_OFFSET_NS).rem_euclid(DAY_NS)
}

/// 逐笔：返回 (平移后钟点纳秒, 价格)，顺序 = 上午段 + 下午段（与 old new_time 一致）
fn read_trade_new_time(code: &str, date: i64) -> Option<Vec<(i64, f64)>> {
    let path = find_file(code, date, "transaction", "transaction")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;
    let mut out: Vec<(i64, f64)> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next();
    let mut first_afternoon: Vec<(i64, f64)> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        split_fields(bytes, &mut f);
        // 撤单过滤：flag != 32（"32.0" 也要能滤掉）
        if parse_i64_at(f[TRD_COL_FLAG]) == 32 {
            continue;
        }
        let c = clock_ns(parse_i64_at(f[TRD_COL_EXCHTIME]));
        let price = parse_f64_at(f[TRD_COL_PRICE]);
        if c >= MORNING_LO && c <= MORNING_HI {
            out.push((c, price));
        } else if c >= AFTERNOON_LO && c <= AFTERNOON_HI {
            first_afternoon.push((c - SHIFT_NS, price));
        }
    }
    out.extend(first_afternoon);
    Some(out)
}

/// 盘口：先 df[last_prc != 0]，再把 10 档拆成长表（丢掉 price==0 的档），并做下午平移。
/// 返回 (asks 长表, bids 长表)，每行是 (平移后钟点纳秒, 价, 量)。
fn read_market_new_time(code: &str, date: i64) -> Option<(Vec<(i64, f64, f64)>, Vec<(i64, f64, f64)>)> {
    let path = find_file(code, date, "market_data", "market_data")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;
    let mut asks: Vec<(i64, f64, f64)> = Vec::new();
    let mut bids: Vec<(i64, f64, f64)> = Vec::new();
    let mut lines = content.lines();
    lines.next();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; MKT_NCOL] = [&[][..]; MKT_NCOL];
        split_fields(bytes, &mut f);
        let last_prc = parse_f64_at(f[MKT_COL_LAST_PRC]);
        // pandas 的 `!= 0`：NaN 不等于 0，所以 NaN 行会留下
        if last_prc == 0.0 {
            continue;
        }
        let c0 = clock_ns(parse_i64_at(f[MKT_COL_EXCHTIME]));
        let c = if c0 >= MORNING_LO && c0 <= MORNING_HI {
            c0
        } else if c0 >= AFTERNOON_LO && c0 <= AFTERNOON_HI {
            c0 - SHIFT_NS
        } else {
            continue;
        };
        for k in 0..10 {
            let base = MKT_ASK_PRC_BASE + k * 4;
            let ap = parse_f64_at(f[base]);
            let av = parse_f64_at(f[base + 1]);
            let bp = parse_f64_at(f[base + 2]);
            let bv = parse_f64_at(f[base + 3]);
            // 旧代码是 asks[asks.price != 0]（NaN 价会留下，但永远匹配不上）
            if !(ap == 0.0) {
                asks.push((c, ap, av));
            }
            if !(bp == 0.0) {
                bids.push((c, bp, bv));
            }
        }
    }
    Some((asks, bids))
}

// ============================================================================
// max_range_loop / min_range_loop（抄 rp 的实现）
// ============================================================================

fn max_range_loop(s: &[f64], allow_equal: bool) -> Vec<i64> {
    let mut out = Vec::with_capacity(s.len());
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..s.len() {
        let mut done = false;
        while let Some(&j) = stack.last() {
            if (!allow_equal && s[j] >= s[i]) || (allow_equal && s[j] > s[i]) {
                out.push((i - j) as i64);
                done = true;
                break;
            }
            stack.pop();
        }
        if !done {
            out.push(i as i64 + 1);
        }
        stack.push(i);
    }
    out
}

fn min_range_loop(s: &[f64], allow_equal: bool) -> Vec<i64> {
    let mut out = Vec::with_capacity(s.len());
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..s.len() {
        let mut done = false;
        while let Some(&j) = stack.last() {
            if (!allow_equal && s[j] <= s[i]) || (allow_equal && s[j] < s[i]) {
                out.push((i - j) as i64);
                done = true;
                break;
            }
            stack.pop();
        }
        if !done {
            out.push(i as i64 + 1);
        }
        stack.push(i);
    }
    out
}

// ============================================================================
// rp.rolling_future(window).max()（rust_pyfunc/src/pandas_ext/mod.rs 的 "max" 分支）
// ============================================================================
//
// 逐行复刻：窗口是 (i, i+window]（include_current=false），
// 而且 window_end 是跨 i 复用的——索引不单调时它会带回已经出窗的元素。
// 旧代码的索引（上午段接下午段）正好不单调，所以必须照抄，不能写成「正确的」窗口最大值。

fn rolling_future_max(times: &[i64], values: &[f64], window_ns: i64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(n);
    let mut window_end = 0usize;
    for i in 0..n {
        if i > 0 && window_end > i {
            while let Some(&f) = deque.front() {
                if f < i + 1 {
                    deque.pop_front();
                } else {
                    break;
                }
            }
        } else {
            window_end = i + 1;
            deque.clear();
        }
        while window_end < n && times[window_end] - times[i] <= window_ns {
            if window_end >= i + 1 {
                while let Some(&b) = deque.back() {
                    if values[b] <= values[window_end] {
                        deque.pop_back();
                    } else {
                        break;
                    }
                }
                deque.push_back(window_end);
            }
            window_end += 1;
        }
        if let Some(&f) = deque.front() {
            result[i] = values[f];
        }
    }
    result
}

// ============================================================================
// numpy x86-simd-sort argsort(int64) 的逐行复刻
// ============================================================================
//
// 8 通道寄存器用定长数组表示；掩码用 u8（bit l 对应通道 l）。
// 所有比较都是 int64 有符号比较。

const LANES: usize = 8;
const I64MAX: i64 = i64::MAX;

#[inline]
fn swap_perm(scale: usize) -> [usize; LANES] {
    match scale {
        2 => [1, 0, 3, 2, 5, 4, 7, 6],
        4 => [2, 3, 0, 1, 6, 7, 4, 5],
        _ => [4, 5, 6, 7, 0, 1, 2, 3],
    }
}

#[inline]
fn reverse_perm(scale: usize) -> [usize; LANES] {
    match scale {
        2 => [1, 0, 3, 2, 5, 4, 7, 6],
        4 => [3, 2, 1, 0, 7, 6, 5, 4],
        _ => [7, 6, 5, 4, 3, 2, 1, 0],
    }
}

#[inline]
fn perm_k(k: &[i64; LANES], p: &[usize; LANES]) -> [i64; LANES] {
    let mut o = [0i64; LANES];
    for l in 0..LANES {
        o[l] = k[p[l]];
    }
    o
}

#[inline]
fn perm_i(i: &[i64; LANES], p: &[usize; LANES]) -> [i64; LANES] {
    perm_k(i, p)
}

/// 只按键的 cmp_merge：0 -> min，1 -> max
fn cmp_merge_k(k1: &[i64; LANES], k2: &[i64; LANES], mask: u8) -> [i64; LANES] {
    let mut o = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        o[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
    }
    o
}

/// 带下标的 cmp_merge（xss-network-keyvaluesort.hpp）
fn cmp_merge_ki(
    k1: &[i64; LANES],
    i1: &[i64; LANES],
    k2: &[i64; LANES],
    i2: &[i64; LANES],
    mask: u8,
) -> ([i64; LANES], [i64; LANES]) {
    let mut ko = [0i64; LANES];
    let mut io = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        ko[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
        // 结果来自 in1 就保留下标 1，否则用下标 2
        io[l] = if ko[l] == a { i1[l] } else { i2[l] };
    }
    (ko, io)
}

/// COEX：稳定的比较交换
fn coex(
    k1: &[i64; LANES],
    i1: &[i64; LANES],
    k2: &[i64; LANES],
    i2: &[i64; LANES],
) -> ([i64; LANES], [i64; LANES], [i64; LANES], [i64; LANES]) {
    let mut ko1 = [0i64; LANES];
    let mut ko2 = [0i64; LANES];
    let mut io1 = [0i64; LANES];
    let mut io2 = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        ko1[l] = lo;
        ko2[l] = hi;
        let eq = lo == a;
        io1[l] = if eq { i1[l] } else { i2[l] };
        io2[l] = if eq { i2[l] } else { i1[l] };
    }
    (ko1, io1, ko2, io2)
}

fn sort_reg_8lanes(k: &[i64; LANES], i: &[i64; LANES]) -> ([i64; LANES], [i64; LANES]) {
    let (mut k, mut i) = cmp_merge_ki(k, i, &perm_k(k, &reverse_perm(2)), &perm_i(i, &reverse_perm(2)), 0xAA);
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &reverse_perm(4)), &perm_i(&i, &reverse_perm(4)), 0xCC);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &swap_perm(2)), &perm_i(&i, &swap_perm(2)), 0xAA);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &reverse_perm(8)), &perm_i(&i, &reverse_perm(8)), 0xF0);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &swap_perm(4)), &perm_i(&i, &swap_perm(4)), 0xCC);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &swap_perm(2)), &perm_i(&i, &swap_perm(2)), 0xAA);
    (t.0, t.1)
}

fn bitonic_merge_reg_8lanes(k: &[i64; LANES], i: &[i64; LANES]) -> ([i64; LANES], [i64; LANES]) {
    let t = cmp_merge_ki(k, i, &perm_k(k, &swap_perm(8)), &perm_i(i, &swap_perm(8)), 0xF0);
    let t = cmp_merge_ki(&t.0, &t.1, &perm_k(&t.0, &swap_perm(4)), &perm_i(&t.1, &swap_perm(4)), 0xCC);
    cmp_merge_ki(&t.0, &t.1, &perm_k(&t.0, &swap_perm(2)), &perm_i(&t.1, &swap_perm(2)), 0xAA)
}

fn bitonic_clean_n_vec(keys: &mut [[i64; LANES]], idxs: &mut [[i64; LANES]], base: usize, num_vecs: usize) {
    let mut num = num_vecs / 2;
    while num >= 2 {
        let mut j = 0usize;
        while j < num_vecs {
            for i in 0..num / 2 {
                let a = base + j + i;
                let b = base + j + i + num / 2;
                let t = coex(&keys[a], &idxs[a], &keys[b], &idxs[b]);
                keys[a] = t.0;
                idxs[a] = t.1;
                keys[b] = t.2;
                idxs[b] = t.3;
            }
            j += num;
        }
        num /= 2;
    }
}

fn bitonic_merge_n_vec(keys: &mut [[i64; LANES]], idxs: &mut [[i64; LANES]], base: usize, num_vecs: usize) {
    if num_vecs == 2 {
        let j = base + 1;
        keys[j].reverse();
        idxs[j].reverse();
        let t = coex(&keys[base], &idxs[base], &keys[j], &idxs[j]);
        keys[base] = t.0;
        idxs[base] = t.1;
        keys[j] = t.2;
        idxs[j] = t.3;
        keys[j].reverse();
        idxs[j].reverse();
    } else if num_vecs > 2 {
        for i in 0..num_vecs / 2 {
            let a = base + i;
            let j = base + num_vecs - i - 1;
            keys[j].reverse();
            idxs[j].reverse();
            let t = coex(&keys[a], &idxs[a], &keys[j], &idxs[j]);
            keys[a] = t.0;
            idxs[a] = t.1;
            keys[j] = t.2;
            idxs[j] = t.3;
            keys[j].reverse();
            idxs[j].reverse();
        }
    }
    bitonic_clean_n_vec(keys, idxs, base, num_vecs);
    for i in 0..num_vecs {
        let t = bitonic_merge_reg_8lanes(&keys[base + i], &idxs[base + i]);
        keys[base + i] = t.0;
        idxs[base + i] = t.1;
    }
}

fn bitonic_fullmerge_n_vec(
    keys: &mut [[i64; LANES]],
    idxs: &mut [[i64; LANES]],
    base: usize,
    num_vecs: usize,
    num_per: usize,
) {
    if num_per > num_vecs {
        return;
    }
    for i in 0..num_vecs / num_per {
        bitonic_merge_n_vec(keys, idxs, base + i * num_per, num_per);
    }
    bitonic_fullmerge_n_vec(keys, idxs, base, num_vecs, num_per * 2);
}

/// argsort_n&lt;vtype, argtype, 256&gt; 的 numVecs 会按 N 递归减半。
fn argsort_n_vec(arr: &[i64], ind: &mut [i64], n: usize, num_vecs: usize) {
    let mut num_vecs = num_vecs;
    while num_vecs > 1 && n * 2 <= num_vecs * LANES {
        num_vecs /= 2;
    }
    let mut keys = vec![[0i64; LANES]; num_vecs];
    let mut idxs = vec![[0i64; LANES]; num_vecs];
    let half = num_vecs / 2;
    for i in 0..half {
        for l in 0..LANES {
            let v = ind[i * LANES + l];
            idxs[i][l] = v;
            keys[i][l] = arr[v as usize];
        }
    }
    for i in half..num_vecs {
        let num_to_read = if n > i * LANES { (n - i * LANES).min(LANES) } else { 0 };
        for l in 0..LANES {
            if l < num_to_read {
                let v = ind[i * LANES + l];
                idxs[i][l] = v;
                keys[i][l] = arr[v as usize];
            } else {
                // indexType::zmm_max() / keyType::zmm_max()：补齐的通道放最大值
                idxs[i][l] = I64MAX;
                keys[i][l] = I64MAX;
            }
        }
    }
    for i in 0..num_vecs {
        let t = sort_reg_8lanes(&keys[i], &idxs[i]);
        keys[i] = t.0;
        idxs[i] = t.1;
    }
    bitonic_fullmerge_n_vec(&mut keys, &mut idxs, 0, num_vecs, 2);
    for i in 0..half {
        for l in 0..LANES {
            ind[i * LANES + l] = idxs[i][l];
        }
    }
    for i in half..num_vecs {
        let num_to_write = if n > i * LANES { (n - i * LANES).min(LANES) } else { 0 };
        for l in 0..num_to_write {
            ind[i * LANES + l] = idxs[i][l];
        }
    }
}

fn argsort_n(arr: &[i64], ind: &mut [i64], left: usize, n: usize) {
    let mut sub: Vec<i64> = ind[left..left + n].to_vec();
    argsort_n_vec(arr, &mut sub, n, 256 / LANES);
    ind[left..left + n].copy_from_slice(&sub);
}

/// get_pivot_64bit（numlanes=8 的 median-of-8 分支）
fn get_pivot(arr: &[i64], ind: &[i64], left: usize, right: usize) -> i64 {
    if right - left >= LANES {
        let size = (right - left) / LANES;
        // _mm512_set_epi64(v1..v8)：通道 0 放最后一个参数
        let rand_vec: [i64; LANES] = [
            arr[ind[left + 8 * size] as usize],
            arr[ind[left + 7 * size] as usize],
            arr[ind[left + 6 * size] as usize],
            arr[ind[left + 5 * size] as usize],
            arr[ind[left + 4 * size] as usize],
            arr[ind[left + 3 * size] as usize],
            arr[ind[left + 2 * size] as usize],
            arr[ind[left + 1 * size] as usize],
        ];
        // vtype::sort_vec：只按键的排序网络
        let mut k = rand_vec;
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(2)), 0xAA);
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(4)), 0xCC);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(2)), 0xAA);
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(8)), 0xF0);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(4)), 0xCC);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(2)), 0xAA);
        return k[4];
    }
    arr[ind[right] as usize]
}

/// partition_vec（argsort 版：>= pivot 的压缩写到右侧，< pivot 的写到左侧，通道序保持）
fn partition_vec(
    ind: &mut [i64],
    l_store: usize,
    r_store_plus: usize,
    arg_vec: &[i64; LANES],
    curr_vec: &[i64; LANES],
    pivot: i64,
) -> usize {
    let mut gt = [false; LANES];
    let mut amount = 0usize;
    for l in 0..LANES {
        gt[l] = curr_vec[l] >= pivot;
        if gt[l] {
            amount += 1;
        }
    }
    let mut p = l_store;
    for l in 0..LANES {
        if !gt[l] {
            ind[p] = arg_vec[l];
            p += 1;
        }
    }
    let mut p = r_store_plus - amount;
    for l in 0..LANES {
        if gt[l] {
            ind[p] = arg_vec[l];
            p += 1;
        }
    }
    amount
}

fn argpartition_scalar(
    arr: &[i64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: i64,
    smallest: &mut i64,
    biggest: &mut i64,
) -> usize {
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % LANES;
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        *smallest = (*smallest).min(v);
        *biggest = (*biggest).max(v);
        if !(v < pivot) {
            right -= 1;
            ind.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    if right - left == LANES {
        let mut av = [0i64; LANES];
        av.copy_from_slice(&ind[left..left + LANES]);
        let mut cv = [0i64; LANES];
        for l in 0..LANES {
            cv[l] = arr[av[l] as usize];
        }
        let amount = partition_vec(ind, left, left + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            *smallest = (*smallest).min(cv[l]);
            *biggest = (*biggest).max(cv[l]);
        }
        return left + (LANES - amount);
    }
    let mut av_left = [0i64; LANES];
    av_left.copy_from_slice(&ind[left..left + LANES]);
    let mut cv_left = [0i64; LANES];
    for l in 0..LANES {
        cv_left[l] = arr[av_left[l] as usize];
    }
    let mut av_right = [0i64; LANES];
    av_right.copy_from_slice(&ind[right - LANES..right]);
    let mut cv_right = [0i64; LANES];
    for l in 0..LANES {
        cv_right[l] = arr[av_right[l] as usize];
    }
    let mut r_store = right - LANES;
    let mut l_store = left;
    let mut left = left + LANES;
    let mut right = right - LANES;
    while right != left {
        let (av, cv) = if (r_store + LANES) - right < left - l_store {
            right -= LANES;
            let mut av = [0i64; LANES];
            av.copy_from_slice(&ind[right..right + LANES]);
            let mut cv = [0i64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            (av, cv)
        } else {
            let mut av = [0i64; LANES];
            av.copy_from_slice(&ind[left..left + LANES]);
            let mut cv = [0i64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            left += LANES;
            (av, cv)
        };
        let amount = partition_vec(ind, l_store, r_store + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            *smallest = (*smallest).min(cv[l]);
            *biggest = (*biggest).max(cv[l]);
        }
        r_store -= amount;
        l_store += LANES - amount;
    }
    let amount = partition_vec(ind, l_store, r_store + LANES, &av_left, &cv_left, pivot);
    for l in 0..LANES {
        *smallest = (*smallest).min(cv_left[l]);
        *biggest = (*biggest).max(cv_left[l]);
    }
    l_store += LANES - amount;
    let amount = partition_vec(ind, l_store, l_store + LANES, &av_right, &cv_right, pivot);
    for l in 0..LANES {
        *smallest = (*smallest).min(cv_right[l]);
        *biggest = (*biggest).max(cv_right[l]);
    }
    l_store + LANES - amount
}

const NUM_UNROLL: usize = 4;

fn argpartition_unrolled(
    arr: &[i64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: i64,
    smallest: &mut i64,
    biggest: &mut i64,
) -> usize {
    if right_in - left_in <= 8 * NUM_UNROLL * LANES {
        return argpartition_scalar(arr, ind, left_in, right_in, pivot, smallest, biggest);
    }
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % (NUM_UNROLL * LANES);
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        *smallest = (*smallest).min(v);
        *biggest = (*biggest).max(v);
        if !(v < pivot) {
            right -= 1;
            ind.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    let mut kv_left = [[0i64; LANES]; NUM_UNROLL];
    let mut iv_left = [[0i64; LANES]; NUM_UNROLL];
    let mut kv_right = [[0i64; LANES]; NUM_UNROLL];
    let mut iv_right = [[0i64; LANES]; NUM_UNROLL];
    for ii in 0..NUM_UNROLL {
        iv_left[ii].copy_from_slice(&ind[left + LANES * ii..left + LANES * (ii + 1)]);
        for l in 0..LANES {
            kv_left[ii][l] = arr[iv_left[ii][l] as usize];
        }
        let s = right - LANES * (NUM_UNROLL - ii);
        iv_right[ii].copy_from_slice(&ind[s..s + LANES]);
        for l in 0..LANES {
            kv_right[ii][l] = arr[iv_right[ii][l] as usize];
        }
    }
    let mut r_store = right - LANES;
    let mut l_store = left;
    let mut left = left + NUM_UNROLL * LANES;
    let mut right = right - NUM_UNROLL * LANES;

    macro_rules! upd {
        ($cv:expr) => {
            for l in 0..LANES {
                *smallest = (*smallest).min($cv[l]);
                *biggest = (*biggest).max($cv[l]);
            }
        };
    }

    while right != left {
        let mut avs = [[0i64; LANES]; NUM_UNROLL];
        let mut cvs = [[0i64; LANES]; NUM_UNROLL];
        if (r_store + LANES) - right < left - l_store {
            right -= NUM_UNROLL * LANES;
            for ii in 0..NUM_UNROLL {
                avs[ii].copy_from_slice(&ind[right + ii * LANES..right + (ii + 1) * LANES]);
            }
        } else {
            for ii in 0..NUM_UNROLL {
                avs[ii].copy_from_slice(&ind[left + ii * LANES..left + (ii + 1) * LANES]);
            }
            left += NUM_UNROLL * LANES;
        }
        for ii in 0..NUM_UNROLL {
            for l in 0..LANES {
                cvs[ii][l] = arr[avs[ii][l] as usize];
            }
            let amount = partition_vec(ind, l_store, r_store + LANES, &avs[ii], &cvs[ii], pivot);
            upd!(cvs[ii]);
            l_store += LANES - amount;
            r_store -= amount;
        }
    }
    for ii in 0..NUM_UNROLL {
        let amount = partition_vec(ind, l_store, r_store + LANES, &iv_left[ii], &kv_left[ii], pivot);
        upd!(kv_left[ii]);
        l_store += LANES - amount;
        r_store -= amount;
    }
    for ii in 0..NUM_UNROLL {
        let amount = partition_vec(ind, l_store, r_store + LANES, &iv_right[ii], &kv_right[ii], pivot);
        upd!(kv_right[ii]);
        l_store += LANES - amount;
        r_store -= amount;
    }
    l_store
}

fn argsort_rec(arr: &[i64], ind: &mut [i64], left: usize, right: i64, max_iters: i64) {
    if max_iters <= 0 {
        // x86-simd-sort 这里退回 std::sort（libstdc++ introsort）。本复刻用「按值稳定排序」
        // 代替：实测 4 组验收数据的 48 个候选数组一次都没走到这里。
        let mut sub: Vec<i64> = ind[left..=(right as usize)].to_vec();
        sub.sort_by(|&a, &b| arr[a as usize].cmp(&arr[b as usize]));
        ind[left..=(right as usize)].copy_from_slice(&sub);
        return;
    }
    if (right + 1) as usize - left <= 256 {
        argsort_n(arr, ind, left, (right + 1) as usize - left);
        return;
    }
    let (l, r) = (left, right as usize);
    let pivot = get_pivot(arr, ind, l, r);
    let mut smallest = I64MAX;
    let mut biggest = i64::MIN;
    let pivot_index = argpartition_unrolled(arr, ind, l, r + 1, pivot, &mut smallest, &mut biggest);
    if pivot != smallest {
        argsort_rec(arr, ind, left, pivot_index as i64 - 1, max_iters - 1);
    }
    if pivot != biggest {
        argsort_rec(arr, ind, pivot_index, right, max_iters - 1);
    }
}

/// numpy 2.3.5 在 AVX512 机器上 `np.argsort(int64_array, kind="quicksort")` 的结果（升序下标）。
fn xss_argsort_i64(arr: &[i64]) -> Vec<i64> {
    let n = arr.len();
    let mut ind: Vec<i64> = (0..n as i64).collect();
    if n <= 1 {
        return ind;
    }
    // std::is_sorted(arr, arr+n, arr[a] < arr[b]) 的提前返回
    if (1..n).all(|i| arr[i - 1] <= arr[i]) {
        return ind;
    }
    let mut log2 = 0i64;
    while (1i64 << (log2 + 1)) <= n as i64 {
        log2 += 1;
    }
    let max_iters = 2 * log2;
    argsort_rec(arr, &mut ind, 0, n as i64 - 1, max_iters);
    ind
}

/// pandas `sort_values(col, ascending=False, kind="quicksort")` 的 indexer
/// （pandas/core/sorting.py 的 nargsort：先把数组倒过来稳定地取 argsort，再倒回来）
fn pandas_argsort_desc(vals: &[i64]) -> Vec<usize> {
    let n = vals.len();
    let rev: Vec<i64> = vals.iter().rev().cloned().collect();
    let order = xss_argsort_i64(&rev);
    let mut indexer: Vec<usize> = order.iter().map(|&o| n - 1 - o as usize).collect();
    indexer.reverse();
    indexer
}

// ============================================================================
// 盘口长表：按 (价, 时间) 排序 + 前缀和，支持 O(log n) 的「同价、时间窗内 vol 均值」
// ============================================================================

struct Book {
    price: Vec<f64>,
    start: Vec<usize>,
    end: Vec<usize>,
    times: Vec<i64>,
    psum: Vec<f64>,
    pcnt: Vec<usize>,
}

impl Book {
    fn build(rows: &[(i64, f64, f64)]) -> Book {
        // NaN 价永远匹配不上（pandas 的 `price == p1` 对 NaN 为 False），建表时直接丢掉
        let mut v: Vec<(f64, i64, f64)> = rows
            .iter()
            .filter(|&&(_, p, _)| !p.is_nan())
            .map(|&(t, p, vol)| (p, t, vol))
            .collect();
        v.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut b = Book {
            price: Vec::new(),
            start: Vec::new(),
            end: Vec::new(),
            times: Vec::with_capacity(v.len()),
            psum: vec![0.0],
            pcnt: vec![0],
        };
        let mut i = 0usize;
        while i < v.len() {
            let mut j = i + 1;
            while j < v.len() && v[j].0 == v[i].0 {
                j += 1;
            }
            b.price.push(v[i].0);
            b.start.push(b.times.len());
            let mut s = *b.psum.last().unwrap();
            let mut c = *b.pcnt.last().unwrap();
            for k in i..j {
                b.times.push(v[k].1);
                if !v[k].2.is_nan() {
                    s += v[k].2;
                    c += 1;
                }
                b.psum.push(s);
                b.pcnt.push(c);
            }
            b.end.push(b.times.len());
            i = j;
        }
        b
    }

    /// 同价、时间落在闭区间 [lo, hi] 内的 vol 均值（NaN 跳过；没有有效值给 NaN）
    fn mean_vol(&self, price: f64, lo: i64, hi: i64) -> f64 {
        if price.is_nan() {
            return f64::NAN;
        }
        // 二分找价格组（NaN 价在这里被排到末尾，永远匹配不上）
        let mut a = 0usize;
        let mut b = self.price.len();
        while a < b {
            let m = (a + b) / 2;
            if self.price[m] < price {
                a = m + 1;
            } else if self.price[m] > price {
                b = m;
            } else {
                let s = self.start[m];
                let e = self.end[m];
                let t = &self.times[s..e];
                // lower_bound(lo)
                let mut x = 0usize;
                let mut y = t.len();
                while x < y {
                    let mm = (x + y) / 2;
                    if t[mm] < lo {
                        x = mm + 1;
                    } else {
                        y = mm;
                    }
                }
                let l = x;
                // upper_bound(hi)
                let mut x = l;
                let mut y = t.len();
                while x < y {
                    let mm = (x + y) / 2;
                    if t[mm] <= hi {
                        x = mm + 1;
                    } else {
                        y = mm;
                    }
                }
                let r = x;
                let cnt = self.pcnt[s + r] - self.pcnt[s + l];
                if cnt == 0 {
                    return f64::NAN;
                }
                return (self.psum[s + r] - self.psum[s + l]) / cnt as f64;
            }
        }
        f64::NAN
    }
}

// ============================================================================
// 指标
// ============================================================================

/// 与 pandas `asks[before_mask].vol.mean()` 对应
fn ratio_sum(
    tmax: &[(i64, f64)],
    tmin: &[(i64, f64)],
    asks: &Book,
    bids: &Book,
    future_minute: i64,
) -> [f64; 4] {
    if tmax.is_empty() || tmin.is_empty() {
        return [f64::NAN; 4];
    }
    let wa = future_minute * MIN_NS;
    let wb = 3 * MIN_NS; // bids 侧写死 3 分钟
    let (mut apos, mut aneg) = (0.0f64, 0.0f64);
    for &(t1, p1) in tmax.iter() {
        let before = asks.mean_vol(p1, t1 - wa, t1);
        let after = asks.mean_vol(p1, t1, t1 + wa);
        let ratio = (after - before) / before;
        if ratio > 0.0 {
            apos += ratio;
        } else if ratio < 0.0 {
            aneg += ratio;
        }
    }
    let (mut bpos, mut bneg) = (0.0f64, 0.0f64);
    for &(t1, p1) in tmin.iter() {
        let before = bids.mean_vol(p1, t1 - wb, t1);
        let after = bids.mean_vol(p1, t1, t1 + wb);
        let ratio = (after - before) / before;
        if ratio > 0.0 {
            bpos += ratio;
        } else if ratio < 0.0 {
            bneg += ratio;
        }
    }
    [apos, bpos, aneg, bneg]
}

#[inline]
fn in_window(c: i64, lo: i64, hi: i64) -> bool {
    c >= lo && c <= hi
}

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trade_new_time(code, date)?;
    if trades.is_empty() {
        return None;
    }
    let (mkt_ask, mkt_bid) = read_market_new_time(code, date)?;
    let asks = Book::build(&mkt_ask);
    let bids = Book::build(&mkt_bid);

    let prices: Vec<f64> = trades.iter().map(|&(_, p)| p).collect();
    let price_max = max_range_loop(&prices, false);
    let price_min = min_range_loop(&prices, false);
    let times: Vec<i64> = trades.iter().map(|&(t, _)| t).collect();

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for &param in PARAMS.iter() {
        for &fut in FUTURES.iter() {
            let w = fut * SEC_NS;
            let fmax = rolling_future_max(&times, &to_f64(&price_max), w);
            let fmin = rolling_future_max(&times, &to_f64(&price_min), w);

            // 候选：该列 > 未来窗口最大值（NaN 比较为 False）
            let sel_max = pick(&price_max, &fmax, param, &trades);
            let sel_min = pick(&price_min, &fmin, param, &trades);

            for &(lo, hi) in SESSIONS.iter() {
                let tm: Vec<(i64, f64)> = sel_max
                    .iter()
                    .filter(|&&(t, _)| in_window(t, lo, hi))
                    .cloned()
                    .collect();
                let tn: Vec<(i64, f64)> = sel_min
                    .iter()
                    .filter(|&&(t, _)| in_window(t, lo, hi))
                    .cloned()
                    .collect();
                for &fm in [1i64, 3i64].iter() {
                    let r = ratio_sum(&tm, &tn, &asks, &bids, fm);
                    out.extend(r.iter().map(|&v| v as f32));
                }
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

#[inline]
fn to_f64(v: &[i64]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// 候选行里取该列最大的 param 行（并列顺序由 numpy 决定）。
/// 返回 (平移后钟点, 价格) 列表；候选不多于 param 行时全部返回（求和与顺序无关）。
fn pick(col: &[i64], futmax: &[f64], param: usize, trades: &[(i64, f64)]) -> Vec<(i64, f64)> {
    let idx: Vec<usize> = (0..col.len())
        .filter(|&i| (col[i] as f64) > futmax[i])
        .collect();
    if idx.is_empty() {
        return Vec::new();
    }
    if idx.len() <= param {
        return idx.iter().map(|&i| trades[i]).collect();
    }
    let vals: Vec<i64> = idx.iter().map(|&i| col[i]).collect();
    let order = pandas_argsort_desc(&vals);
    order[..param].iter().map(|&k| trades[idx[k]]).collect()
}

/// hm039 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
