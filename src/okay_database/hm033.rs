//! hm033 相似单时间抱团特征 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm33_相似单时间抱团特征/hm33_okay.py
//! get_many=0 / read_full=0 / change_time=1，逐笔一整张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码在算什么
//! ════════════════════════════════════════════════════════════════════
//! 1) 逐笔按 09:30:00~14:57:00 取窗口（真实钟点，闭区间，微秒级）
//! 2) 按 ask_order / bid_order 分组，每组取 (exchtime 最小值, turnover 求和, volume 求和)，
//!    再按 turnover 降序排序。**并列顺序由 numpy 决定**（见第三节）。
//! 3) 把每组的最小成交时间写成 "HHMMSSffffff"，切掉首字符和末 3 位（slice(1,-3)），
//!    得到 8 个字符：时的个位 + 分 + 秒 + 毫秒 3 位；再按字符做词频向量
//!    （rp.vectorize_sentences_list 按空白分词，其实就是统计每个数字字符出现几次）。
//!    于是 asks/bids 各是一张「行=委托、列=数字 0..9、值=该数字出现次数」的计数表。
//! 4) 对 lag=0..29 算行与行之间计数向量的余弦相似度：
//!       sim_i[r] = dot(v_r, v_{r-i}) / (|v_r| |v_{r-i}|)
//!    （r<i 时是 NaN）。把 30 条 sim 首尾拼起来当一条序列，算它的
//!    mean / std / skew / autocorr(1..10)。
//!    ！！bids 侧写死了 shift(1)，30 条完全相同，这是旧代码的原样。
//! 5) corr 那部分：把 df 的 exchtime 列**覆盖**成 sim_0，再对
//!    [第二列, 第一列, exchtime, turnover, volume] 求相关矩阵，取 6 个位置。
//!    sim_0 在数值上恒等于 s/(sqrt(s)*sqrt(s))，其中 s = Σ_d count_d² 是整数，
//!    所以这一列是「1 加上几个 ulp 的浮点噪声」，corr 值完全由这点噪声决定，
//!    sim_0 必须逐位复现（用 f64 的 sqrt 与 IEEE 除法即可，np.power(x,0.5)==sqrt(x)）。
//! 6) 五段窗口各做一遍，再对全体行做一遍、对「volume>100 且 volume<=其 0.7 分位」的子集做一遍。
//!    子集那一遍里，sim 序列仍然来自全体行（旧代码的 sing 只换 df，不换 asks/bids），
//!    只有 corr 那 18 个值用的是子集的表。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、名字与值的个数对不上（旧代码原样，必须照抄）
//! ════════════════════════════════════════════════════════════════════
//! 旧代码 names 有 40 个基础名 × 5 段 × {full, part} = 400 个；
//! 而 one() 真正返回的值只有 37 个（少了 3 个：names 里多写了三个 ratio 名）。
//! 于是 names_single 与值从第 19 位起就整体错位 3 格，旧框架
//! `zip(names_single, res)` 只截到 370 个。本文件照抄这个行为：
//!   names() 返回 400 个名字，compute_stock 返回 370 个值（正常情形）。
//! 另外窗口不足 5 行时 one() 返回 40 个 NaN（而不是 37 个），长度会变成 373/376…，
//! 这里也照抄。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、排序并列顺序：逐行复刻 numpy 的 x86-simd-sort argsort
//! ════════════════════════════════════════════════════════════════════
//! `sort_values(["turnover"], ascending=False)` 是单键 → pandas nargsort →
//! np.argsort(kind="quicksort")，numpy 2.x 在 AVX512 机器上对 float64 走
//! x86-simd-sort 的 8 通道位元排序网络 + 分区。本系列的 turnover 并列极其严重
//! （20250303 000001 有 30489/32436 个并列），实测换一个稳定排序，400 个值里有
//! 279 个对不上，所以必须逐位复刻。
//! 复刻对象：numpy/_core/src/npysort/x86-simd-sort/src/
//!   {xss-common-argsort.h, xss-network-keyvaluesort.hpp, xss-reg-networks.hpp}
//! 的 double 路径（numlanes=8，argtype=size_t 但下标不影响顺序）。
//! 与 int64 版的唯一差别：补齐通道的键用 +inf（不是 i64::MAX），
//! smallest/biggest 初值用 +inf / -inf。
//! 迭代次数耗尽时 x86-simd-sort 退回 std::sort；实测真实数据上每个数组会触发
//! 0~19 次，所以这里把 libstdc++ 的 introsort（含 __partial_sort 的堆排序）也一并复刻。
//!
//! 对拍：本文件的两个复刻都先在 Python 里逐行实现，与 numpy 2.3.5 / pandas 2.3.3
//! 在本机 AVX512 上比过；48 组随机数组 + 8 组真实（日期,股票）的 turnover 数组，
//! argsort 结果与 pandas 排序后的行序完全一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、几个必须逐位对齐的 pandas 细节
//! ════════════════════════════════════════════════════════════════════
//! · deal_data 把 exchtime 从「微秒浮点数」转成 Timestamp：先 /1e6 得秒，
//!   再走 to_timedelta(unit="s")，即 cast_from_unit_vectorized：
//!       base = (int64)秒; frac = 秒 - base; frac = np.round(frac, 9); ns = base*1e9 + (int64)(frac*1e9)
//!   np.round(x,9) 内部是 rint(x*1e9)/1e9，会再舍入一次，所以末位 ns 有噪声；
//!   这个噪声会翻动毫秒位，进而改变字符向量，必须照抄。
//! · groupby(...).sum() 用的是 Kahan（Neumaier）补偿求和，不是朴素累加；
//!   turnover 有小数（如 212010.00000000003），朴素累加会让原本并列的组不再并列。
//! · DataFrame.corr() 走 pandas._libs.algos.nancorr，是 **Welford** 在线算法，
//!   covxy 的累加项是 dx(旧均值) * (vy - meany)(新均值)；Series.autocorr 走
//!   np.corrcoef（两遍法）。两者公式不同，不能互相代用。
//! · Series.mean/std/skew 按 pandas nanops 的写法（skew 用平方和与三次方和，
//!   末位小量 <1e-14 抹成 0）。
//! · Series.quantile(0.7) 是线性插值 ((n-1)*q 位置)。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

// ============================================================================
// 常量
// ============================================================================

const DAY_NS: i64 = 86_400_000_000_000;

const fn hms(h: i64, m: i64, s: i64) -> i64 {
    ((h * 3600 + m * 60 + s) * 1_000_000_000) as i64
}

/// 五个时段（真实钟点的闭区间，纳秒）。顺序 = 值/名字的拼接顺序。
const SESSIONS: [(&str, i64, i64); 5] = [
    ("allday", hms(9, 30, 0), hms(14, 57, 0)),
    ("open30", hms(9, 30, 0), hms(10, 0, 0)),
    ("morning", hms(9, 30, 0), hms(11, 30, 0)),
    ("afternoon", hms(13, 0, 0), hms(14, 57, 0)),
    ("close30", hms(14, 30, 0), hms(14, 57, 0)),
];

/// 旧代码 one() 真正返回 37 个值（这个名字与值错位的历史遗留见文件头第二节）
#[allow(dead_code)]
const N_VALUE: usize = 37;
/// 旧代码 names 里一段有 40 个名字
const N_METRIC: usize = 40;
/// 5 段 × 40 名 × 2（full/part）
pub const N_OUT: usize = 5 * N_METRIC * 2;

const PARAM: usize = 30;

/// 旧代码 names 里 40 个基础名，顺序原样
const METRIC_NAMES: [&str; N_METRIC] = [
    "asks_mean",
    "asks_std",
    "asks_skew",
    "bids_mean",
    "bids_std",
    "bids_skew",
    "asks_mean_minus_bids_mean",
    "asks_std_minus_bids_std",
    "asks_skew_minus_bids_skew",
    "asks_corr1_minus_bids_corr1",
    "asks_corr2_minus_bids_corr2",
    "asks_corr3_minus_bids_corr3",
    "asks_corr4_minus_bids_corr4",
    "asks_corr5_minus_bids_corr5",
    "asks_corr6_minus_bids_corr6",
    "asks_corr7_minus_bids_corr7",
    "asks_corr8_minus_bids_corr8",
    "asks_corr9_minus_bids_corr9",
    "asks_corr10_minus_bids_corr10",
    "asks_mean_minus_bids_mean_ratio",
    "asks_std_minus_bids_std_ratio",
    "asks_skew_minus_bids_skew_ratio",
    "asks_corr_num_order",
    "asks_corr_num_sim",
    "asks_corr_num_turnover",
    "asks_corr_order_sim",
    "asks_corr_order_turnover",
    "asks_corr_sim_turnover",
    "bids_corr_num_order",
    "bids_corr_num_sim",
    "bids_corr_num_turnover",
    "bids_corr_order_sim",
    "bids_corr_order_turnover",
    "bids_corr_sim_turnover",
    "asks_corr_num_order_minus_bids_corr_num_order",
    "asks_corr_num_sim_minus_bids_corr_num_sim",
    "asks_corr_num_turnover_minus_bids_corr_num_turnover",
    "asks_corr_order_sim_minus_bids_corr_order_sim",
    "asks_corr_order_turnover_minus_bids_corr_order_turnover",
    "asks_corr_sim_turnover_minus_bids_corr_sim_turnover",
];

/// 400 个名字（与旧代码 names_single 完全一致）
pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for suffix in ["_full", "_part"] {
        for (sess, _, _) in SESSIONS.iter() {
            for m in METRIC_NAMES.iter() {
                out.push(format!("{}_{}{}", m, sess, suffix));
            }
        }
    }
    out
}

// ============================================================================
// CSV 读取（自己读，字段索引与 fast_csv_reader 的常量一致）
// ============================================================================
// symbol,datetime,date,time,exchtime,localtime,index,price,volume,turnover,
// flag,order_type,func_code,ask_order,bid_order[,num]
const TRD_NCOL: usize = 15;
const COL_EXCHTIME: usize = 4;
const COL_VOLUME: usize = 8;
const COL_TURNOVER: usize = 9;
const COL_FLAG: usize = 10;
const COL_ASK: usize = 13;
const COL_BID: usize = 14;

#[inline]
fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && (b[i] == b' ' || b[i] == b'\t') {
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

/// 10 的幂表：用 Rust 自己的正确舍入解析 "1e{k}" 建出来，
/// 与 pandas/tokenizer.c 里那张写死的 e[] 表逐位一致。
fn pow10_table() -> &'static [f64; 309] {
    static T: std::sync::OnceLock<[f64; 309]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [1.0f64; 309];
        for (k, v) in t.iter_mut().enumerate() {
            *v = format!("1e{}", k).parse::<f64>().unwrap();
        }
        t
    })
}

/// pandas C 解析器 `precise_xstrtod`（_libs/src/parser/tokenizer.c）的逐行复刻。
///
/// ！！这不是「正确舍入」的十进制转二进制：它只把**前 17 位有效数字**
/// 用 `number = number*10 + digit` 累加进 double，多出来的整数位转成指数，
/// 多出来的小数位直接丢掉，最后用缓存好的 10 的幂做**一次**乘或除。
/// 实测 20250303 000001 有 260/76488 个 turnover 与 Rust `str::parse` 差 1 ulp
/// （例如 "2165.7599999999998" 这里读成 2165.76）。这点差别会让原本并列的组
/// 不再并列，进而改变 turnover 降序的并列顺序，所以必须照抄。
fn parse_f64_pandas(b: &[u8]) -> f64 {
    const MAX_DIGITS: usize = 17;
    let n = b.len();
    let mut p = 0usize;
    while p < n && matches!(b[p], b' ' | b'\t' | b'\r' | b'\n') {
        p += 1;
    }
    let mut negative = false;
    if p < n && (b[p] == b'-' || b[p] == b'+') {
        negative = b[p] == b'-';
        p += 1;
    }
    let mut number = 0.0f64;
    let mut exponent: i32 = 0;
    let mut num_digits = 0usize;
    let mut num_decimals = 0usize;
    while p < n && b[p].is_ascii_digit() {
        if num_digits < MAX_DIGITS {
            number = number * 10.0 + (b[p] - b'0') as f64;
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    if p < n && b[p] == b'.' {
        p += 1;
        while num_digits < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            number = number * 10.0 + (b[p] - b'0') as f64;
            p += 1;
            num_digits += 1;
            num_decimals += 1;
        }
        if num_digits >= MAX_DIGITS {
            while p < n && b[p].is_ascii_digit() {
                p += 1;
            }
        }
        exponent -= num_decimals as i32;
    }
    if num_digits == 0 {
        return 0.0;
    }
    if negative {
        number = -number;
    }
    // 指数段（本数据没有 'e'，按源码照抄）
    if p < n && (b[p] | 0x20) == b'e' {
        p += 1;
        let mut eneg = false;
        if p < n && (b[p] == b'-' || b[p] == b'+') {
            eneg = b[p] == b'-';
            p += 1;
        }
        let mut nn: i32 = 0;
        let mut nd = 0usize;
        while nd < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            nn = nn * 10 + (b[p] - b'0') as i32;
            nd += 1;
            p += 1;
        }
        if eneg {
            exponent -= nn;
        } else {
            exponent += nn;
        }
    }
    if exponent > 308 {
        return f64::INFINITY;
    } else if exponent > 0 {
        number * pow10_table()[exponent as usize]
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            let t = pow10_table();
            (number / t[(-308 - exponent) as usize]) / t[308]
        }
    } else {
        number / pow10_table()[(-exponent) as usize]
    }
}

fn split_fields<'a>(line: &'a [u8], f: &mut [&'a [u8]]) {
    for x in f.iter_mut() {
        *x = &line[..0];
    }
    let n = f.len();
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

fn find_file(code: &str, date: i64) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 当天钟点（纳秒，含 8 小时偏移）
#[inline]
fn clock_ns(ns: i64) -> i64 {
    ns.rem_euclid(DAY_NS)
}

/// pandas cast_from_unit_vectorized(values=秒, unit="s") → 纳秒
/// base=(int64)v; frac=v-base; frac=np.round(frac,9); out=base*1e9+(int64)(frac*1e9)
#[inline]
fn sec_to_ns(sec: f64) -> i64 {
    let base = sec as i64; // 截断（本数据恒为正）
    let frac = sec - base as f64;
    // np.round(x, 9) 内部：rint(x*1e9)/1e9
    let scaled = (frac * 1e9).round_ties_even() / 1e9;
    base.wrapping_mul(1_000_000_000) + (scaled * 1e9) as i64
}

// ============================================================================
// numpy x86-simd-sort argsort(double) 的逐行复刻（AVX512，8 通道）
// ============================================================================

const LANES: usize = 8;

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
fn perm_k(k: &[f64; LANES], p: &[usize; LANES]) -> [f64; LANES] {
    let mut o = [0.0f64; LANES];
    for l in 0..LANES {
        o[l] = k[p[l]];
    }
    o
}

#[inline]
fn perm_i(i: &[i64; LANES], p: &[usize; LANES]) -> [i64; LANES] {
    let mut o = [0i64; LANES];
    for l in 0..LANES {
        o[l] = i[p[l]];
    }
    o
}

/// 只按键的 cmp_merge：0 -> min，1 -> max
fn cmp_merge_k(k1: &[f64; LANES], k2: &[f64; LANES], mask: u8) -> [f64; LANES] {
    let mut o = [0.0f64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        o[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
    }
    o
}

/// 带下标的 cmp_merge（xss-network-keyvaluesort.hpp）
/// 结果取自 in1 就保留下标 1，否则用下标 2（用 `ko == a` 判，与 C++ 的 eq 掩码一致）
fn cmp_merge_ki(
    k1: &[f64; LANES],
    i1: &[i64; LANES],
    k2: &[f64; LANES],
    i2: &[i64; LANES],
    mask: u8,
) -> ([f64; LANES], [i64; LANES]) {
    let mut ko = [0.0f64; LANES];
    let mut io = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        ko[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
        io[l] = if ko[l] == a { i1[l] } else { i2[l] };
    }
    (ko, io)
}

/// COEX：稳定的比较交换
fn coex(
    k1: &[f64; LANES],
    i1: &[i64; LANES],
    k2: &[f64; LANES],
    i2: &[i64; LANES],
) -> ([f64; LANES], [i64; LANES], [f64; LANES], [i64; LANES]) {
    let mut ko1 = [0.0f64; LANES];
    let mut ko2 = [0.0f64; LANES];
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

fn sort_reg_8lanes(k: &[f64; LANES], i: &[i64; LANES]) -> ([f64; LANES], [i64; LANES]) {
    let (mut k, mut i) = cmp_merge_ki(
        k,
        i,
        &perm_k(k, &reverse_perm(2)),
        &perm_i(i, &reverse_perm(2)),
        0xAA,
    );
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &reverse_perm(4)),
        &perm_i(&i, &reverse_perm(4)),
        0xCC,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &swap_perm(2)),
        &perm_i(&i, &swap_perm(2)),
        0xAA,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &reverse_perm(8)),
        &perm_i(&i, &reverse_perm(8)),
        0xF0,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &swap_perm(4)),
        &perm_i(&i, &swap_perm(4)),
        0xCC,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &swap_perm(2)),
        &perm_i(&i, &swap_perm(2)),
        0xAA,
    );
    (t.0, t.1)
}

fn bitonic_merge_reg_8lanes(k: &[f64; LANES], i: &[i64; LANES]) -> ([f64; LANES], [i64; LANES]) {
    let t = cmp_merge_ki(
        k,
        i,
        &perm_k(k, &swap_perm(8)),
        &perm_i(i, &swap_perm(8)),
        0xF0,
    );
    let t = cmp_merge_ki(
        &t.0,
        &t.1,
        &perm_k(&t.0, &swap_perm(4)),
        &perm_i(&t.1, &swap_perm(4)),
        0xCC,
    );
    cmp_merge_ki(
        &t.0,
        &t.1,
        &perm_k(&t.0, &swap_perm(2)),
        &perm_i(&t.1, &swap_perm(2)),
        0xAA,
    )
}

fn bitonic_clean_n_vec(keys: &mut [[f64; LANES]], idxs: &mut [[i64; LANES]], base: usize, num_vecs: usize) {
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

fn bitonic_merge_n_vec(keys: &mut [[f64; LANES]], idxs: &mut [[i64; LANES]], base: usize, num_vecs: usize) {
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
    keys: &mut [[f64; LANES]],
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

/// argsort_n<vtype, argtype, 256>（numVecs 按 N 递归减半，等价于 while）
fn argsort_n_vec(arr: &[f64], ind: &mut [i64], left: usize, n: usize, num_vecs: usize) {
    let mut num_vecs = num_vecs;
    while num_vecs > 1 && n * 2 <= num_vecs * LANES {
        num_vecs /= 2;
    }
    let mut keys = vec![[0.0f64; LANES]; num_vecs];
    let mut idxs = vec![[0i64; LANES]; num_vecs];
    let half = num_vecs / 2;
    for i in 0..half {
        for l in 0..LANES {
            let v = ind[left + i * LANES + l];
            idxs[i][l] = v;
            keys[i][l] = arr[v as usize];
        }
    }
    for i in half..num_vecs {
        let num_to_read = if n > i * LANES {
            (n - i * LANES).min(LANES)
        } else {
            0
        };
        for l in 0..LANES {
            if l < num_to_read {
                let v = ind[left + i * LANES + l];
                idxs[i][l] = v;
                keys[i][l] = arr[v as usize];
            } else {
                // keyType::zmm_max() = +inf；下标补齐值不参与比较
                idxs[i][l] = i64::MAX;
                keys[i][l] = f64::INFINITY;
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
            ind[left + i * LANES + l] = idxs[i][l];
        }
    }
    for i in half..num_vecs {
        let num_to_write = if n > i * LANES {
            (n - i * LANES).min(LANES)
        } else {
            0
        };
        for l in 0..num_to_write {
            ind[left + i * LANES + l] = idxs[i][l];
        }
    }
}

fn argsort_n(arr: &[f64], ind: &mut [i64], left: usize, n: usize) {
    let mut sub: Vec<i64> = ind[left..left + n].to_vec();
    argsort_n_vec(arr, &mut sub, 0, n, 256 / LANES);
    ind[left..left + n].copy_from_slice(&sub);
}

/// get_pivot_64bit（numlanes=8 的 median-of-8 分支）
fn get_pivot(arr: &[f64], ind: &[i64], left: usize, right: usize) -> f64 {
    if right - left >= LANES {
        let size = (right - left) / LANES;
        // _mm512_set_pd(v1..v8)：通道 0 放最后一个参数
        let rand_vec: [f64; LANES] = [
            arr[ind[left + 8 * size] as usize],
            arr[ind[left + 7 * size] as usize],
            arr[ind[left + 6 * size] as usize],
            arr[ind[left + 5 * size] as usize],
            arr[ind[left + 4 * size] as usize],
            arr[ind[left + 3 * size] as usize],
            arr[ind[left + 2 * size] as usize],
            arr[ind[left + 1 * size] as usize],
        ];
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

/// partition_vec_avx512：>= pivot 的压缩写到右侧，< pivot 的写到左侧，通道序保持
fn partition_vec(
    ind: &mut [i64],
    l_store: usize,
    r_store_plus: usize,
    arg_vec: &[i64; LANES],
    curr_vec: &[f64; LANES],
    pivot: f64,
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
    arr: &[f64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: f64,
    smallest: &mut f64,
    biggest: &mut f64,
) -> usize {
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % LANES;
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        *smallest = smallest.min(v);
        *biggest = biggest.max(v);
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
        let mut cv = [0.0f64; LANES];
        for l in 0..LANES {
            cv[l] = arr[av[l] as usize];
        }
        let amount = partition_vec(ind, left, left + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            *smallest = smallest.min(cv[l]);
            *biggest = biggest.max(cv[l]);
        }
        return left + (LANES - amount);
    }
    let mut av_left = [0i64; LANES];
    av_left.copy_from_slice(&ind[left..left + LANES]);
    let mut cv_left = [0.0f64; LANES];
    for l in 0..LANES {
        cv_left[l] = arr[av_left[l] as usize];
    }
    let mut av_right = [0i64; LANES];
    av_right.copy_from_slice(&ind[right - LANES..right]);
    let mut cv_right = [0.0f64; LANES];
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
            let mut cv = [0.0f64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            (av, cv)
        } else {
            let mut av = [0i64; LANES];
            av.copy_from_slice(&ind[left..left + LANES]);
            let mut cv = [0.0f64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            left += LANES;
            (av, cv)
        };
        let amount = partition_vec(ind, l_store, r_store + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            *smallest = smallest.min(cv[l]);
            *biggest = biggest.max(cv[l]);
        }
        r_store -= amount;
        l_store += LANES - amount;
    }
    let amount = partition_vec(ind, l_store, r_store + LANES, &av_left, &cv_left, pivot);
    for l in 0..LANES {
        *smallest = smallest.min(cv_left[l]);
        *biggest = biggest.max(cv_left[l]);
    }
    l_store += LANES - amount;
    let amount = partition_vec(ind, l_store, l_store + LANES, &av_right, &cv_right, pivot);
    for l in 0..LANES {
        *smallest = smallest.min(cv_right[l]);
        *biggest = biggest.max(cv_right[l]);
    }
    l_store + LANES - amount
}

const NUM_UNROLL: usize = 4;

fn argpartition_unrolled(
    arr: &[f64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: f64,
    smallest: &mut f64,
    biggest: &mut f64,
) -> usize {
    if right_in - left_in <= 8 * NUM_UNROLL * LANES {
        return argpartition_scalar(arr, ind, left_in, right_in, pivot, smallest, biggest);
    }
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % (NUM_UNROLL * LANES);
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        *smallest = smallest.min(v);
        *biggest = biggest.max(v);
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
    let mut kv_left = [[0.0f64; LANES]; NUM_UNROLL];
    let mut iv_left = [[0i64; LANES]; NUM_UNROLL];
    let mut kv_right = [[0.0f64; LANES]; NUM_UNROLL];
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
                *smallest = smallest.min($cv[l]);
                *biggest = biggest.max($cv[l]);
            }
        };
    }

    while right != left {
        let mut avs = [[0i64; LANES]; NUM_UNROLL];
        let mut cvs = [[0.0f64; LANES]; NUM_UNROLL];
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

// ---- libstdc++ std::sort（x86-simd-sort 迭代耗尽时的回退，stl_algo.h 的 introsort）

const SORT_THRESHOLD: usize = 16;

fn lg2_usize(n: usize) -> usize {
    let mut k = 0usize;
    while (1usize << (k + 1)) <= n {
        k += 1;
    }
    k
}

fn ins_sort(a: &mut [i64], first: usize, last: usize, key: &[f64]) {
    if first == last {
        return;
    }
    let mut i = first + 1;
    while i != last {
        if key[a[i] as usize] < key[a[first] as usize] {
            let val = a[i];
            let mut k = i;
            while k > first {
                a[k] = a[k - 1];
                k -= 1;
            }
            a[first] = val;
        } else {
            unguarded_linear_insert(a, i, key);
        }
        i += 1;
    }
}

fn unguarded_linear_insert(a: &mut [i64], last: usize, key: &[f64]) {
    let val = a[last];
    let mut cur = last;
    let mut nxt = last - 1;
    while key[val as usize] < key[a[nxt] as usize] {
        a[cur] = a[nxt];
        cur = nxt;
        nxt -= 1;
    }
    a[cur] = val;
}

fn unguarded_insertion_sort(a: &mut [i64], first: usize, last: usize, key: &[f64]) {
    for i in first..last {
        unguarded_linear_insert(a, i, key);
    }
}

fn final_insertion_sort(a: &mut [i64], first: usize, last: usize, key: &[f64]) {
    if last - first > SORT_THRESHOLD {
        ins_sort(a, first, first + SORT_THRESHOLD, key);
        unguarded_insertion_sort(a, first + SORT_THRESHOLD, last, key);
    } else {
        ins_sort(a, first, last, key);
    }
}

fn unguarded_partition(
    a: &mut [i64],
    mut first: usize,
    mut last: usize,
    pivot_val: f64,
    key: &[f64],
) -> usize {
    loop {
        while key[a[first] as usize] < pivot_val {
            first += 1;
        }
        last -= 1;
        while pivot_val < key[a[last] as usize] {
            last -= 1;
        }
        if !(first < last) {
            return first;
        }
        a.swap(first, last);
        first += 1;
    }
}

fn move_median_to_first(
    a: &mut [i64],
    result: usize,
    ai: usize,
    bi: usize,
    ci: usize,
    key: &[f64],
) {
    let (va, vb, vc) = (key[a[ai] as usize], key[a[bi] as usize], key[a[ci] as usize]);
    if va < vb {
        if vb < vc {
            a.swap(result, bi);
        } else if va < vc {
            a.swap(result, ci);
        } else {
            a.swap(result, ai);
        }
    } else if va < vc {
        a.swap(result, ai);
    } else if vb < vc {
        a.swap(result, ci);
    } else {
        a.swap(result, bi);
    }
}

// ---- __partial_sort（middle == last 时退化为 make_heap + sort_heap）

fn push_heap(a: &mut [i64], first: usize, hole_in: usize, top: usize, value: i64, key: &[f64]) {
    let mut hole = hole_in;
    let mut parent = (hole - 1) / 2;
    while hole > top && key[a[first + parent] as usize] < key[value as usize] {
        a[first + hole] = a[first + parent];
        hole = parent;
        if hole == 0 {
            break;
        }
        parent = (hole - 1) / 2;
    }
    a[first + hole] = value;
}

fn adjust_heap(a: &mut [i64], first: usize, hole_in: usize, length: usize, value: i64, key: &[f64]) {
    let top = hole_in;
    let mut hole = hole_in;
    let mut second = hole_in;
    while second < (length - 1) / 2 {
        second = 2 * (second + 1);
        if key[a[first + second] as usize] < key[a[first + second - 1] as usize] {
            second -= 1;
        }
        a[first + hole] = a[first + second];
        hole = second;
    }
    if (length & 1) == 0 && second == (length - 2) / 2 {
        second = 2 * (second + 1);
        a[first + hole] = a[first + second - 1];
        hole = second - 1;
    }
    push_heap(a, first, hole, top, value, key);
}

fn pop_heap(a: &mut [i64], first: usize, last: usize, result: usize, key: &[f64]) {
    let value = a[result];
    a[result] = a[first];
    adjust_heap(a, first, 0, last - first, value, key);
}

fn make_heap(a: &mut [i64], first: usize, last: usize, key: &[f64]) {
    if last - first < 2 {
        return;
    }
    let length = last - first;
    let mut parent = (length - 2) / 2;
    loop {
        let value = a[first + parent];
        adjust_heap(a, first, parent, length, value, key);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn sort_heap(a: &mut [i64], first: usize, mut last: usize, key: &[f64]) {
    while last - first > 1 {
        last -= 1;
        pop_heap(a, first, last, last, key);
    }
}

fn partial_sort(a: &mut [i64], first: usize, middle: usize, last: usize, key: &[f64]) {
    make_heap(a, first, middle, key);
    let mut i = middle;
    while i < last {
        if key[a[i] as usize] < key[a[first] as usize] {
            pop_heap(a, first, middle, i, key);
        }
        i += 1;
    }
    sort_heap(a, first, middle, key);
}

fn introsort_loop(a: &mut [i64], first: usize, last_in: usize, depth_limit_in: usize, key: &[f64]) {
    let mut last = last_in;
    let mut depth_limit = depth_limit_in;
    while last - first > SORT_THRESHOLD {
        if depth_limit == 0 {
            partial_sort(a, first, last, last, key);
            return;
        }
        depth_limit -= 1;
        let mid = first + (last - first) / 2;
        move_median_to_first(a, first, first + 1, mid, last - 1, key);
        let pivot_val = key[a[first] as usize];
        let cut = unguarded_partition(a, first + 1, last, pivot_val, key);
        introsort_loop(a, cut, last, depth_limit, key);
        last = cut;
    }
}

/// std::sort(ind[first:last], comp = key[i] < key[j])
fn std_sort_range(ind: &mut [i64], first: usize, last: usize, key: &[f64]) {
    if first == last {
        return;
    }
    let n = last - first;
    let mut sub: Vec<i64> = ind[first..last].to_vec();
    introsort_loop(&mut sub, 0, n, 2 * lg2_usize(n), key);
    final_insertion_sort(&mut sub, 0, n, key);
    ind[first..last].copy_from_slice(&sub);
}

fn argsort_rec(arr: &[f64], ind: &mut [i64], left: usize, right: i64, max_iters: i64) {
    if max_iters <= 0 {
        // x86-simd-sort 这里退回 std::sort（libstdc++ introsort），逐行复刻
        std_sort_range(ind, left, right as usize + 1, arr);
        return;
    }
    if (right + 1) as usize - left <= 256 {
        argsort_n(arr, ind, left, (right + 1) as usize - left);
        return;
    }
    let (l, r) = (left, right as usize);
    let pivot = get_pivot(arr, ind, l, r);
    let mut smallest = f64::INFINITY;
    let mut biggest = f64::NEG_INFINITY;
    let pivot_index = argpartition_unrolled(arr, ind, l, r + 1, pivot, &mut smallest, &mut biggest);
    if pivot != smallest {
        argsort_rec(arr, ind, left, pivot_index as i64 - 1, max_iters - 1);
    }
    if pivot != biggest {
        argsort_rec(arr, ind, pivot_index, right, max_iters - 1);
    }
}

/// np.argsort(float64, kind="quicksort") 在 AVX512 机器上的结果（升序下标）
fn xss_argsort_f64(arr: &[f64]) -> Vec<i64> {
    let n = arr.len();
    let mut ind: Vec<i64> = (0..n as i64).collect();
    if n <= 1 {
        return ind;
    }
    if (1..n).all(|i| arr[i - 1] <= arr[i]) {
        return ind;
    }
    let max_iters = 2 * lg2_usize(n) as i64;
    argsort_rec(arr, &mut ind, 0, n as i64 - 1, max_iters);
    ind
}

/// pandas `sort_values(col, ascending=False, kind="quicksort")` 的 indexer
/// （nargsort：先把数组倒过来取 argsort，再倒回来）
fn pandas_argsort_desc(vals: &[f64]) -> Vec<usize> {
    let n = vals.len();
    let rev: Vec<f64> = vals.iter().rev().cloned().collect();
    let order = xss_argsort_f64(&rev);
    let mut indexer: Vec<usize> = order.iter().map(|&o| n - 1 - o as usize).collect();
    indexer.reverse();
    indexer
}

// ============================================================================
// pandas 统计量的复刻
// ============================================================================

/// pandas._libs.algos.nancorr（DataFrame.corr 的 pearson 路径）= Welford 在线算法。
/// 注意 covxy 的累加项是「更新前的 dx」乘「更新后的 (vy - meany)」——
/// 这是与 libalgos.pyx 源码字面不同、但与本机 pandas 实测逐位一致的形式。
fn welford_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let (mut nobs, mut meanx, mut meany) = (0i64, 0.0f64, 0.0f64);
    let (mut ssqdmx, mut ssqdmy, mut covxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (vx, vy) = (x[i], y[i]);
        if !(vx.is_finite() && vy.is_finite()) {
            continue;
        }
        nobs += 1;
        let dx = vx - meanx;
        let dy = vy - meany;
        meanx += 1.0 / nobs as f64 * dx;
        meany += 1.0 / nobs as f64 * dy;
        ssqdmx += (vx - meanx) * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += dx * (vy - meany);
    }
    if nobs < 1 {
        return f64::NAN;
    }
    let div = (ssqdmx * ssqdmy).sqrt();
    if div == 0.0 {
        return f64::NAN;
    }
    covxy / div
}

/// pandas nanmean：非 NaN 的和除以个数
fn nan_mean(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0i64;
    for &t in v.iter() {
        if t.is_finite() {
            s += t;
            n += 1;
        }
    }
    if n == 0 {
        return f64::NAN;
    }
    s / n as f64
}

/// pandas nanstd（ddof=1）
fn nan_std(v: &[f64]) -> f64 {
    let m = nan_mean(v);
    if !m.is_finite() {
        return f64::NAN;
    }
    let mut s = 0.0f64;
    let mut n = 0i64;
    for &t in v.iter() {
        if t.is_finite() {
            let d = t - m;
            s += d * d;
            n += 1;
        }
    }
    if n <= 1 {
        return f64::NAN;
    }
    (s / (n - 1) as f64).sqrt()
}

#[inline]
fn zero_out_fperr(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        0.0
    } else {
        x
    }
}

/// pandas nanskew（无偏，等价于 scipy.stats.skew(bias=False)）
fn nan_skew(v: &[f64]) -> f64 {
    let m = nan_mean(v);
    if !m.is_finite() {
        return f64::NAN;
    }
    let mut n = 0i64;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &t in v.iter() {
        if t.is_finite() {
            let d = t - m;
            let a2 = d * d;
            m2 += a2;
            m3 += a2 * d;
            n += 1;
        }
    }
    if n < 3 {
        return f64::NAN;
    }
    let m2 = zero_out_fperr(m2);
    let m3 = zero_out_fperr(m3);
    let count = n as f64;
    (count * (count - 1.0).sqrt() / (count - 2.0)) * (m3 / m2.powf(1.5))
}

/// np.corrcoef(a, b)[0, 1]：两遍法，均值用 np.average
fn corrcoef2(a: &[f64], b: &[f64]) -> f64 {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            xs.push(a[i]);
            ys.push(b[i]);
        }
    }
    let m = xs.len();
    if m == 0 {
        return f64::NAN;
    }
    let mx = nan_mean(&xs);
    let my = nan_mean(&ys);
    let mut c00 = 0.0f64;
    let mut c11 = 0.0f64;
    let mut c01 = 0.0f64;
    for i in 0..m {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        c00 += dx * dx;
        c11 += dy * dy;
        c01 += dx * dy;
    }
    let f = (m - 1) as f64;
    let c00 = c00 / f;
    let c11 = c11 / f;
    let c01 = c01 / f;
    c01 / (c00.sqrt() * c11.sqrt())
}

/// pandas Series.quantile(q) 的线性插值
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = (n - 1) as f64 * q;
    let lo = idx.floor();
    let hi = idx.ceil();
    if lo == hi {
        return sorted[lo as usize];
    }
    let v0 = sorted[lo as usize];
    let v1 = sorted[hi as usize];
    v0 + (v1 - v0) * (idx - lo)
}

// ============================================================================
// 分组与主逻辑
// ============================================================================

#[derive(Clone, Copy)]
struct Agg {
    ns: i64,
    turnover: f64,
    volume: i64,
}

/// 逐笔按指定键分组：min(ns) / Kahan 求和(turnover) / 求和(volume)。
/// 返回按「组键升序」排列的聚合结果（pandas groupby 默认 sort=True）。
fn group_by(rows: &[(i64, f64, i64, i64, i64)], key_idx: usize) -> Vec<Agg> {
    let mut slot: HashMap<i64, usize> = HashMap::new();
    let mut keys: Vec<i64> = Vec::new();
    let mut agg: Vec<Agg> = Vec::new();
    let mut comp: Vec<f64> = Vec::new();
    for r in rows.iter() {
        let key = match key_idx {
            0 => r.4, // ask_order
            _ => r.3, // bid_order
        };
        let s = match slot.get(&key) {
            Some(&s) => s,
            None => {
                let s = agg.len();
                slot.insert(key, s);
                keys.push(key);
                agg.push(Agg {
                    ns: i64::MAX,
                    turnover: 0.0,
                    volume: 0,
                });
                comp.push(0.0);
                s
            }
        };
        if r.0 < agg[s].ns {
            agg[s].ns = r.0;
        }
        // Kahan（Neumaier）补偿求和，与 pandas group_sum 一致
        let y = r.1 - comp[s];
        let t = agg[s].turnover + y;
        comp[s] = t - agg[s].turnover - y;
        if comp[s].is_nan() {
            comp[s] = 0.0;
        }
        agg[s].turnover = t;
        agg[s].volume += r.2;
    }
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
    idx.iter().map(|&i| agg[i]).collect()
}

/// 由成交时间算出 8 个字符的数字计数（"HHMMSSffffff".slice(1,-3)）
#[inline]
fn digit_counts(ns: i64) -> [i32; 10] {
    let tod = clock_ns(ns);
    let h = tod / 3_600_000_000_000;
    let m = (tod / 60_000_000_000) % 60;
    let s = (tod / 1_000_000_000) % 60;
    let us = (tod / 1_000) % 1_000_000;
    let mut c = [0i32; 10];
    // 字符序列：时的个位, 分的十位, 分的个位, 秒的十位, 秒的个位, 微秒前三位
    let ds = [
        (h % 10) as usize,
        (m / 10) as usize,
        (m % 10) as usize,
        (s / 10) as usize,
        (s % 10) as usize,
        (us / 100_000) as usize,
        ((us / 10_000) % 10) as usize,
        ((us / 1_000) % 10) as usize,
    ];
    for d in ds.iter() {
        c[*d] += 1;
    }
    c
}

/// 行内计数向量的平方模（整数，f64 精确）
#[inline]
fn norm_sq(c: &[i32; 10]) -> f64 {
    let mut s = 0i64;
    for d in 0..10 {
        s += (c[d] as i64) * (c[d] as i64);
    }
    s as f64
}

#[inline]
fn dot_counts(a: &[i32; 10], b: &[i32; 10]) -> f64 {
    let mut s = 0i64;
    for d in 0..10 {
        s += (a[d] as i64) * (b[d] as i64);
    }
    s as f64
}

/// 一个窗口一侧的 sim 序列（30 条，每条长度 m；r<i 处为 NaN）
fn sim_series(counts: &[[i32; 10]], lag_fixed_one: bool) -> Vec<Vec<f64>> {
    let m = counts.len();
    let sq: Vec<f64> = counts.iter().map(|c| norm_sq(c)).collect();
    let sqr: Vec<f64> = sq.iter().map(|v| v.sqrt()).collect();
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(PARAM);
    for i in 0..PARAM {
        let lag = if lag_fixed_one { 1 } else { i };
        let mut v = vec![f64::NAN; m];
        if m > lag {
            for r in lag..m {
                let num = dot_counts(&counts[r], &counts[r - lag]);
                v[r] = num / (sqr[r] * sqr[r - lag]);
            }
        }
        out.push(v);
    }
    out
}

/// 把 30 条 sim 首尾拼成一条序列
fn flatten_sims(sims: &[Vec<f64>], buf: &mut Vec<f64>) {
    buf.clear();
    for s in sims.iter() {
        buf.extend_from_slice(s);
    }
}

/// autocorr(lag) = np.corrcoef 的 Pearson（成对完整观测）
fn autocorr(flat: &[f64], lag: usize) -> f64 {
    let l = flat.len();
    if lag >= l {
        return f64::NAN;
    }
    let a = &flat[lag..];
    let b = &flat[..l - lag];
    corrcoef2(a, b)
}

struct SessionOut {
    stat: [f64; 19],   // mean/std/skew ×2 + 10 个 corrK 差 + 3 个 ratio
    corr_full: [f64; 18],
    corr_part: [f64; 18],
}

/// 一段窗口里需要的 6 个相关值
fn corr6_real(col0: &[f64], col1: &[f64], exch: &[f64], turn: &[f64]) -> [f64; 6] {
    [
        welford_corr(col0, col1),
        welford_corr(col0, exch),
        welford_corr(col0, turn),
        welford_corr(col1, exch),
        welford_corr(col1, turn),
        welford_corr(exch, turn),
    ]
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let path = find_file(code, date)?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    // 逐笔：flag != 32；exchtime 走 pandas 的秒→纳秒转换再补 8 小时；
    // 只保留钟点在 [09:30:00, 14:57:00] 的行（旧代码的 between_time）。
    let lo = SESSIONS[0].1;
    let hi = SESSIONS[0].2;
    let mut rows: Vec<(i64, f64, i64, i64, i64)> = Vec::new(); // (ns, turnover, volume, bid, ask)
    let mut lines = content.lines();
    lines.next();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        split_fields(bytes, &mut f);
        if parse_i64_at(f[COL_FLAG]) == 32 {
            continue;
        }
        let us = parse_f64_pandas(f[COL_EXCHTIME]);
        let ns = sec_to_ns(us / 1e6) + 8 * 3_600 * 1_000_000_000;
        let c = clock_ns(ns);
        if c < lo || c > hi {
            continue;
        }
        rows.push((
            ns,
            parse_f64_pandas(f[COL_TURNOVER]),
            parse_i64_at(f[COL_VOLUME]),
            parse_i64_at(f[COL_BID]),
            parse_i64_at(f[COL_ASK]),
        ));
    }
    if rows.is_empty() {
        return None;
    }

    let asks = sort_desc(group_by(&rows, 0));
    let bids = sort_desc(group_by(&rows, 1));
    let n_a = asks.len();
    let n_b = bids.len();
    if n_a == 0 || n_b == 0 {
        return None;
    }

    // 数字计数、平方模
    let ca: Vec<[i32; 10]> = asks.iter().map(|a| digit_counts(a.ns)).collect();
    let cb: Vec<[i32; 10]> = bids.iter().map(|b| digit_counts(b.ns)).collect();
    let ta: Vec<f64> = asks.iter().map(|a| a.turnover).collect();
    let va: Vec<f64> = asks.iter().map(|a| a.volume as f64).collect();
    let tb: Vec<f64> = bids.iter().map(|b| b.turnover).collect();
    let vb: Vec<f64> = bids.iter().map(|b| b.volume as f64).collect();

    // part 子集：volume>100，再取 volume <= 其 0.7 分位
    let part_a = part_index(&va);
    let part_b = part_index(&vb);

    let colfull_a: Vec<f64> = (0..n_a).map(|i| i as f64).collect();
    let colfull_b: Vec<f64> = (0..n_b).map(|i| i as f64).collect();

    let mut flat_a: Vec<f64> = Vec::new();
    let mut flat_b: Vec<f64> = Vec::new();

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    let mut sess_outs: Vec<Option<SessionOut>> = Vec::with_capacity(5);

    for (_sess, slo, shi) in SESSIONS.iter() {
        // 窗口内全体行的位置（保持原顺序）
        let wa: Vec<usize> = (0..n_a)
            .filter(|&i| {
                let c = clock_ns(asks[i].ns);
                c >= *slo && c <= *shi
            })
            .collect();
        let wb: Vec<usize> = (0..n_b)
            .filter(|&i| {
                let c = clock_ns(bids[i].ns);
                c >= *slo && c <= *shi
            })
            .collect();
        let m_a = wa.len();
        let m_b = wb.len();
        if m_a < 5 || m_b < 5 {
            sess_outs.push(None);
            continue;
        }
        let cwa: Vec<[i32; 10]> = wa.iter().map(|&i| ca[i]).collect();
        let cwb: Vec<[i32; 10]> = wb.iter().map(|&i| cb[i]).collect();
        let sims_a = sim_series(&cwa, false);
        let sims_b = sim_series(&cwb, true);
        // 统计量（asks/bids 拼起来的那条序列）
        flatten_sims(&sims_a, &mut flat_a);
        flatten_sims(&sims_b, &mut flat_b);
        let a_mean = nan_mean(&flat_a);
        let a_std = nan_std(&flat_a);
        let a_skew = nan_skew(&flat_a);
        let ac: Vec<f64> = (1..=10).map(|k| autocorr(&flat_a, k)).collect();
        let b_mean = nan_mean(&flat_b);
        let b_std = nan_std(&flat_b);
        let b_skew = nan_skew(&flat_b);
        let bc: Vec<f64> = (1..=10).map(|k| autocorr(&flat_b, k)).collect();
        let mut stat = [f64::NAN; 19];
        stat[0] = a_mean;
        stat[1] = a_std;
        stat[2] = a_skew;
        stat[3] = b_mean;
        stat[4] = b_std;
        stat[5] = b_skew;
        for k in 0..10 {
            stat[6 + k] = ac[k] - bc[k];
        }
        stat[16] = (a_mean - b_mean) / (a_mean + b_mean);
        stat[17] = (a_std - b_std) / (a_std + b_std);
        stat[18] = (a_skew - b_skew) / (a_skew + b_skew);

        // exchtime 列 = sim_0，按位置填进长度 n（full）或 p（part）的表，多出来的位置是 NaN
        let sim0a = &sims_a[0];
        let sim0b = &sims_b[0];

        let full_a_exch = pad_to(sim0a, n_a);
        let full_b_exch = pad_to(sim0b, n_b);
        let cf = corr6_real(&colfull_a, &colfull_a, &full_a_exch, &ta);
        let cf2 = corr6_real(&colfull_b, &colfull_b, &full_b_exch, &tb);
        let mut corr_full = [f64::NAN; 18];
        corr_full[0..6].copy_from_slice(&cf);
        corr_full[6..12].copy_from_slice(&cf2);
        for k in 0..6 {
            corr_full[12 + k] = cf[k] - cf2[k];
        }

        // part：col0 = 0..p-1；col1 = 子集行在全表中的位置；exchtime = sim_0 的前 p 个
        let p_a = part_a.len();
        let p_b = part_b.len();
        let col0_a: Vec<f64> = (0..p_a).map(|i| i as f64).collect();
        let col1_a: Vec<f64> = part_a.iter().map(|&i| i as f64).collect();
        let col0_b: Vec<f64> = (0..p_b).map(|i| i as f64).collect();
        let col1_b: Vec<f64> = part_b.iter().map(|&i| i as f64).collect();
        let pa_exch = pad_to(sim0a, p_a);
        let pb_exch = pad_to(sim0b, p_b);
        let tpa: Vec<f64> = part_a.iter().map(|&i| ta[i]).collect();
        let tpb: Vec<f64> = part_b.iter().map(|&i| tb[i]).collect();
        let cp = corr6_real(&col0_a, &col1_a, &pa_exch, &tpa);
        let cp2 = corr6_real(&col0_b, &col1_b, &pb_exch, &tpb);
        let mut corr_part = [f64::NAN; 18];
        corr_part[0..6].copy_from_slice(&cp);
        corr_part[6..12].copy_from_slice(&cp2);
        for k in 0..6 {
            corr_part[12 + k] = cp[k] - cp2[k];
        }

        sess_outs.push(Some(SessionOut {
            stat,
            corr_full,
            corr_part,
        }));
    }

    // 拼输出：full 五段在前，part 五段在后；窗口不足 5 行的那一段输出 40 个 NaN
    for pass in 0..2 {
        for s in sess_outs.iter() {
            match s {
                None => {
                    for _ in 0..N_METRIC {
                        out.push(f32::NAN);
                    }
                }
                Some(so) => {
                    for v in so.stat.iter() {
                        out.push(*v as f32);
                    }
                    let corr = if pass == 0 { &so.corr_full } else { &so.corr_part };
                    for v in corr.iter() {
                        out.push(*v as f32);
                    }
                }
            }
        }
    }
    Some(out)
}

/// 按 turnover 降序（pandas 并列顺序）重排
fn sort_desc(agg: Vec<Agg>) -> Vec<Agg> {
    let vals: Vec<f64> = agg.iter().map(|a| a.turnover).collect();
    let perm = pandas_argsort_desc(&vals);
    perm.iter().map(|&i| agg[i]).collect()
}

/// part 子集的原始位置（升序）：volume>100 且 volume <= 该子集的 0.7 分位
fn part_index(vol: &[f64]) -> Vec<usize> {
    let first: Vec<usize> = (0..vol.len()).filter(|&i| vol[i] > 100.0).collect();
    if first.is_empty() {
        return first;
    }
    let mut sorted: Vec<f64> = first.iter().map(|&i| vol[i]).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = quantile_linear(&sorted, 0.7);
    first.into_iter().filter(|&i| vol[i] <= q).collect()
}

/// 把 sim_0 的前若干项填进长度 len 的列，不足补 NaN（pandas 的索引对齐赋值）
fn pad_to(src: &[f64], len: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    let k = src.len().min(len);
    v[..k].copy_from_slice(&src[..k]);
    v
}

/// hm033 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
