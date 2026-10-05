//! hm70「橘生淮南」——旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm70_橘生淮南/hm70_okay.py
//! 数据来源：Level2 盘口快照（run_base 里只读 read_market_data_single，
//! get_many=0 / read_full=0 / change_time=1，一整天的盘口一张表，窗口由旧代码自己切）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 get_factor 逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()
//!      读盘口 CSV 的全部 61 列（2018 及更早的文件末尾多一列 num，前 61 列位置不变），
//!      change_time=1 → deal_data：exchtime/localtime 由 epoch 微秒换成 Timestamp，
//!      即 ts = pd.to_timedelta(us/1e6, unit="s") + Timestamp("1970-01-01 08:00:00")。
//! 2) df = dwo.adjust_afternoon(df)
//!      以 exchtime 为索引切两段：[09:30:00, 11:30:00] 与 [13:00:00, 14:57:00]（两端都含，
//!      微秒精度），下午段索引整体减 90 分钟，然后 concat（上午段在前、各自保持原行序）。
//!      窗口之外的行整行丢掉；调整后所有时刻落在 09:30:00 ~ 13:27:00。
//! 3) df.exchtime = df.exchtime.to_numpy(int)     → 纳秒整数（i64，转 f64 用）
//! 4) df.volume = df.volume.diff()                → 相邻快照的累计成交量差分，首行 NaN
//! 5) cols 固定 27 列，顺序 = exchtime, last_prc, volume, weighted_ask_prc, weighted_bid_prc,
//!    total_ask_vol, total_bid_vol, ask_vol1, bid_vol1, ..., ask_vol10, bid_vol10
//! 6) 对 27 个排序列各做一次
//!        df_sort = df.sort_values(by=排序列)[cols]      ← 单键、升序、pandas 默认 quicksort
//!        corrs   = df_sort.corrwith(df_sort.shift(1))   ← 27 个 Pearson 相关系数
//!    排序键是 last_prc..bid_vol10 这 26 列（不含 exchtime），另有 exchtime 自己那一组
//!    单独算，只用来做后面的差值。
//! 7) 返回值 = 702 个 pri + 702 个 priabs + 702 个 minustimesort + 702 个 minustimesortabs
//!         pri            = 第 b 组排序、第 c 列的那 26×27 个相关系数
//!         priabs         = 上一条的绝对值
//!         minustimesort  = 上面的值 − 同列「按 exchtime 排序」的相关系数
//!         minustimesortabs = 上一条的绝对值
//!    顺序：外层按排序键（last_prc → bid_vol10），内层按 27 列。
//!
//! names_save（要做成 base 的 68 个）与 gap 名单原样搬进 hm70_okay_database.py。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、排序的并列顺序：逐行复刻 numpy 的 x86-simd-sort argsort（float64）
//! ════════════════════════════════════════════════════════════════════
//! `df.sort_values(by=col)` 是单键 → pandas nargsort → np.argsort(kind="quicksort")。
//! numpy 2.x 在支持 AVX512 的机器上对 float64 走 **x86-simd-sort** 而不是 npysort 的
//! introsort，并列顺序由它的「8 通道位元排序网络 + 分区」决定。
//!
//! 实测影响面（000001@20250303）：26 组排序键里有 663/702 个相关系数在
//! 「quicksort 并列顺序」与「稳定排序」之间相差超过 1e-4，最大绝对差 0.49，
//! 所以这里必须逐行复刻，不能换稳定排序（换 int64 版的做法见 hm039.rs）。
//!
//! 复刻分两块：
//!   · 主体照 src/okay_database/hm039.rs（那份已对 numpy 的 int64 路径验证过），
//!     只把键的类型从 i64 换成 f64、哨兵换成 f64::MAX；
//!   · 深度耗尽时 x86-simd-sort 会退回 std::sort（libstdc++ 的 introsort），
//!     hm039 当时用「按值稳定排序」代替、并在注释里说明没触发；hm70 会触发
//!     （并列极多时），所以这里把 libstdc++ 的 std::sort 也照抄了一份
//!     （阈值 16、median-of-3 分区、partial_sort 走堆排、最后一遍插入排序）。
//!
//! 验证：先在 sandbox_xss070 里对 170 组随机数组（含全相等、无并列、尺寸 1~4739）
//! 与 numpy 2.3.5 的 argsort 结果逐位对拍，全部一致；再走 verify_series.py 端到端。
//!
//! NaN 的处理：pandas 的 nargsort 先把 NaN 摘出来（排在最后、按原下标升序），
//! 只有非 NaN 的部分进 argsort，所以进本模块的数组里没有 NaN，
//! 比较一律用普通 `<`（不做 total_cmp）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、相关系数的算法（pandas corrwith 的逐项复刻）
//! ════════════════════════════════════════════════════════════════════
//! pandas 的 corrwith(method="pearson") 不是 nanops 的 corr，而是：
//!     left  = left + right*0 ; right = right + left*0      ← 成对掩掉 NaN
//!     ldem  = left - left.mean() ; rdem = right - right.mean()
//!     num   = (ldem * rdem).sum()
//!     dom   = (left.count() - 1) * left.std() * right.std()
//!     r     = num / dom
//! 全部是「按列顺序累加」（本机 numpy 对 C 序二维块沿 axis=0 求和就是顺序累加，
//! 实测与顺序循环逐位相同），所以这里用顺序求和即可；std 是两遍算法
//! （先算均值，再 Σ(均值−值)²/(n−1)）。成对有效性判据用 is_nan（inf 不算缺失，
//! 与 pandas 的 isna 一致）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、精度与已知差异
//! ════════════════════════════════════════════════════════════════════
//! · 盘口价格与成交量都按 f64 解析（不能用 fast_csv_reader 的 f32：这一族的量是
//!   相关系数之差，量级可以小到 1e-7，f32 会把小值整体抹平）。
//! · np.log / SVML 不涉及本系列。
//! · 已知残余差异：实测 4 组验收数据里最大相对误差 8.7e-9（出现在一个 6.2e-7 的小量上），
//!   来自顺序求和与 pandas 内部求和细节的末位差异，远低于 1e-4 验收线。

use std::fs::File;
use std::io::Read;

// ============================================================================
// 常量
// ============================================================================

/// 盘口 CSV 的列数（2018 及更早多为 62 列，末尾多的那一列不使用）
const MKT_NCOL: usize = 62;
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_COL_VOLUME: usize = 14;
const MKT_COL_WEIGHTED_ASK_PRC: usize = 17;
const MKT_COL_WEIGHTED_BID_PRC: usize = 18;
const MKT_COL_TOTAL_ASK_VOL: usize = 19;
const MKT_COL_TOTAL_BID_VOL: usize = 20;
/// ask_vol1=21+1、bid_vol1=21+3、ask_vol2=21+5 …… 每档 4 列
const MKT_COL_ASK_VOL_BASE: usize = 22;

const DAY_NS: i64 = 86_400_000_000_000;
/// 北京时间偏移：CSV 里的 exchtime 是 epoch 微秒，旧框架再加 8 小时
const BJ_OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
/// 下午段整体减 90 分钟
const SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;

const fn hm(h: i64, m: i64) -> i64 {
    (h * 3600 + m * 60) * 1_000_000_000
}

const MORNING_LO: i64 = hm(9, 30);
const MORNING_HI: i64 = hm(11, 30);
const AFTERNOON_LO: i64 = hm(13, 0);
const AFTERNOON_HI: i64 = hm(14, 57);

/// cols 的 27 个列名，顺序必须与旧代码里 cols 列表一致
const COL_NAMES: [&str; NCOL] = [
    "exchtime",
    "last_prc",
    "volume",
    "weighted_ask_prc",
    "weighted_bid_prc",
    "total_ask_vol",
    "total_bid_vol",
    "ask_vol1",
    "bid_vol1",
    "ask_vol2",
    "bid_vol2",
    "ask_vol3",
    "bid_vol3",
    "ask_vol4",
    "bid_vol4",
    "ask_vol5",
    "bid_vol5",
    "ask_vol6",
    "bid_vol6",
    "ask_vol7",
    "bid_vol7",
    "ask_vol8",
    "bid_vol8",
    "ask_vol9",
    "bid_vol9",
    "ask_vol10",
    "bid_vol10",
];

const NCOL: usize = 27;
/// 排序键的个数（去掉 exchtime 自己那一组）
const NBLOCK: usize = NCOL - 1;
pub const N_OUT: usize = 4 * NBLOCK * NCOL;

/// 27 列各自对应 CSV 里的列下标
fn csv_col(idx: usize) -> usize {
    match idx {
        0 => MKT_COL_EXCHTIME,
        1 => MKT_COL_LAST_PRC,
        2 => MKT_COL_VOLUME,
        3 => MKT_COL_WEIGHTED_ASK_PRC,
        4 => MKT_COL_WEIGHTED_BID_PRC,
        5 => MKT_COL_TOTAL_ASK_VOL,
        6 => MKT_COL_TOTAL_BID_VOL,
        _ => {
            // 7..27：ask_vol1, bid_vol1, ask_vol2, bid_vol2, ...
            let k = idx - 7;
            MKT_COL_ASK_VOL_BASE + (k / 2) * 4 + if k % 2 == 0 { 0 } else { 2 }
        }
    }
}

// ============================================================================
// 盘口 CSV 的读取与 adjust_afternoon
// ============================================================================

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

/// 读一天盘口，按旧代码 adjust_afternoon 的结果给出 27 列矩阵（行序 = 上午段接下午段）。
/// volume 列这里还是**累计量**的原始值，差分在调用处做（与旧代码顺序一致）。
fn read_market_rows(code: &str, date: i64) -> Option<Vec<[f64; NCOL]>> {
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

    let mut morning: Vec<[f64; NCOL]> = Vec::new();
    let mut afternoon: Vec<[f64; NCOL]> = Vec::new();

    let mut lines = content.lines();
    lines.next(); // 表头
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

        let ex_us = parse_f64_at(f[MKT_COL_EXCHTIME]);
        if ex_us.is_nan() {
            // pandas 侧是 NaT，between_time 会把这种行丢掉
            continue;
        }
        // 旧框架：to_timedelta(us/1e6, unit="s") + 1970-01-01 08:00:00
        let ns = ((ex_us / 1e6) * 1e9) as i64 + BJ_OFFSET_NS;
        let tod = ns.rem_euclid(DAY_NS);

        let mut row = [f64::NAN; NCOL];
        row[0] = ns as f64;
        for c in 1..NCOL {
            row[c] = parse_f64_at(f[csv_col(c)]);
        }

        if tod >= MORNING_LO && tod <= MORNING_HI {
            morning.push(row);
        } else if tod >= AFTERNOON_LO && tod <= AFTERNOON_HI {
            row[0] = (ns - SHIFT_NS) as f64;
            afternoon.push(row);
        }
    }

    let mut rows = morning;
    rows.append(&mut afternoon);
    let n = rows.len();
    if n > 0 {
        // df.volume = df.volume.diff()：首行 NaN，其余是相邻快照的差分
        for i in (1..n).rev() {
            rows[i][2] -= rows[i - 1][2];
        }
        rows[0][2] = f64::NAN;
    }
    Some(rows)
}

// ============================================================================
// numpy x86-simd-sort argsort(float64) 的逐行复刻
// ============================================================================

const LANES: usize = 8;
const KEY_MAX: f64 = f64::MAX;
const IDX_MAX: i64 = i64::MAX;

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
        // 结果来自 in1 就保留下标 1，否则用下标 2
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

fn bitonic_clean_n_vec(
    keys: &mut [[f64; LANES]],
    idxs: &mut [[i64; LANES]],
    base: usize,
    num_vecs: usize,
) {
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

fn bitonic_merge_n_vec(
    keys: &mut [[f64; LANES]],
    idxs: &mut [[i64; LANES]],
    base: usize,
    num_vecs: usize,
) {
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

/// argsort_n&lt;vtype, argtype, 256&gt; 的 numVecs 会按 N 递归减半。
fn argsort_n_vec(arr: &[f64], ind: &mut [i64], n: usize, num_vecs: usize) {
    let mut num_vecs = num_vecs;
    while num_vecs > 1 && n * 2 <= num_vecs * LANES {
        num_vecs /= 2;
    }
    let mut keys = vec![[0.0f64; LANES]; num_vecs];
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
        let num_to_read = if n > i * LANES {
            (n - i * LANES).min(LANES)
        } else {
            0
        };
        for l in 0..LANES {
            if l < num_to_read {
                let v = ind[i * LANES + l];
                idxs[i][l] = v;
                keys[i][l] = arr[v as usize];
            } else {
                // indexType::zmm_max() / keyType::zmm_max()：补齐的通道放最大值
                idxs[i][l] = IDX_MAX;
                keys[i][l] = KEY_MAX;
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
        let num_to_write = if n > i * LANES {
            (n - i * LANES).min(LANES)
        } else {
            0
        };
        for l in 0..num_to_write {
            ind[i * LANES + l] = idxs[i][l];
        }
    }
}

fn argsort_n(arr: &[f64], ind: &mut [i64], left: usize, n: usize) {
    let mut sub: Vec<i64> = ind[left..left + n].to_vec();
    argsort_n_vec(arr, &mut sub, n, 256 / LANES);
    ind[left..left + n].copy_from_slice(&sub);
}

/// get_pivot_64bit（numlanes=8 的 median-of-8 分支）
fn get_pivot(arr: &[f64], ind: &[i64], left: usize, right: usize) -> f64 {
    if right - left >= LANES {
        let size = (right - left) / LANES;
        // _mm512_set_epi64(v1..v8)：通道 0 放最后一个参数
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

#[inline]
fn upd_minmax(smallest: &mut f64, biggest: &mut f64, v: f64) {
    if v < *smallest {
        *smallest = v;
    }
    if *biggest < v {
        *biggest = v;
    }
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
        upd_minmax(smallest, biggest, v);
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
            upd_minmax(smallest, biggest, cv[l]);
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
            upd_minmax(smallest, biggest, cv[l]);
        }
        r_store -= amount;
        l_store += LANES - amount;
    }
    let amount = partition_vec(ind, l_store, r_store + LANES, &av_left, &cv_left, pivot);
    for l in 0..LANES {
        upd_minmax(smallest, biggest, cv_left[l]);
    }
    l_store += LANES - amount;
    let amount = partition_vec(ind, l_store, l_store + LANES, &av_right, &cv_right, pivot);
    for l in 0..LANES {
        upd_minmax(smallest, biggest, cv_right[l]);
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
        upd_minmax(smallest, biggest, v);
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
            for l in 0..LANES {
                upd_minmax(smallest, biggest, cvs[ii][l]);
            }
            l_store += LANES - amount;
            r_store -= amount;
        }
    }
    for ii in 0..NUM_UNROLL {
        let amount = partition_vec(ind, l_store, r_store + LANES, &iv_left[ii], &kv_left[ii], pivot);
        for l in 0..LANES {
            upd_minmax(smallest, biggest, kv_left[ii][l]);
        }
        l_store += LANES - amount;
        r_store -= amount;
    }
    for ii in 0..NUM_UNROLL {
        let amount =
            partition_vec(ind, l_store, r_store + LANES, &iv_right[ii], &kv_right[ii], pivot);
        for l in 0..LANES {
            upd_minmax(smallest, biggest, kv_right[ii][l]);
        }
        l_store += LANES - amount;
        r_store -= amount;
    }
    l_store
}

fn argsort_rec(arr: &[f64], ind: &mut [i64], left: usize, right: i64, max_iters: i64) {
    if (right + 1) as usize - left <= 256 {
        argsort_n(arr, ind, left, (right + 1) as usize - left);
        return;
    }
    if max_iters <= 0 {
        // x86-simd-sort 的兜底：std::sort(arg+low, arg+high+1, [arr](a,b){return arr[a]<arr[b];})
        gcc_std_sort(ind, left, right as usize + 1, arr);
        return;
    }
    let (l, r) = (left, right as usize);
    let pivot = get_pivot(arr, ind, l, r);
    let mut smallest = KEY_MAX;
    let mut biggest = -KEY_MAX;
    let pivot_index = argpartition_unrolled(arr, ind, l, r + 1, pivot, &mut smallest, &mut biggest);
    if pivot != smallest {
        argsort_rec(arr, ind, left, pivot_index as i64 - 1, max_iters - 1);
    }
    if pivot != biggest {
        argsort_rec(arr, ind, pivot_index, right, max_iters - 1);
    }
}

// ============================================================================
// libstdc++ std::sort（introsort）——x86-simd-sort 深度耗尽时的兜底
// ============================================================================

const GCC_THRESHOLD: usize = 16;

#[inline]
fn ig(ind: &[i64], arr: &[f64], i: usize) -> f64 {
    arr[ind[i] as usize]
}

#[inline]
fn gcc_lg(n: usize) -> i32 {
    63 - (n as u64).leading_zeros() as i32
}

fn gcc_move_median_to_first(
    ind: &mut [i64],
    arr: &[f64],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
) {
    let (ka, kb, kc) = (ig(ind, arr, a), ig(ind, arr, b), ig(ind, arr, c));
    if ka < kb {
        if kb < kc {
            ind.swap(result, b);
        } else if ka < kc {
            ind.swap(result, c);
        } else {
            ind.swap(result, a);
        }
    } else if ka < kc {
        ind.swap(result, a);
    } else if kb < kc {
        ind.swap(result, c);
    } else {
        ind.swap(result, b);
    }
}

fn gcc_unguarded_partition(
    ind: &mut [i64],
    arr: &[f64],
    mut first: usize,
    mut last: usize,
    pivot: usize,
) -> usize {
    loop {
        while ig(ind, arr, first) < ig(ind, arr, pivot) {
            first += 1;
        }
        last -= 1;
        while ig(ind, arr, pivot) < ig(ind, arr, last) {
            last -= 1;
        }
        if !(first < last) {
            return first;
        }
        ind.swap(first, last);
        first += 1;
    }
}

fn gcc_unguarded_partition_pivot(
    ind: &mut [i64],
    arr: &[f64],
    first: usize,
    last: usize,
) -> usize {
    let mid = first + (last - first) / 2;
    gcc_move_median_to_first(ind, arr, first, first + 1, mid, last - 1);
    gcc_unguarded_partition(ind, arr, first + 1, last, first)
}

fn gcc_push_heap(ind: &mut [i64], arr: &[f64], mut hole: usize, top: usize, value: i64) {
    while hole > top {
        let parent = (hole - 1) / 2;
        if !(ig(ind, arr, parent) < arr[value as usize]) {
            break;
        }
        ind[hole] = ind[parent];
        hole = parent;
    }
    ind[hole] = value;
}

fn gcc_adjust_heap(ind: &mut [i64], arr: &[f64], mut hole: usize, len: usize, value: i64) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if ig(ind, arr, second) < ig(ind, arr, second - 1) {
            second -= 1;
        }
        ind[hole] = ind[second];
        hole = second;
    }
    if (len & 1) == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        ind[hole] = ind[second - 1];
        hole = second - 1;
    }
    gcc_push_heap(ind, arr, hole, top, value);
}

fn gcc_pop_heap(ind: &mut [i64], arr: &[f64], first: usize, last: usize, result: usize) {
    let value = ind[result];
    ind[result] = ind[first];
    gcc_adjust_heap(ind, arr, first, last - first, value);
}

fn gcc_make_heap(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    if last - first < 2 {
        return;
    }
    let len = last - first;
    let mut parent = (len - 2) / 2;
    loop {
        let value = ind[first + parent];
        gcc_adjust_heap(ind, arr, parent, len, value);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn gcc_partial_sort(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    gcc_make_heap(ind, arr, first, last);
    let mut hi = last;
    while hi - first > 1 {
        hi -= 1;
        gcc_pop_heap(ind, arr, first, hi, hi);
    }
}

fn gcc_unguarded_linear_insert(ind: &mut [i64], arr: &[f64], last: usize) {
    let val = ind[last];
    let mut cur = last;
    let mut nxt = last - 1;
    while arr[val as usize] < ig(ind, arr, nxt) {
        ind[cur] = ind[nxt];
        cur = nxt;
        nxt -= 1;
    }
    ind[cur] = val;
}

fn gcc_insertion_sort(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        if ig(ind, arr, i) < ig(ind, arr, first) {
            let val = ind[i];
            let mut j = i;
            while j > first {
                ind[j] = ind[j - 1];
                j -= 1;
            }
            ind[first] = val;
        } else {
            gcc_unguarded_linear_insert(ind, arr, i);
        }
    }
}

fn gcc_unguarded_insertion_sort(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    for i in first..last {
        gcc_unguarded_linear_insert(ind, arr, i);
    }
}

fn gcc_introsort_loop(ind: &mut [i64], arr: &[f64], first: usize, last_in: usize, mut depth: i32) {
    let mut last = last_in;
    while last - first > GCC_THRESHOLD {
        if depth == 0 {
            gcc_partial_sort(ind, arr, first, last);
            return;
        }
        depth -= 1;
        let cut = gcc_unguarded_partition_pivot(ind, arr, first, last);
        gcc_introsort_loop(ind, arr, cut, last, depth);
        last = cut;
    }
}

/// libstdc++ std::sort：introsort + 最后一遍插入排序
fn gcc_std_sort(ind: &mut [i64], first: usize, last: usize, arr: &[f64]) {
    if first == last {
        return;
    }
    let n = last - first;
    gcc_introsort_loop(ind, arr, first, last, 2 * gcc_lg(n));
    if n > GCC_THRESHOLD {
        gcc_insertion_sort(ind, arr, first, first + GCC_THRESHOLD);
        gcc_unguarded_insertion_sort(ind, arr, first + GCC_THRESHOLD, last);
    } else {
        gcc_insertion_sort(ind, arr, first, last);
    }
}

/// numpy 2.3.5 在 AVX512 机器上 `np.argsort(float64_array, kind="quicksort")` 的升序下标。
/// 前提：数组里没有 NaN（pandas 的 nargsort 会先把 NaN 摘掉再调 argsort）。
fn xss_argsort_f64(arr: &[f64]) -> Vec<i64> {
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

// ============================================================================
// 输出
// ============================================================================

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for suffix in ["_pri", "_priabs", "_minustimesort", "_minustimesortabs"] {
        for b in 1..NCOL {
            for c in 0..NCOL {
                out.push(format!(
                    "{}_autocorr_sortby_{}{}",
                    COL_NAMES[c], COL_NAMES[b], suffix
                ));
            }
        }
    }
    out
}

/// pandas nargsort(ascending=True, na_position="last") 的复刻：
/// NaN 先摘出来排在最后（按原下标升序），其余走 x86-simd-sort 的 argsort。
fn pandas_argsort(vals: &[f64]) -> Vec<usize> {
    let mut nn: Vec<f64> = Vec::with_capacity(vals.len());
    let mut nni: Vec<usize> = Vec::with_capacity(vals.len());
    let mut nan_idx: Vec<usize> = Vec::new();
    for (i, &v) in vals.iter().enumerate() {
        if v.is_nan() {
            nan_idx.push(i);
        } else {
            nn.push(v);
            nni.push(i);
        }
    }
    let order = xss_argsort_f64(&nn);
    let mut out: Vec<usize> = Vec::with_capacity(vals.len());
    for &o in order.iter() {
        out.push(nni[o as usize]);
    }
    out.extend(nan_idx);
    out
}

/// 行序取 order 之后，第 c 列与它的 shift(1) 的 Pearson 相关系数。
///
/// 与 pandas corrwith 逐项对应：成对有效的位置才进 sum/count，
/// 均值与平方偏差和都是顺序累加，dom = (n-1)*std_x*std_y。
fn corr_at_order(rows: &[[f64; NCOL]], order: &[usize], c: usize) -> f64 {
    let n = order.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut cnt = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut prev = rows[order[0]][c];
    for &oi in order.iter().skip(1) {
        let cur = rows[oi][c];
        if !cur.is_nan() && !prev.is_nan() {
            cnt += 1;
            sx += cur;
            sy += prev;
        }
        prev = cur;
    }
    if cnt == 0 {
        return f64::NAN;
    }
    let mx = sx / cnt as f64;
    let my = sy / cnt as f64;
    let mut num = 0.0f64;
    let mut qx = 0.0f64;
    let mut qy = 0.0f64;
    let mut prev = rows[order[0]][c];
    for &oi in order.iter().skip(1) {
        let cur = rows[oi][c];
        if !cur.is_nan() && !prev.is_nan() {
            num += (cur - mx) * (prev - my);
            let a = mx - cur;
            let b = my - prev;
            qx += a * a;
            qy += b * b;
        }
        prev = cur;
    }
    if cnt <= 1 {
        return f64::NAN;
    }
    let d = (cnt - 1) as f64;
    let stdx = (qx / d).sqrt();
    let stdy = (qy / d).sqrt();
    num / ((d * stdx) * stdy)
}

fn block_corrs(rows: &[[f64; NCOL]], key_idx: usize) -> Vec<f64> {
    let key: Vec<f64> = rows.iter().map(|r| r[key_idx]).collect();
    let order = pandas_argsort(&key);
    (0..NCOL).map(|c| corr_at_order(rows, &order, c)).collect()
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_market_rows(code, date)?;
    if rows.is_empty() {
        return None;
    }

    // 26 组排序键 × 27 列
    let mut res: Vec<f64> = Vec::with_capacity(NBLOCK * NCOL);
    for b in 1..NCOL {
        res.extend(block_corrs(&rows, b));
    }
    // exchtime 自己那一组（只用于做差）
    let ex = block_corrs(&rows, 0);

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for &v in res.iter() {
        out.push(v as f32);
    }
    for &v in res.iter() {
        out.push(v.abs() as f32);
    }
    for b in 0..NBLOCK {
        for c in 0..NCOL {
            out.push((res[b * NCOL + c] - ex[c]) as f32);
        }
    }
    for b in 0..NBLOCK {
        for c in 0..NCOL {
            out.push((res[b * NCOL + c] - ex[c]).abs() as f32);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// 本系列是逐股（per_stock），没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
