//! hm078「万年革命」—— 旧框架模块级 go(date, code) 的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm78_万年革命/hm78_okay.py
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 go 做的事（逐步对照，写代码前先抄在这里）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = rp.adjust_afternoon(rp.read_market(code, date))
//!       .set_index("exchtime").resample("3s").last().ffill()
//!    `read_market` 只做两件事：读整张 market_data CSV（61 列），
//!    exchtime 换成「1970-01-01 08:00:00 + exchtime_us/1e6 秒」的 Timestamp，
//!    然后丢掉 ask_prc1 == 0 或 bid_prc1 == 0 的行（涨跌停/无效快照）。
//!    `adjust_afternoon` 与逐笔那套完全一样：只留 09:30~11:30 与 13:00~14:57
//!    两段（钟点闭区间），下午段整体前移 90 分钟，上午段在前、下午段在后拼接。
//!    `resample("3s")` 的 bin 从**当日 00:00:00** 起按 3 秒对齐（pandas 的
//!    origin 默认 start_day）、label 取左边界、closed 左闭右开；`.last()` 是
//!    「每个 bin 里每列最后一个非空值」，空 bin 给 NaN，`.ffill()` 再把 NaN 往前填。
//! 2) df = df[df.last_prc != 0]；df.volume = df.volume.diff() 再除以全列和。
//! 3) twins = df[["last_prc", "volume"]].reset_index(drop=True)
//! 4) res = compute(df.volume, df.volume, twins) + compute(df.last_prc, df.volume, twins)
//!
//! compute(arr_a, arr_b, twins) 里：
//!   raw = rp.local_correlation(arr_a, arr_b, 20)      # (n, 12) 的 f64 矩阵
//!   df  = DataFrame(raw[0], columns=ORIG_COLS).dropna()
//!   再补 12 个派生列（比值、绝对值差），得到 24 列的 BASE_COLS；
//!   stat_df = {mean, std, skew, kurt, max, ac, trend} 七行 × 24 列，
//!     其中 ac = df.corrwith(df.shift(1))、trend = rp.trend_2d(..., axis=0)；
//!   res 的前 168 项 = [stat_df.loc[c, s] for s in STATS for c in BASE_COLS]；
//!   然后把 twins 里 index 落在 df.index 的那些行并进来，dropna 后取
//!   `df_full.corr()` 的严格上三角（26 列 → 325 个），再拼一份取绝对值（325 个）。
//!   一份 818，两份共 1636，最后按 names_save 取 82 个返回。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、本文件的实现选择
//! ════════════════════════════════════════════════════════════════════
//! · 口径全部按 pandas：这是「旧库 v.s. 新库」的对账，差 1e-4 就算不通过。
//! · local_correlation 直接用 src/statistics/local_correlation.rs 里同一个引擎
//!   （新加的纯 Rust 入口 local_correlation_inner），不复制一份，
//!   免得以后两边逻辑漂移。旧 go 调的 rp.local_correlation 就是那个 pyfunction。
//! · market_data CSV 自己按 f64 解析（列 4=exchtime、6=last_prc、14=volume、
//!   21=ask_prc1、23=bid_prc1）。浮点用 pandas 的 precise_xstrtod 复刻
//!   （hm033.rs / hm090.rs 里也各有一份）：只有在「精确等于 0」被当判定条件时
//!   才致命，这里的 ask_prc1/bid_prc1/last_prc != 0 恰好就是这种判定。
//! · 时间戳照抄 pandas 的 cast_from_unit_vectorized（hm064.rs 里的 pandas_ns_of
//!   同式）：`exchtime_us / 1e6` 先变成秒的 f64，再拆成整数秒 + rint(frac*1e9)/1e9，
//!   自带 ±120ns 噪声。between_time / resample 的边界判定都落在这个纳秒值上，
//!   用「微秒 × 1000」会差几个纳秒，正好卡在 3 秒 bin 边界上。
//! · 统计量：mean/std/skew/kurt 照 pandas nanops 的算法（hm064.rs 的同名函数同式），
//!   null 判定用 notna（inf 算有效）；相关矩阵 `df_full.corr()` 走 pandas 的
//!   libalgos.nancorr（Welford 在线算法、mask 用 isfinite、min_periods=1），
//!   `corrwith` 走 nanops.nancorr → np.corrcoef（两遍法）。这两条路不一样，别混。
//!
//! 验证：hm78_okay.py 对 20250303:000001 / 20250304:600519 / 20180601:000002 /
//! 20231020:300750 四组，82 个值全部一致（rtol 1e-4）。

use crate::data_paths;
use crate::statistics::local_correlation::local_correlation_inner;
use std::collections::BTreeMap;

/// local_correlation 的窗口大小（旧 go 写死 20）。
const WINDOW: usize = 20;
/// 每份特征的列数：12 个原始列 + 12 个派生列。
const NBASE: usize = 24;
/// 相关矩阵的列数：24 个 BASE_COLS + last_prc + volume。
const NALL: usize = 26;
/// 每份特征的取值个数：7×24 统计 + 325 相关 + 325 绝对相关。
const NHALF: usize = 7 * NBASE + NALL * (NALL - 1) / 2 * 2;
/// 两只手拼起来的总长度（vv 一份 + pv 一份）。
const FULL: usize = 2 * NHALF;

const ORIG_COLS: [&str; 12] = [
    "bk_max_r",
    "bk_max_v",
    "bk_min_r",
    "bk_min_v",
    "bk_btw_r",
    "bk_btw_v",
    "fw_max_r",
    "fw_max_v",
    "fw_min_r",
    "fw_min_v",
    "fw_btw_r",
    "fw_btw_v",
];

const BASE_COLS: [&str; NBASE] = [
    "bk_max_r",
    "bk_max_v",
    "bk_min_r",
    "bk_min_v",
    "bk_btw_r",
    "bk_btw_v",
    "fw_max_r",
    "fw_max_v",
    "fw_min_r",
    "fw_min_v",
    "fw_btw_r",
    "fw_btw_v",
    "r_bk_rev",
    "r_bk_sl",
    "r_fw_rev",
    "r_fw_sl",
    "v_bk_rev",
    "v_bk_sl",
    "v_fw_rev",
    "v_fw_sl",
    "r_bk_sld",
    "r_fw_sld",
    "v_bk_sld",
    "v_fw_sld",
];

/// stat_df 的取值顺序（空 ROW_ORDER 的取值顺序就是它）。
const STATS: [&str; 7] = ["mean", "std", "skew", "kurt", "max", "ac", "trend"];

// NAMES 是旧 go() 按 names_save_unfold 取出的 82 项。
// 其中 vv_r_bk_sld_x_v_bk_sld_abs 与 vv_last_prc_x_volume 各出现两次
// （一次是原值、一次来自它自己的 _fold 版本），按规范第二次加 _dup2。
// IDX 是这 82 项在 1636 长向量里的下标，下标由 names() 的构造式直接生成后逐项查出来的。
const NAMES: [&str; 82] = [
    "pv_fw_min_v_x_v_fw_rev_abs",
    "vv_v_bk_rev_ac",
    "vv_bk_max_v_x_r_bk_sld_abs",
    "pv_fw_min_r_x_v_fw_sld",
    "vv_v_bk_sl_x_volume",
    "vv_v_bk_rev_x_r_bk_sld",
    "vv_fw_min_v_mean",
    "vv_bk_max_v_trend",
    "pv_v_fw_rev_x_v_fw_sl",
    "vv_r_bk_sld_x_v_bk_sld_abs",
    "vv_bk_min_r_x_bk_btw_v",
    "pv_v_fw_sl_std",
    "vv_v_bk_rev_std",
    "vv_r_bk_sl_x_volume_abs",
    "vv_v_bk_sl_kurt",
    "vv_bk_min_v_x_fw_min_v",
    "vv_r_bk_rev_x_v_bk_sld_abs",
    "vv_bk_max_v_mean",
    "pv_fw_btw_r_x_v_fw_sld",
    "pv_fw_btw_v_x_r_fw_sl",
    "pv_fw_min_v_max",
    "pv_bk_btw_v_x_v_bk_rev",
    "vv_bk_min_r_x_last_prc",
    "pv_fw_btw_v_x_volume",
    "vv_bk_btw_v_x_v_fw_sld",
    "vv_last_prc_x_volume",
    "vv_fw_max_v_ac",
    "vv_fw_min_v_x_v_bk_sld_abs",
    "vv_bk_max_v_x_r_bk_sld",
    "vv_bk_min_v_x_r_fw_sld",
    "vv_bk_min_v_x_v_fw_sld_abs",
    "vv_bk_btw_v_x_v_bk_sl_abs",
    "vv_v_bk_sl_x_volume_abs",
    "pv_v_fw_sl_x_volume",
    "pv_v_bk_rev_max",
    "vv_r_bk_sl_x_r_bk_sld_abs",
    "vv_fw_min_v_max",
    "vv_v_fw_sl_max",
    "vv_bk_max_v_x_bk_min_r",
    "vv_r_bk_rev_x_volume",
    "vv_v_bk_rev_x_r_bk_sld_abs",
    "vv_r_bk_sld_x_v_bk_sld_abs_dup2",
    "vv_fw_max_v_x_v_bk_sl",
    "pv_bk_max_v_x_r_fw_rev",
    "vv_r_bk_rev_x_v_bk_rev",
    "vv_v_bk_sld_x_volume",
    "vv_r_fw_sld_x_v_bk_sld_abs",
    "vv_fw_max_v_x_last_prc",
    "vv_r_bk_sl_x_v_bk_sld_abs",
    "vv_r_bk_sld_skew",
    "vv_bk_min_v_x_r_bk_rev",
    "vv_bk_min_v_trend",
    "vv_fw_max_v_max",
    "vv_fw_min_v_std",
    "pv_v_fw_sl_mean",
    "vv_bk_max_r_std",
    "vv_bk_max_r_x_bk_min_v",
    "vv_last_prc_x_volume_dup2",
    "pv_r_bk_sld_x_last_prc",
    "vv_fw_btw_r_x_fw_btw_v",
    "vv_bk_btw_v_x_v_bk_rev",
    "vv_fw_max_r_trend",
    "vv_bk_btw_r_x_volume",
    "vv_v_bk_sld_trend",
    "vv_fw_max_v_x_volume",
    "vv_bk_max_r_x_fw_max_r_abs",
    "vv_bk_max_r_x_fw_max_r",
    "pv_r_fw_rev_kurt",
    "pv_fw_min_v_x_fw_btw_v_abs",
    "vv_bk_max_v_x_fw_min_v",
    "vv_v_fw_sld_mean",
    "vv_v_bk_sl_ac",
    "pv_fw_btw_r_x_r_fw_sl",
    "vv_r_bk_sl_x_volume",
    "vv_fw_min_v_x_v_bk_sld",
    "pv_v_fw_rev_x_volume",
    "vv_fw_btw_r_x_v_bk_sld_abs",
    "vv_bk_max_r_x_fw_min_v",
    "vv_bk_btw_v_x_fw_min_r",
    "vv_fw_max_v_skew",
    "vv_bk_min_r_max",
    "vv_v_fw_sld_max",
];

const IDX: [usize; 82] = [
    1508, 136, 536, 1172, 464, 451, 9, 145, 1283, 804, 219, 861, 40, 751, 89, 245, 736, 1, 1203,
    1209, 923, 1111, 238, 1219, 300, 492, 127, 694, 211, 257, 584, 619, 789, 1295, 930, 746, 105,
    115, 193, 414, 776, 804, 331, 1023, 405, 489, 808, 338, 748, 68, 248, 147, 103, 33, 837, 24,
    170, 492, 1299, 373, 293, 150, 282, 166, 339, 498, 173, 904, 1501, 200, 23, 137, 1195, 426,
    369, 1289, 709, 176, 285, 55, 98, 119,
];

pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}

// ============================================================================
// 全 1636 项的列名构造（与旧 go 里的 _make_names 逐字一致）
// ============================================================================

/// 旧代码 `_make_names(prefix)` 的复刻：
///   7 个统计 × 24 列 → 168 个 `{prefix}_{列}_{统计}`；
///   26 列的严格上三角 → 325 个 `{prefix}_{列i}_x_{列j}`；
///   再拼同样 325 个 `_abs`。
fn make_names(prefix: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(NHALF);
    for s in STATS.iter() {
        for c in BASE_COLS.iter() {
            out.push(format!("{}_{}_{}", prefix, c, s));
        }
    }
    let all: Vec<String> = BASE_COLS
        .iter()
        .map(|s| s.to_string())
        .chain(["last_prc".to_string(), "volume".to_string()])
        .collect();
    let mut corr = Vec::with_capacity(NALL * (NALL - 1) / 2);
    for i in 0..NALL {
        for j in (i + 1)..NALL {
            corr.push(format!("{}_{}_x_{}", prefix, all[i], all[j]));
        }
    }
    for n in corr.iter() {
        out.push(n.clone());
    }
    for n in corr.iter() {
        out.push(format!("{}_abs", n));
    }
    debug_assert_eq!(out.len(), NHALF);
    out
}

// ============================================================================
// 读盘口快照：rp.read_market + adjust_afternoon + resample("3s").last().ffill()
// ============================================================================

/// C 的 rint：舍到最近的整数，正好 .5 时取偶数（与 numpy 的 rint 一致）。
#[inline]
fn rint(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = x.floor();
    let d = x - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// 复刻 pandas `cast_from_unit_vectorized` 的纳秒换算（与 hm064.rs 的 pandas_ns_of 同式）。
///
/// 旧代码拿到的是 `pd.to_timedelta(exchtime_us / 1e6, unit="s") + Timestamp("1970-01-01 08:00:00")`，
/// 本函数返回的是**不含那 8 小时**的纳秒值（就是 exchtime 对应的 `ns since epoch`），
/// 调用方再补 8 小时，两者相加是整数运算，与 pandas 完全一致。
/// 这条链自带 ±120ns 噪声，3 秒 bin 的边界判定就落在这几个纳秒上。
#[inline]
fn pandas_ns_of(exchtime_us: i64) -> i64 {
    let val = (exchtime_us as f64) / 1_000_000.0;
    let base = val as i64; // C 的转 int64 = 向零截断
    let frac = val - (base as f64);
    let fr = rint(frac * 1e9) / 1e9;
    base * 1_000_000_000 + (fr * 1e9) as i64
}

/// 去掉字段两端空白（含行尾 `\r`）。
fn trim_ascii(raw: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = raw.len();
    while s < e && raw[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && raw[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    &raw[s..e]
}

/// 取字段的整数前缀（exchtime 是 `1740960939000000.0` 这种浮点串）。
fn int_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    let mut i = 0usize;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
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
/// 与 src/okay_database/hm033.rs、hm090.rs 里的同名函数一致，这里复制一份免得跨系列互相依赖。
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
        f64::INFINITY
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

/// 字段 → f64（空字段给 NaN，与 pandas 一致）。
fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
}

/// 读一个交易日的盘口快照，复刻旧 go 的那一串前处理，返回 (last_prc 序列, volume 序列)。
///
/// 步骤与 pandas 一一对应：
///   read_market → 丢 ask_prc1==0 / bid_prc1==0 → adjust_afternoon（微秒闭区间 + 下午前移）
///   → resample("3s").last()（bin 从当日 00:00 起、左闭右开、每列取最后一个非空）
///   → ffill → df[last_prc != 0] → volume.diff() → volume / volume.sum()
fn load_series(code: &str, date: i64) -> Option<(Vec<f64>, Vec<f64>)> {
    const COL_EXCHTIME: usize = 4;
    const COL_LAST_PRC: usize = 6;
    const COL_VOLUME: usize = 14;
    const COL_ASK_PRC1: usize = 21;
    const COL_BID_PRC1: usize = 23;
    const NCOLS: usize = 61;

    const NS_PER_S: i64 = 1_000_000_000;
    /// 旧代码给时间戳加的 8 小时（`Timestamp("1970-01-01 08:00:00")`）。
    const OFFSET_NS: i64 = 8 * 3600 * NS_PER_S;
    /// 3 秒 bin。
    const BIN_NS: i64 = 3 * NS_PER_S;
    /// 下午前移 90 分钟。
    const SHIFT_NS: i64 = 90 * 60 * NS_PER_S;
    const LD_0930: i64 = (9 * 3600 + 30 * 60) * NS_PER_S;
    const LD_1130: i64 = (11 * 3600 + 30 * 60) * NS_PER_S;
    const LD_1300: i64 = 13 * 3600 * NS_PER_S;
    const LD_1457: i64 = (14 * 3600 + 57 * 60) * NS_PER_S;

    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut data = None;
    for root in data_paths::level2_roots() {
        let path = root
            .join(date.to_string())
            .join("market_data")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    // bin → (last_prc, volume) 两个槽，各自记「该列最后一个非空值」。
    // BTreeMap 的键是纳秒时间戳整除 3e9 的商，天然按时间升序，正好是 resample 的行序。
    let mut bins: BTreeMap<i64, (Option<f64>, Option<f64>)> = BTreeMap::new();

    let mut morning: Vec<(i64, f64, f64)> = Vec::new();
    let mut afternoon: Vec<(i64, f64, f64)> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue; // 表头 / 空行
        }
        let mut fields: [&[u8]; NCOLS] = [&[][..]; NCOLS];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < NCOLS {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < NCOLS {
            fields[commas] = &line[start..];
        }
        if commas + 1 < NCOLS {
            continue;
        }

        // rp.read_market 的 with_high_low_limited=0：ask_prc1 或 bid_prc1 为 0 的行丢掉。
        // 注意 pandas 里空字段是 NaN，NaN != 0 为真，所以空字段的行**保留**——这里同式。
        if num(fields[COL_ASK_PRC1]) == 0.0 || num(fields[COL_BID_PRC1]) == 0.0 {
            continue;
        }

        let exchtime_us = int_prefix(fields[COL_EXCHTIME]);
        if exchtime_us == 0 {
            continue;
        }
        // pandas 把 exchtime（f64 的微秒）除以 1e6 换成秒，再走 cast_from_unit_vectorized。
        let tns = OFFSET_NS + pandas_ns_of(exchtime_us);
        let tod = tns.rem_euclid(86_400 * NS_PER_S);
        let last_prc = num(fields[COL_LAST_PRC]);
        let volume = num(fields[COL_VOLUME]);
        if (LD_0930..=LD_1130).contains(&tod) {
            morning.push((tns, last_prc, volume));
        } else if (LD_1300..=LD_1457).contains(&tod) {
            afternoon.push((tns - SHIFT_NS, last_prc, volume));
        }
    }

    // pd.concat([df1, df2])：上午段全部在前、下午段全部在后（各自保持文件序）。
    // 同 bin 的行后写覆盖先写，正是 `resample.last()` 的「最后一个非空」。
    for (tns, prc, vol) in morning.into_iter().chain(afternoon.into_iter()) {
        let key = tns.div_euclid(BIN_NS);
        let slot = bins.entry(key).or_insert((None, None));
        if !prc.is_nan() {
            slot.0 = Some(prc);
        }
        if !vol.is_nan() {
            slot.1 = Some(vol);
        }
    }
    if bins.is_empty() {
        return None;
    }

    // `.ffill()`：从前一个非空 bin 拿值；开头没有前值的仍是 NaN。
    let first = *bins.keys().next().unwrap();
    let last = *bins.keys().next_back().unwrap();
    let mut prc_series: Vec<f64> = Vec::new();
    let mut vol_series: Vec<f64> = Vec::new();
    let mut last_prc = f64::NAN;
    let mut last_vol = f64::NAN;
    let mut k = first;
    while k <= last {
        if let Some(slot) = bins.get(&k) {
            if let Some(v) = slot.0 {
                last_prc = v;
            }
            if let Some(v) = slot.1 {
                last_vol = v;
            }
        }
        prc_series.push(last_prc);
        vol_series.push(last_vol);
        k += 1;
    }

    // df = df[df.last_prc != 0]（NaN != 0 为真，NaN 行保留）。
    let keep: Vec<usize> = (0..prc_series.len())
        .filter(|&i| prc_series[i] != 0.0)
        .collect();
    if keep.is_empty() {
        return None;
    }
    let prc_kept: Vec<f64> = keep.iter().map(|&i| prc_series[i]).collect();
    let vol_kept: Vec<f64> = keep.iter().map(|&i| vol_series[i]).collect();

    // df.volume = df.volume.diff()；df.volume = df.volume / df.volume.sum()
    // diff 的第一项是 NaN（前值不存在），sum 跳过 NaN。
    let mut diff: Vec<f64> = Vec::with_capacity(vol_kept.len());
    for i in 0..vol_kept.len() {
        if i == 0 {
            diff.push(f64::NAN);
        } else {
            diff.push(vol_kept[i] - vol_kept[i - 1]);
        }
    }
    let total: f64 = diff.iter().filter(|x| !x.is_nan()).sum();
    let vol_norm: Vec<f64> = diff.iter().map(|&x| x / total).collect();

    Some((prc_kept, vol_norm))
}

// ============================================================================
// pandas 统计量（与 hm064.rs 同名函数同式）
// ============================================================================

fn seq_sum(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in v {
        s += x;
    }
    s
}

#[inline]
fn zero_out_fperr(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        0.0
    } else {
        x
    }
}

/// pandas 的 notna：只有 NaN 算缺失，inf 算有效。
fn valid_of(c: &[f64]) -> Vec<f64> {
    c.iter().copied().filter(|x| !x.is_nan()).collect()
}

fn col_mean(c: &[f64]) -> f64 {
    let v = valid_of(c);
    if v.is_empty() {
        return f64::NAN;
    }
    seq_sum(&v) / v.len() as f64
}

fn col_std(c: &[f64]) -> f64 {
    let v = valid_of(c);
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let avg = seq_sum(&v) / n as f64;
    let mut sq = 0.0;
    for &x in v.iter() {
        let d = avg - x;
        sq += d * d;
    }
    (sq / (n as f64 - 1.0)).sqrt()
}

fn col_skew(c: &[f64]) -> f64 {
    let v = valid_of(c);
    let n = v.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = seq_sum(&v) / nf;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in v.iter() {
        let d = x - mu;
        m2 += d * d;
        m3 += (d * d) * d;
    }
    let m2 = zero_out_fperr(m2);
    let m3 = zero_out_fperr(m3);
    if m2 == 0.0 {
        return 0.0;
    }
    (nf * (nf - 1.0).sqrt() / (nf - 2.0)) * (m3 / m2.powf(1.5))
}

fn col_kurt(c: &[f64]) -> f64 {
    let v = valid_of(c);
    let n = v.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = seq_sum(&v) / nf;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &x in v.iter() {
        let d = x - mu;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    let m2 = zero_out_fperr(m2);
    let m4 = zero_out_fperr(m4);
    let adj = 3.0 * (nf - 1.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0));
    let numerator = zero_out_fperr(((nf * (nf + 1.0)) * (nf - 1.0)) * m4);
    let denominator = zero_out_fperr(((nf - 2.0) * (nf - 3.0)) * (m2 * m2));
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator - adj
}

fn col_max(c: &[f64]) -> f64 {
    let v = valid_of(c);
    if v.is_empty() {
        return f64::NAN;
    }
    let mut m = f64::NEG_INFINITY;
    for &x in v.iter() {
        if x > m {
            m = x;
        }
    }
    m
}

/// pandas 的 Pearson 相关（np.corrcoef 语义，用于 corrwith / Series.corr）：
/// 成对非 NaN，样本<2 或任一列方差为 0 → NaN。
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n_all = a.len().min(b.len());
    let mut n = 0usize;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..n_all {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        n += 1;
        sx += a[i];
        sy += b[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = sx / nf;
    let my = sy / nf;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..n_all {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        let dx = a[i] - mx;
        let dy = b[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// pandas `libalgos.nancorr`（DataFrame.corr 的 float64 路径）：Welford 在线算法，
/// mask 用 **isfinite**（inf 也算缺失，和统计量那边的 notna 不一样），min_periods=1。
/// 分母是 sqrt(ssqdmx * ssqdmy)，为 0 时结果是 NaN。
fn welford_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut nobs = 0.0f64;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    for i in 0..n {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let vx = x[i];
        let vy = y[i];
        nobs += 1.0;
        let dx = vx - meanx;
        let dy = vy - meany;
        meanx += dx / nobs;
        meany += dy / nobs;
        ssqdmx += (vx - meanx) * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += (vx - meanx) * dy;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// rp.trend_2d(..., axis=0) 的单列版（与 src/time_series/trend_mod.rs::calculate_trend_1d 同式）。
fn trend_1d(data: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::with_capacity(data.len());
    let mut indices: Vec<f64> = Vec::with_capacity(data.len());
    for (i, &val) in data.iter().enumerate() {
        if val.is_finite() {
            values.push(val);
            indices.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = values.iter().sum::<f64>() / nf;
    let mean_y = indices.iter().sum::<f64>() / nf;
    let mut covariance = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = indices[i] - mean_y;
        covariance += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    covariance / (var_x.sqrt() * var_y.sqrt())
}

// ============================================================================
// 一次 compute：818 个值
// ============================================================================

/// 旧 go 里 `compute(arr_a, arr_b, twins)` 的复刻。
///
/// `twins` 是 (last_prc, volume) 两列，index 与价格/成交量序列的位置一一对应。
fn compute_one(prices: &[f64], volumes: &[f64], twins: (&[f64], &[f64])) -> Vec<f64> {
    let raw = local_correlation_inner(prices, volumes, WINDOW);
    let n = raw.nrows();

    // DataFrame(raw[0], columns=ORIG_COLS).dropna()
    let orig: Vec<Vec<f64>> = (0..12)
        .map(|j| (0..n).map(|i| raw[[i, j]]).collect())
        .collect();
    let rows: Vec<usize> = (0..n)
        .filter(|&i| (0..12).all(|j| !orig[j][i].is_nan()))
        .collect();

    // 12 个派生列，顺序与旧代码 assign 的顺序一致。
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(NBASE);
    for j in 0..12 {
        cols.push(rows.iter().map(|&i| orig[j][i]).collect());
    }
    let dv = |a: usize, b: usize| -> Vec<f64> {
        rows.iter().map(|&i| orig[a][i] / orig[b][i]).collect()
    };
    cols.push(dv(4, 0)); // r_bk_rev = bk_btw_r / bk_max_r
    cols.push(dv(2, 0)); // r_bk_sl  = bk_min_r / bk_max_r
    cols.push(dv(10, 6)); // r_fw_rev = fw_btw_r / fw_max_r
    cols.push(dv(8, 6)); // r_fw_sl  = fw_min_r / fw_max_r
    cols.push(dv(5, 1)); // v_bk_rev = bk_btw_v / bk_max_v
    cols.push(dv(3, 1)); // v_bk_sl  = bk_min_v / bk_max_v
    cols.push(dv(11, 7)); // v_fw_rev = fw_btw_v / fw_max_v
    cols.push(dv(9, 7)); // v_fw_sl  = fw_min_v / fw_max_v
    let dsub = |a: usize, b: usize| -> Vec<f64> {
        rows.iter().map(|&i| (orig[a][i] - orig[b][i]).abs()).collect()
    };
    cols.push(dsub(2, 0)); // r_bk_sld = |bk_min_r - bk_max_r|
    cols.push(dsub(8, 6)); // r_fw_sld = |fw_min_r - fw_max_r|
    cols.push(dsub(3, 1)); // v_bk_sld = |bk_min_v - bk_max_v|
    cols.push(dsub(9, 7)); // v_fw_sld = |fw_min_v - fw_max_v|
    debug_assert_eq!(cols.len(), NBASE);

    // stat_df：7 个统计 × 24 列；取值顺序是 [for s in STATS for c in BASE_COLS]。
    let mut out: Vec<f64> = Vec::with_capacity(NHALF);
    let means: Vec<f64> = cols.iter().map(|c| col_mean(c)).collect();
    let stds: Vec<f64> = cols.iter().map(|c| col_std(c)).collect();
    let skews: Vec<f64> = cols.iter().map(|c| col_skew(c)).collect();
    let kurts: Vec<f64> = cols.iter().map(|c| col_kurt(c)).collect();
    let maxs: Vec<f64> = cols.iter().map(|c| col_max(c)).collect();
    // ac = df.corrwith(df.shift(1))：逐列把 (x[i-1], x[i]) 配成对再算 Pearson，
    // i=0 那对因为 shift 出来是 NaN 自动落空。
    let acs: Vec<f64> = cols
        .iter()
        .map(|c| {
            let shifted: Vec<f64> = std::iter::once(f64::NAN)
                .chain(c.iter().copied().take(c.len().saturating_sub(1)))
                .collect();
            pearson(c, &shifted)
        })
        .collect();
    let trends: Vec<f64> = cols.iter().map(|c| trend_1d(c)).collect();
    for group in [&means, &stds, &skews, &kurts, &maxs, &acs, &trends] {
        out.extend_from_slice(group);
    }
    debug_assert_eq!(out.len(), 7 * NBASE);

    // df_full = concat([df, twins_sub]).dropna()：
    // twins 里 index 落在 df.index 的那些行并进来后，要求 26 列全都非 NaN。
    let keep: Vec<usize> = (0..rows.len())
        .filter(|&k| {
            (0..NBASE).all(|j| !cols[j][k].is_nan())
                && !twins.0[rows[k]].is_nan()
                && !twins.1[rows[k]].is_nan()
        })
        .collect();

    let mut all: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| keep.iter().map(|&k| c[k]).collect())
        .collect();
    all.push(keep.iter().map(|&k| twins.0[rows[k]]).collect());
    all.push(keep.iter().map(|&k| twins.1[rows[k]]).collect());
    debug_assert_eq!(all.len(), NALL);

    // df_full.corr() 的严格上三角，再拼一份取绝对值。
    let mut corr: Vec<f64> = Vec::with_capacity(NALL * (NALL - 1) / 2);
    for i in 0..NALL {
        for j in (i + 1)..NALL {
            corr.push(welford_corr(&all[i], &all[j]));
        }
    }
    out.extend_from_slice(&corr);
    out.extend(corr.iter().map(|x| x.abs()));
    debug_assert_eq!(out.len(), NHALF);
    out
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let (prc, vol) = load_series(code, date)?;
    let twins = (prc.as_slice(), vol.as_slice());
    // 旧 go：先 compute(volume, volume)，再 compute(last_prc, volume)。
    let vv = compute_one(&vol, &vol, twins);
    let pv = compute_one(&prc, &vol, twins);
    let mut full = vv;
    full.extend(pv);
    if full.len() != FULL {
        return None;
    }
    // 流水线对外是 f32（TaskResult.facs 是 f32），只在最后一步降精度。
    Some(IDX.iter().map(|&i| full[i] as f32).collect())
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
