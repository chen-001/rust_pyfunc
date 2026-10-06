//! hm081「被动者背后」—— 旧框架模块级 go(date, code) 的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm81_被动者背后/hm81_okay.py
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 go 做的事（逐步对照）
//! ════════════════════════════════════════════════════════════════════
//! 1) trade_data  = rp.adjust_afternoon(rp.read_trade(code, date))
//!    market_data = rp.adjust_afternoon(rp.read_market(code, date))
//!                   .set_index('exchtime').resample('15s').last().reset_index()
//!    · read_trade 丢 flag==32 的撤单行（数值比较），exchtime 换成
//!      「1970-01-01 08:00:00 + epoch_us/1e6 秒」的 Timestamp
//!    · read_market 丢 ask_prc1==0 或 bid_prc1==0 的快照，同样换 Timestamp
//!    · adjust_afternoon：上午 [09:30:00, 11:30:00]、下午 [13:00:00, 14:57:00]
//!      两段**纳秒闭区间**，下午段整体前移 90 分钟，concat（上午全部在前、下午全部在后）
//!    · resample('15s').last()：bin 从当日 00:00:00 起按 15 秒对齐、label 取左边界，
//!      `.last()` 是**逐列**取该 bin 内最后一个非空值
//! 2) trade_times/flags/bid_order/ask_order/volume 与 market_times（bin 标签的纳秒值）
//!    交给 rp.calculate_passive_order_features(compute_direction_ratio=True,
//!    compute_flag_ratio=False)，得到 (n_intervals × 84) 的特征矩阵 + 列名。
//!    n_intervals = bin 数 - 1；区间是左开右闭 (market_times[i], market_times[i+1]]，
//!    指针 left/right 跨区间不重置（见 passive_order_features.rs 的 inner）。
//! 3) 在 features 上补两列残差：
//!      passive_bid_order_resi = rp.ols_residuals(nums, df['passive_bid_order_mean'])
//!      passive_ask_order_resi = 同上（ask）
//!    nums = df.index.to_numpy(float).reshape(-1,1)，即自变量就是行号 0..n-1。
//! 4) 丢掉 12 列（非 dedup 的 mean/std/skew 九列 + 三列 volume kurtosis），剩 74 列。
//! 5) market_data.volume = .diff()、total_ask_vol_diff / total_bid_vol_diff = .diff()，
//!    取 6 列 iloc[1:,:] 作为 append_for_corr，连同 df 一起交给
//!    rp.get_features_factors(df, with_threshold_counts=False,
//!                            with_lyapunov_exponent=False, append_for_corr=append)
//!    —— 共 4566 项。
//! 6) 按 names_save 去 _fold 后的 79 个名字取值返回（顺序即 names()）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、两处必须自己写、不能套现成入口的地方
//! ════════════════════════════════════════════════════════════════════
//!
//! 【1】features 的 append_for_corr 是**按 index 对齐**的，不是按位置对齐。
//!    df（947 行）的 index 是 RangeIndex 0..946；append 来自
//!    `market_data[cols].iloc[1:, :]`，**index 是 1..947**。
//!    `pd.concat([df, append], axis=1)` 按标签对齐 → 结果 948 行：
//!      · 标签 0：df 有效、append 为 NaN
//!      · 标签 1..946：df 第 j 行与 market_data 第 j 行配对
//!      · 标签 947：df 为 NaN、append 有效
//!    两个来源长度都是 947，长度检查（ap_rows == n_rows）会通过，但**位置对齐会整体错一行**。
//!    所以这里不走 features_f64 的 append 开关（那条路是给 index 相同的调用方用的），
//!    而是自己拼 80 列（74 + 6）的**并集矩阵**（948 行，各自的洞填 NaN），
//!    自己算相关块，再把它插回 no-append 结果里。
//!
//! 【2】相关块必须照 fast_correlation_v2 的实现，不能照 features_f64 的 corr_pair。
//!    Python 侧 get_features_factors 的相关块走 rp.fast_correlation_matrix_v2_df
//!    → src/statistics/fast_correlation_v2.rs。它与 features_f64::corr_pair 有两处不同：
//!      · 列均值用 **Kahan 补偿求和**（corr_pair 是普通顺序累加）
//!      · 方差和 < f64::EPSILON 时给 **NaN**（corr_pair 给 0.0）
//!    并集矩阵里每一列都带 NaN（df 列在标签 947、append 列在标签 0），
//!    所以所有配对都走「共同有效观测」那条分支，均值取自共同观测。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、时间与字符解析
//! ════════════════════════════════════════════════════════════════════
//! · 时间一律用精确纳秒：`exchtime_us * 1000 + 8h`。pandas 走
//!   cast_from_unit_vectorized 会有 ±120ns 噪声，但所有比较都发生在
//!   15 秒 bin 边界（15e9 的整数倍）上，而边界处的 exchtime 是整秒
//!   （us 是 1e6 的整数倍），那时 `us/1e6` 在 double 里精确、frac 恰好为 0，
//!   噪声为零。所以精确纳秒与 pandas 的比较结果逐行一致。
//! · 撤单按**数值**判（CSV 里写的是 "32.0"，按字节比 "32" 会漏）。
//! · 浮点按 pandas 的 precise_xstrtod 复刻（parse_f64_pandas，照抄 hm090.rs）。
//!   这里只有 ask_prc1/bid_prc1 的「精确等于 0」判定吃这个，其余量差 1 ulp 无影响。
//! · 成交明细的 price/turnover 本系列用不到，不解析。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、重名
//! ════════════════════════════════════════════════════════════════════
//! 79 个输出名里有 2 个重复（都是自己的 _fold 版本展开后与本体同名）：
//!   passive_all_order_autocorr_mean、passive_bid_order_autocorr_period_diff
//! 按规范第二次出现加 `_dup2` 后缀，值不变、列数不变。
//!
//! 验证：四组 (日期, 股票) 的 79 个值与旧 go 全部一致（rtol 1e-4）。

use crate::data_paths;
use crate::features_f64::{self, FeatureOpts};
use crate::passive_order_features;
use crate::statistics::solve_linear_system3;
use ndarray::{Array1, Array2};
use std::collections::HashSet;

/// 一天的纳秒数。
const DAY_NS: i64 = 86_400_000_000_000;
/// 纳秒 / 微秒。
const NS_PER_US: i64 = 1_000;
/// 旧框架的 _TRADE_EPOCH_OFFSET = 1970-01-01 08:00:00。
const EPOCH_OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
/// 下午段整体前移的纳秒数。
const AFTERNOON_SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;

const MORNING_START_NS: i64 = (9 * 3600 + 30 * 60) * 1_000_000_000;
const MORNING_END_NS: i64 = (11 * 3600 + 30 * 60) * 1_000_000_000;
const AFTERNOON_START_NS: i64 = 13 * 3600 * 1_000_000_000;
const AFTERNOON_END_NS: i64 = (14 * 3600 + 57 * 60) * 1_000_000_000;

/// resample('15s') 的 bin 宽度。
const BIN_NS: i64 = 15_000_000_000;

/// 84 个被动订单特征的列数（compute_direction_ratio=true、compute_flag_ratio=false）。
const N_FEAT: usize = 84;
/// drop 之后进 get_features_factors 的列数。
const N_DF_COL: usize = 74;
/// append_for_corr 的 6 列。
const APPEND_COLS: [&str; 6] = [
    "last_prc",
    "volume",
    "total_ask_vol",
    "total_bid_vol",
    "total_ask_vol_diff",
    "total_bid_vol_diff",
];

/// 旧 go() 的返回顺序 = names_save 去掉 _fold 之后的 79 项（含 2 个重复名）。
const LOOKUP: [&str; 79] = [
    "ask_order_volume_autocorr_corr_passive_ask_order_dedup_std",
    "ask_order_volume_lz_complexity_autocorr1",
    "ask_order_volume_lz_complexity_corr_last_prc",
    "ask_order_volume_lz_complexity_corr_passive_all_order_dedup_std",
    "ask_order_volume_lz_complexity_corr_passive_ask_order_resi",
    "ask_order_volume_lz_complexity_corr_volume",
    "ask_order_volume_std_corr_total_ask_vol_diff",
    "ask_order_volume_std_period_ratio",
    "bid_order_volume_autocorr_corr_ask_order_volume_skew",
    "bid_order_volume_autocorr_p75",
    "bid_order_volume_lz_complexity_corr_ask_order_volume_skew",
    "bid_order_volume_lz_complexity_corr_passive_ask_order_dedup_kurtosis",
    "bid_order_volume_mean_corr_direction_ratio_bid_lz_complexity",
    "bid_order_volume_skew_corr_passive_all_order_dedup_skew",
    "bid_order_volume_skew_corr_passive_bid_order_dedup_skew",
    "bid_order_volume_skew_corr_volume",
    "bid_order_volume_std_corr_direction_ratio_ask_lz_complexity",
    "direction_ratio_all_autocorr_iqr",
    "direction_ratio_all_autocorr_p95",
    "direction_ratio_all_autocorr_period_diff",
    "direction_ratio_all_lz_complexity_corr_volume",
    "direction_ratio_all_skew_cv",
    "direction_ratio_all_trend_p5",
    "direction_ratio_ask_autocorr_p95",
    "direction_ratio_ask_mean_median",
    "direction_ratio_ask_mean_p75",
    "direction_ratio_ask_std_corr_last_prc",
    "direction_ratio_ask_std_corr_volume",
    "direction_ratio_bid_autocorr_autocorr1_abs",
    "direction_ratio_bid_mean_entropy_1d",
    "direction_ratio_bid_mean_p25",
    "direction_ratio_bid_std_corr_volume",
    "direction_ratio_bid_trend_corr_direction_ratio_ask_trend",
    "last_prc_corr_volume",
    "order_volume_lz_complexity_autocorr1_abs",
    "order_volume_lz_complexity_corr_direction_ratio_bid_std",
    "order_volume_lz_complexity_corr_volume",
    "order_volume_mean_corr_passive_bid_order_dedup_std",
    "order_volume_skew_autocorr1_abs",
    "order_volume_skew_corr_passive_all_order_dedup_kurtosis",
    "passive_all_order_autocorr_corr_bid_order_volume_lz_complexity",
    "passive_all_order_autocorr_mean",
    "passive_all_order_autocorr_mean",
    "passive_all_order_autocorr_period_diff",
    "passive_all_order_autocorr_std",
    "passive_all_order_dedup_autocorr_mean",
    "passive_all_order_dedup_autocorr_p5",
    "passive_all_order_dedup_mean_corr_last_prc",
    "passive_all_order_dedup_skew_corr_passive_all_order_dedup_kurtosis",
    "passive_all_order_dedup_skew_corr_passive_bid_order_dedup_std",
    "passive_all_order_kurtosis_corr_passive_ask_order_autocorr",
    "passive_all_order_lz_complexity_corr_bid_order_volume_lz_complexity",
    "passive_all_order_trend_median",
    "passive_all_order_trend_period_diff",
    "passive_ask_order_autocorr_autocorr1",
    "passive_ask_order_autocorr_autocorr1_abs",
    "passive_ask_order_autocorr_p5",
    "passive_ask_order_dedup_autocorr_autocorr1",
    "passive_ask_order_dedup_autocorr_autocorr1_abs",
    "passive_ask_order_dedup_lz_complexity_autocorr1",
    "passive_ask_order_dedup_lz_complexity_corr_direction_ratio_bid_mean",
    "passive_ask_order_dedup_lz_complexity_corr_volume",
    "passive_ask_order_dedup_trend_mean",
    "passive_bid_order_autocorr_corr_direction_ratio_ask_mean",
    "passive_bid_order_autocorr_corr_passive_ask_order_autocorr",
    "passive_bid_order_autocorr_mean",
    "passive_bid_order_autocorr_period_diff",
    "passive_bid_order_autocorr_period_diff",
    "passive_bid_order_dedup_autocorr_corr_passive_ask_order_dedup_std",
    "passive_bid_order_dedup_autocorr_std",
    "passive_bid_order_dedup_autocorr_trend",
    "passive_bid_order_dedup_mean_corr_passive_ask_order_dedup_std",
    "passive_bid_order_dedup_mean_corr_passive_ask_order_resi",
    "passive_bid_order_dedup_skew_corr_direction_ratio_all_mean",
    "passive_bid_order_dedup_skew_corr_direction_ratio_ask_mean",
    "passive_bid_order_dedup_skew_corr_direction_ratio_bid_mean",
    "passive_bid_order_dedup_skew_corr_total_bid_vol_diff",
    "passive_bid_order_dedup_trend_std",
    "passive_bid_order_lz_complexity_corr_passive_ask_order_dedup_autocorr",
];

pub fn names() -> Vec<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut out: Vec<String> = Vec::with_capacity(LOOKUP.len());
    for &n in LOOKUP.iter() {
        if seen.insert(n) {
            out.push(n.to_string());
        } else {
            // 同名第二次出现：值不变，只加后缀让列名唯一。
            out.push(format!("{}_dup2", n));
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════════
// 原始数据读取（自己按 f64 解析，不走 fast_csv_reader 的 f32）
// ════════════════════════════════════════════════════════════════════════

/// 一个交易日的逐笔（已按旧框架的 read_trade + adjust_afternoon 处理）。
struct Trades {
    /// 纳秒（含 +8h 偏移；下午段已前移 90 分钟）。
    ns: Vec<i64>,
    flags: Vec<i32>,
    bid: Vec<i64>,
    ask: Vec<i64>,
    vol: Vec<i64>,
}

/// 一个交易日的盘口（已按旧框架的 read_market + adjust_afternoon 处理，只留 5 列）。
struct Market {
    ns: Vec<i64>,
    last_prc: Vec<f64>,
    volume: Vec<f64>,
    total_ask_vol: Vec<f64>,
    total_bid_vol: Vec<f64>,
}

/// 读一个交易日的成交明细。
fn read_trade_day(code: &str, date: i64) -> Option<Trades> {
    const COL_EXCHTIME: usize = 4;
    const COL_VOLUME: usize = 8;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;
    const NFIELDS: usize = 15;

    let data = read_level2_file(code, date, "transaction", "transaction")?;

    let mut morning: Vec<[i64; 5]> = Vec::new();
    let mut afternoon: Vec<[i64; 5]> = Vec::new();
    let mut fields: [&[u8]; NFIELDS] = [&[][..]; NFIELDS];
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        if split_fields(line, &mut fields) < NFIELDS {
            continue;
        }
        let exchtime_us = us_prefix(fields[COL_EXCHTIME]);
        if exchtime_us == 0 {
            continue;
        }
        let raw_flag = num(fields[COL_FLAG]);
        if raw_flag == 32.0 {
            continue; // read_trade 默认丢撤单行（数值比较，不是字节比较）
        }
        let ns = exchtime_us * NS_PER_US + EPOCH_OFFSET_NS;
        let tod = ns.rem_euclid(DAY_NS);
        let row = [
            ns,
            raw_flag as i64,
            int_prefix(fields[COL_BID_ORDER]),
            int_prefix(fields[COL_ASK_ORDER]),
            int_prefix(fields[COL_VOLUME]),
        ];
        if (MORNING_START_NS..=MORNING_END_NS).contains(&tod) {
            morning.push(row);
        } else if (AFTERNOON_START_NS..=AFTERNOON_END_NS).contains(&tod) {
            let mut r = row;
            r[0] = ns - AFTERNOON_SHIFT_NS;
            afternoon.push(r);
        }
    }

    // concat([上午, 下午])：行序 = 上午全部（文件序）在前、下午全部（文件序）在后。
    let mut rows = morning;
    rows.extend(afternoon);

    let n = rows.len();
    let mut t = Trades {
        ns: Vec::with_capacity(n),
        flags: Vec::with_capacity(n),
        bid: Vec::with_capacity(n),
        ask: Vec::with_capacity(n),
        vol: Vec::with_capacity(n),
    };
    for r in rows.iter() {
        t.ns.push(r[0]);
        t.flags.push(r[1] as i32);
        t.bid.push(r[2]);
        t.ask.push(r[3]);
        t.vol.push(r[4]);
    }
    Some(t)
}

/// 读一个交易日的盘口（只解析用得到的 7 列）。
fn read_market_day(code: &str, date: i64) -> Option<Market> {
    const COL_EXCHTIME: usize = 4;
    const COL_LAST_PRC: usize = 6;
    const COL_VOLUME: usize = 14;
    const COL_TOTAL_ASK_VOL: usize = 19;
    const COL_TOTAL_BID_VOL: usize = 20;
    const COL_ASK_PRC1: usize = 21;
    const COL_BID_PRC1: usize = 23;
    const NFIELDS: usize = 61;

    let data = read_level2_file(code, date, "market_data", "market_data")?;

    let mut morning: Vec<(i64, [f64; 4])> = Vec::new();
    let mut afternoon: Vec<(i64, [f64; 4])> = Vec::new();
    let mut fields: [&[u8]; NFIELDS] = [&[][..]; NFIELDS];
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        if split_fields(line, &mut fields) < NFIELDS {
            continue;
        }
        // read_market 默认丢掉涨跌停/无效快照（ask_prc1 或 bid_prc1 精确等于 0）。
        if num(fields[COL_ASK_PRC1]) == 0.0 || num(fields[COL_BID_PRC1]) == 0.0 {
            continue;
        }
        let exchtime_us = us_prefix(fields[COL_EXCHTIME]);
        let ns = exchtime_us * NS_PER_US + EPOCH_OFFSET_NS;
        let tod = ns.rem_euclid(DAY_NS);
        // ns 单独存成 i64：纳秒值约 1.7e18，塞进 f64 会丢精度。
        let vals = [
            num(fields[COL_LAST_PRC]),
            num(fields[COL_VOLUME]),
            num(fields[COL_TOTAL_ASK_VOL]),
            num(fields[COL_TOTAL_BID_VOL]),
        ];
        if (MORNING_START_NS..=MORNING_END_NS).contains(&tod) {
            morning.push((ns, vals));
        } else if (AFTERNOON_START_NS..=AFTERNOON_END_NS).contains(&tod) {
            afternoon.push((ns - AFTERNOON_SHIFT_NS, vals));
        }
    }

    let mut rows = morning;
    rows.extend(afternoon);

    let n = rows.len();
    let mut m = Market {
        ns: Vec::with_capacity(n),
        last_prc: Vec::with_capacity(n),
        volume: Vec::with_capacity(n),
        total_ask_vol: Vec::with_capacity(n),
        total_bid_vol: Vec::with_capacity(n),
    };
    for (ns, vals) in rows.iter() {
        m.ns.push(*ns);
        m.last_prc.push(vals[0]);
        m.volume.push(vals[1]);
        m.total_ask_vol.push(vals[2]);
        m.total_bid_vol.push(vals[3]);
    }
    Some(m)
}

/// 在 level2 数据根目录下找 `<root>/<date>/<sub>/<code>_<date>_<suffix>.csv`。
fn read_level2_file(code: &str, date: i64, sub: &str, suffix: &str) -> Option<Vec<u8>> {
    let filename = format!("{}_{}_{}.csv", code, date, suffix);
    for root in data_paths::level2_roots() {
        let path = root.join(date.to_string()).join(sub).join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            return Some(bytes);
        }
    }
    None
}

/// 按逗号切字段，返回切出的字段数（超出容量的部分丢掉）。
fn split_fields<'a>(line: &'a [u8], out: &mut [&'a [u8]]) -> usize {
    let cap = out.len();
    let mut count = 0usize;
    let mut start = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if count < cap {
                out[count] = &line[start..i];
            }
            count += 1;
            start = i + 1;
        }
    }
    if count < cap {
        out[count] = &line[start..];
    }
    count + 1
}

/// 去掉字段两端的空白（含行尾的 `\r`）。
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

/// 取字段的整数前缀。`"32.0"` 给 32。
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

/// exchtime 列（形如 `1740964500130000.0`）：整数前缀就是微秒数。
fn us_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    if !b.contains(&b'e') && !b.contains(&b'E') {
        return int_prefix(b);
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.round() as i64)
        .unwrap_or(0)
}

/// 小数转 f64（空字段给 NaN，与 pandas 一致）。
fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
}

/// 10 的幂表（照抄 hm090.rs / hm033.rs）。
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

/// pandas C 解析器 precise_xstrtod 的逐行复刻（照抄 hm090.rs）。
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

// ════════════════════════════════════════════════════════════════════════
// resample('15s').last()
// ════════════════════════════════════════════════════════════════════════

/// 复刻 `set_index('exchtime').resample('15s').last().reset_index()`。
///
/// 返回 (bin 标签纳秒, 每列按 bin 的「最后一个非空值」)。bin 范围从
/// floor(首个快照 / 15s) 到 floor(末个快照 / 15s)，中间空 bin 全 NaN。
fn resample_15s(md: &Market) -> Option<(Vec<i64>, [Vec<f64>; 4])> {
    let n = md.ns.len();
    if n == 0 {
        return None;
    }
    let first_bin = md.ns[0].div_euclid(BIN_NS) * BIN_NS;
    let last_bin = md.ns[n - 1].div_euclid(BIN_NS) * BIN_NS;
    let m = ((last_bin - first_bin) / BIN_NS + 1) as usize;

    let mut bins = Vec::with_capacity(m);
    for k in 0..m {
        bins.push(first_bin + k as i64 * BIN_NS);
    }
    let mut cols: [Vec<f64>; 4] = [
        vec![f64::NAN; m],
        vec![f64::NAN; m],
        vec![f64::NAN; m],
        vec![f64::NAN; m],
    ];
    for i in 0..n {
        let k = ((md.ns[i] - first_bin) / BIN_NS) as usize;
        if k >= m {
            continue;
        }
        // .last()：逐个覆盖，最后留下的就是该 bin 内最后一个非空值。
        if !md.last_prc[i].is_nan() {
            cols[0][k] = md.last_prc[i];
        }
        if !md.volume[i].is_nan() {
            cols[1][k] = md.volume[i];
        }
        if !md.total_ask_vol[i].is_nan() {
            cols[2][k] = md.total_ask_vol[i];
        }
        if !md.total_bid_vol[i].is_nan() {
            cols[3][k] = md.total_bid_vol[i];
        }
    }
    Some((bins, cols))
}

// ════════════════════════════════════════════════════════════════════════
// rp.ols_residuals(行号, y)：带截距的一元回归残差
// ════════════════════════════════════════════════════════════════════════

/// 与 src/statistics/mod.rs 的 ols_residuals 同式：样本过滤只按 is_nan，
/// X'X / X'y 用 ndarray 的 .dot()，求解用 crate::statistics::solve_linear_system3。
/// 自变量就是行号 0..n-1（旧代码传的是 df.index）。
fn ols_residual_index(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![f64::NAN; n];
    let idx: Vec<usize> = (0..n).filter(|&i| !y[i].is_nan()).collect();
    if idx.is_empty() {
        return out;
    }
    let m = idx.len();
    let mut design = Array2::<f64>::ones((m, 2));
    let mut yv = Array1::<f64>::zeros(m);
    for (row, &i) in idx.iter().enumerate() {
        design[[row, 1]] = i as f64;
        yv[row] = y[i];
    }
    let xt_x = design.t().dot(&design);
    let xt_y = design.t().dot(&yv);
    let coeff = solve_linear_system3(&xt_x.view(), &xt_y.view());
    let y_pred = design.dot(&coeff);
    for (row, &i) in idx.iter().enumerate() {
        out[i] = y[i] - y_pred[row];
    }
    out
}

// ════════════════════════════════════════════════════════════════════════
// 相关块：照抄 src/statistics/fast_correlation_v2.rs
// ════════════════════════════════════════════════════════════════════════

/// Kahan 补偿求和（fast_correlation_v2 算列均值用的那个）。
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

/// `rp.fast_correlation_matrix_v2_df(df, max_workers=1)` 的一对列版本。
///
/// 本系列走的是「共同有效观测」那条分支：并集矩阵里每一列都带 NaN，
/// 所以 `valid_data.len() != col.len()`，永远进不了 clean 分支。
/// 仍保留 clean 分支，免得以后换用法时静默走错路。
fn corr_pair_fast(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let count_a = a[..n].iter().filter(|v| !v.is_nan()).count();
    let count_b = b[..n].iter().filter(|v| !v.is_nan()).count();
    // min_periods 默认 1
    if count_a < 1 || count_b < 1 {
        return f64::NAN;
    }

    if count_a == n && count_b == n {
        // clean 分支：均值取整列，8 路展开的顺序累加
        if n < 2 {
            return f64::NAN;
        }
        let mean_i = kahan_sum(&a[..n]) / n as f64;
        let mean_j = kahan_sum(&b[..n]) / n as f64;
        let (mut cov, mut vi, mut vj) = (0.0f64, 0.0f64, 0.0f64);
        let chunks = n / 8;
        let rem = n % 8;
        for c in 0..chunks {
            let base = c * 8;
            for k in 0..8 {
                let idx = base + k;
                let di = a[idx] - mean_i;
                let dj = b[idx] - mean_j;
                cov += di * dj;
                vi += di * di;
                vj += dj * dj;
            }
        }
        for k in 0..rem {
            let idx = chunks * 8 + k;
            let di = a[idx] - mean_i;
            let dj = b[idx] - mean_j;
            cov += di * dj;
            vi += di * di;
            vj += dj * dj;
        }
        if vi.abs() < f64::EPSILON || vj.abs() < f64::EPSILON {
            return f64::NAN;
        }
        return cov / (vi.sqrt() * vj.sqrt());
    }

    // 共同有效观测分支
    let mut px: Vec<f64> = Vec::with_capacity(n);
    let mut py: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !a[i].is_nan() && !b[i].is_nan() {
            px.push(a[i]);
            py.push(b[i]);
        }
    }
    let common = px.len();
    if common < 1 {
        return f64::NAN;
    }
    let mean_i = kahan_sum(&px) / common as f64;
    let mean_j = kahan_sum(&py) / common as f64;
    if common < 2 {
        return f64::NAN;
    }
    // fast_correlation_v2 的 compute_correlation_vectorized：
    // 样本 > 1024 时按 1024 分块累加（max_workers=1 时按块序合并），否则顺序累加。
    let (mut cov, mut vi, mut vj) = (0.0f64, 0.0f64, 0.0f64);
    if common > 1024 {
        let mut idx = 0usize;
        while idx < common {
            let end = (idx + 1024).min(common);
            let (mut lc, mut li, mut lj) = (0.0f64, 0.0f64, 0.0f64);
            for k in idx..end {
                let di = px[k] - mean_i;
                let dj = py[k] - mean_j;
                lc += di * dj;
                li += di * di;
                lj += dj * dj;
            }
            cov += lc;
            vi += li;
            vj += lj;
            idx = end;
        }
    } else {
        for k in 0..common {
            let di = px[k] - mean_i;
            let dj = py[k] - mean_j;
            cov += di * dj;
            vi += di * di;
            vj += dj * dj;
        }
    }
    if vi.abs() < f64::EPSILON || vj.abs() < f64::EPSILON {
        return f64::NAN;
    }
    cov / (vi.sqrt() * vj.sqrt())
}

// ════════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════════

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trade_day(code, date)?;
    let market = read_market_day(code, date)?;
    let (bins, mcols) = resample_15s(&market)?;
    if bins.len() < 2 {
        return None;
    }
    let n_bin = bins.len();
    let n = n_bin - 1; // 特征行数 = bin 数 - 1

    let (flat, cols84) = passive_order_features::calculate_passive_order_features_inner(
        &trades.ns,
        &trades.flags,
        &trades.bid,
        &trades.ask,
        &trades.vol,
        &bins,
        true,
        false,
    );
    if cols84.len() != N_FEAT || flat.len() != n * N_FEAT {
        return None;
    }

    // ── 两列残差（在 drop 之前算，自变量是行号） ──
    let pos_of = |name: &str| cols84.iter().position(|c| c == name);
    let i_bid_mean = pos_of("passive_bid_order_mean")?;
    let i_ask_mean = pos_of("passive_ask_order_mean")?;
    let y_bid: Vec<f64> = (0..n).map(|i| flat[i * N_FEAT + i_bid_mean]).collect();
    let y_ask: Vec<f64> = (0..n).map(|i| flat[i * N_FEAT + i_ask_mean]).collect();
    let resi_bid = ols_residual_index(&y_bid);
    let resi_ask = ols_residual_index(&y_ask);

    // ── drop 12 列 + 末尾追加两列残差，得到 74 列 ──
    let drop: HashSet<&str> = [
        "passive_all_order_mean",
        "passive_bid_order_mean",
        "passive_ask_order_mean",
        "passive_all_order_std",
        "passive_bid_order_std",
        "passive_ask_order_std",
        "passive_all_order_skew",
        "passive_bid_order_skew",
        "passive_ask_order_skew",
        "order_volume_kurtosis",
        "bid_order_volume_kurtosis",
        "ask_order_volume_kurtosis",
    ]
    .into_iter()
    .collect();

    let mut df_cols: Vec<String> = Vec::with_capacity(N_DF_COL);
    let mut df_data: Vec<Vec<f64>> = Vec::with_capacity(N_DF_COL);
    for (j, cname) in cols84.iter().enumerate() {
        if drop.contains(cname.as_str()) {
            continue;
        }
        df_cols.push(cname.clone());
        df_data.push((0..n).map(|i| flat[i * N_FEAT + j]).collect());
    }
    df_cols.push("passive_bid_order_resi".to_string());
    df_data.push(resi_bid);
    df_cols.push("passive_ask_order_resi".to_string());
    df_data.push(resi_ask);
    if df_cols.len() != N_DF_COL {
        return None;
    }

    let mut dmat = Array2::<f64>::zeros((n, N_DF_COL));
    for (j, col) in df_data.iter().enumerate() {
        for i in 0..n {
            dmat[[i, j]] = col[i];
        }
    }

    // ── 特征（统计段 + 复杂度段），相关块自己算 ──
    let opts = FeatureOpts {
        with_threshold_counts: false,
        with_curvature: false,
        with_max_min: false,
        with_lyapunov_exponent: false,
    };
    let (v0, n0) = features_f64::get_features_factors_rust_full_ex(
        &dmat.view(),
        &df_cols,
        opts,
        None,
    );
    if n0.len() != v0.len() || n0.is_empty() {
        return None;
    }
    let corr_start = n0.iter().position(|s| s.contains("_corr_"))?;
    let corr_len = N_DF_COL * (N_DF_COL - 1) / 2;
    if n0.len() < corr_start + corr_len {
        return None;
    }
    let suffix_start = corr_start + corr_len;

    // ── 并集矩阵：74 个 df 列 + 6 个 append 列，按 index 标签对齐（948 行） ──
    // df 列在标签 0..n-1 有效、标签 n 为 NaN；append 列在标签 1..n 有效、标签 0 为 NaN。
    let m_rows = n + 1;
    let mut union: Vec<Vec<f64>> = Vec::with_capacity(N_DF_COL + APPEND_COLS.len());
    for (j, _) in df_cols.iter().enumerate() {
        let mut col = vec![f64::NAN; m_rows];
        for i in 0..n {
            col[i] = dmat[[i, j]];
        }
        union.push(col);
    }
    // last_prc / total_ask_vol / total_bid_vol 直接取 bin 值；
    // volume / total_ask_vol_diff / total_bid_vol_diff 是先 resample 再 diff。
    let mut last_prc = vec![f64::NAN; m_rows];
    let mut vol_diff = vec![f64::NAN; m_rows];
    let mut ask_diff = vec![f64::NAN; m_rows];
    let mut bid_diff = vec![f64::NAN; m_rows];
    for k in 1..m_rows {
        last_prc[k] = mcols[0][k];
        vol_diff[k] = mcols[1][k] - mcols[1][k - 1];
        ask_diff[k] = mcols[2][k] - mcols[2][k - 1];
        bid_diff[k] = mcols[3][k] - mcols[3][k - 1];
    }
    let mut total_ask = vec![f64::NAN; m_rows];
    let mut total_bid = vec![f64::NAN; m_rows];
    for k in 1..m_rows {
        total_ask[k] = mcols[2][k];
        total_bid[k] = mcols[3][k];
    }
    union.push(last_prc);
    union.push(vol_diff);
    union.push(total_ask);
    union.push(total_bid);
    union.push(ask_diff);
    union.push(bid_diff);

    // ── 拼最终结果：前缀 + 相关块 + 后缀 ──
    let n_union = union.len();
    let mut all_names: Vec<String> = Vec::with_capacity(n0.len() + n_union * n_union / 2);
    let mut all_vals: Vec<f64> = Vec::with_capacity(n0.len() + n_union * n_union / 2);
    all_names.extend_from_slice(&n0[..corr_start]);
    all_vals.extend_from_slice(&v0[..corr_start]);
    let mut corr_names_all: Vec<String> = df_cols.clone();
    corr_names_all.extend(APPEND_COLS.iter().map(|s| s.to_string()));
    for i in 0..n_union {
        for j in (i + 1)..n_union {
            all_names.push(format!("{}_corr_{}", corr_names_all[i], corr_names_all[j]));
            all_vals.push(corr_pair_fast(&union[i], &union[j]));
        }
    }
    all_names.extend_from_slice(&n0[suffix_start..]);
    all_vals.extend_from_slice(&v0[suffix_start..]);

    if all_names.len() != all_vals.len() {
        return None;
    }
    // 名字 → 下标。同名只会出现在 _dup2 之前那一次，值相同，取谁都行。
    let mut idx_of: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::with_capacity(all_names.len());
    for (i, nm) in all_names.iter().enumerate() {
        idx_of.entry(nm.as_str()).or_insert(i);
    }

    let mut out: Vec<f32> = Vec::with_capacity(LOOKUP.len());
    for &name in LOOKUP.iter() {
        let i = *idx_of.get(name)?;
        out.push(all_vals[i] as f32);
    }
    Some(out)
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
