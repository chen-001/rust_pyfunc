//! hm090「日落」—— 旧框架 go 函数的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm90_日落/hm90_okay.py（模块级 go(date, code)）
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 go 做的事（逐步对照）
//! ════════════════════════════════════════════════════════════════════
//! 1) trade1 = adjust_afternoon(read_trade(code, last_trading_day(date)))
//!    trade2 = adjust_afternoon(read_trade(code, date))
//! 2) prepare(trade) 取 7 列 f64：
//!      [exchtime.astype(int64)//1e9, price, volume, turnover, flag, bid_order, ask_order]
//!    第一列是**整数秒**（换 Timestamp 后 astype(int64) 是纳秒，再 //1e9）。
//! 3) result1, cols1 = calculate_order_pair_metrics_more(t2, t1)          # tolerance 默认 0.001
//!    result2, cols2 = calculate_order_pair_metrics_more_v2(t2, t1, tolerance=0.00001)
//!    两份都是 (配对数, 44) 的 f64 矩阵，44 列列名 = 旧文件 go() 里的 base_cols。
//! 4) vals1, _ = get_features_factors(DataFrame(result1, columns=cols1))   # Python 默认参数
//!    vals2, _ = get_features_factors(DataFrame(result2, columns=cols2))
//!    每份 1914 个值（含 lyapunov 组，44 列 × 43.5 + 946 个相关系数）。
//! 5) res = vals1 + vals2（3828 个），再按 names_save_unfold 取 67 个返回。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、和库里现成流水线的关系（重点，别直接套 pipeline_order_pair_hm90）
//! ════════════════════════════════════════════════════════════════════
//! factor_pipeline.rs 里的 pipeline_order_pair_hm90 看起来就是同一件事，实测四处对不上，
//! 直接拿来当 hm090 会得到完全不同的值：
//!
//! · **撤单行**：流水线用 read_trade_fast_inner(code, date, false, true, ..)，
//!   with_retreat=false 时库里那条过滤是「按字节比 b"32"」，而 2025 年前后的成交明细里
//!   撤单标记写的是 "32.0"，过滤不生效；旧 read_trade 是 `df[df.flag != 32]` 的数值比较，
//!   两边行数差一大截（000001@20250303：旧 56193 行，流水线 75205 行）。
//!   本文件改成 with_retreat=true 读全部，再自己按 flag==32 丢掉。
//!
//! · **时间精度**：流水线走 parse_trades_from_fast，时间取的是 TradeRecord.time_sec（f32），
//!   epoch 秒 1.7e9 在 f32 下分辨率只有 ±128 秒，而「之后 10 秒窗口」「成交时长」这些量
//!   全都要精确到秒。
//!
//! · **价格精度**：同一处的 price/volume/turnover 也是 f32。impact1 / impact2 /
//!   price_radiation 对价格极敏感，实测只把 price 换成 f32，000001@20250303 的
//!   impact1_diff 就有 20660/28965 行相对误差超过 1e-4。
//!   所以本文件自己按 f64 读 CSV，并走 order_pair_metrics_pipeline 的 *_f64_inner。
//!
//! · **特征清单不同**：库里的 features（get_features_factors_rust_full，对应 Python 的
//!   with_lyapunov_exponent=False）关闭了 lyapunov，同时又多了 curvature / quad_coef
//!   两项（每列 2 个），所以每份不是旧代码的 1914 而是 1958，两份合计 3916。
//!   实测 `rp.verify_get_features_factors_rust` 在 44 列上返回 1958 个名字，
//!   前 21 组与 Python 的 get_features_names(with_lyapunov_exponent=False) 逐项一致，
//!   只是中间插了 curvature / quad_coef 两组。入选的 67 个名字一个都没落在
//!   lyapunov / curvature / quad_coef 上，所以按 3916 这套清单重算了 IDX 即可，
//!   值不受影响（下标由 `rp.verify_get_features_factors_rust` 返回的名字直接定位，
//!   不是估出来的）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、下午平移与窗口（对齐 pandas 的 between_time）
//! ════════════════════════════════════════════════════════════════════
//! 旧 adjust_afternoon：set_index('exchtime').between_time(start, end)，
//! 上午 [09:30:00, 11:30:00]、下午 [13:00:00, 14:57:00]，**微秒级闭区间**，
//! 下午部分整体前移 90 分钟，最后 concat（上午全部在前、下午全部在后）。
//! 顺序有意义：order_pair 内部按时间排序用的是稳定排序，同秒的行保持输入顺序。
//! 本文件按 `exchtime 微秒 + 8 小时` 的当日偏移（rem_euclid(一天)）判窗口，微秒精度，
//! 不是按整秒截断。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、名字与重名
//! ════════════════════════════════════════════════════════════════════
//! names() 返回旧 go() 的输出顺序（= 旧文件 names_save 去掉 _fold 后的 67 项）。
//! 其中 v1_price_ratio_diff_mean_below_p10 出现两次（一项来自自己的 _fold 版本），
//! 按规范第二次加 _dup2 后缀，值不变、列数不变。
//!
//! 已知差异（不影响入选的 67 个）：market_fractal_dim_diff / abs_market_fractal_dim_diff
//! 两列在新旧两个库之间就有实现差异（000001@20250303 上 28965 行里 28633 行不同），
//! 入选名单里没有任何 fractal 相关的量，也没有以它为自变量的相关系数项。
//!
//! 价格/量/额一律按 f64 解析（自己读 CSV，不走 fast_csv_reader 的 f32）：
//! impact1 / impact2 / price_radiation 对价格精度极敏感，用 f32 会有一批值过不了 1e-4。

use crate::data_paths;
use crate::features_f64;
use crate::order_pair_metrics_pipeline;
use ndarray::Array2;
use std::sync::LazyLock;

/// 旧 go 的 v1/v2 两份特征总长（各 1914，含 lyapunov）。只作对照记录，不参与计算。
#[allow(dead_code)]
const LEGACY_FULL: usize = 3828;
/// 库里 features 的实际输出长度：v1/v2 两份各 1958，合计 3916。
const FULL: usize = 3916;
/// 每份特征的长度（关了 lyapunov、多了 curvature/quad_coef）。
const HALF: usize = 1958;

const TOLERANCE_V1: f64 = 0.001;
const TOLERANCE_V2: f64 = 0.00001;

/// 一天的微秒数。
const DAY_US: i64 = 86_400_000_000;
/// 下午时段前移的微秒数。
const AFTERNOON_SHIFT_US: i64 = 90 * 60 * 1_000_000;
const MORNING_START_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MORNING_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const AFTERNOON_START_US: i64 = (13 * 3600) * 1_000_000;
const AFTERNOON_END_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;

// NAMES 是旧 go() 的返回顺序；IDX 是每一项在 FULL（3916）向量里的下标。
// 3916 = [v1 的 1958 项] + [v2 的 1958 项]。1958 那套清单来自
// `rp.verify_get_features_factors_rust(44 列矩阵, base_cols)` 返回的名字，
// 与 worker 里 get_features_factors_rust_full(.., true) 用的是同一条代码路径。
// base_cols 就是旧文件 go() 里的那 44 列，顺序与 order_pair 返回的列名一致。
const NAMES: [&str; 67] = [
    "v1_abs_avg_trade_amount_diff_corr_trade_count",
    "v1_abs_avg_trade_amount_diff_p5",
    "v1_abs_buy_order_trend_diff_corr_abs_sell_order_trend_diff",
    "v1_abs_impact1_diff_corr_market_interval_cv_diff",
    "v1_abs_impact1_diff_mean",
    "v1_abs_impact2_diff_corr_abs_price_radiation_diff",
    "v1_abs_impact2_diff_std",
    "v1_abs_market_hurst_exp_diff_cv",
    "v1_abs_market_hurst_exp_diff_mean_above_p90",
    "v1_abs_market_hurst_exp_diff_p95",
    "v1_abs_opposite_direction_trend_diff_cv",
    "v1_abs_price_radiation_diff_corr_market_interval_cv_diff",
    "v1_abs_same_direction_trend_diff_corr_abs_sell_order_trend_diff",
    "v1_abs_same_direction_trend_diff_corr_trade_count",
    "v1_abs_signed_turnover_diff_p25",
    "v1_abs_turnover_diff_std",
    "v1_avg_trade_amount_diff_corr_market_lz_complexity_diff",
    "v1_buy_ratio_stability_median",
    "v1_impact1_diff_corr_price_radiation_diff",
    "v1_impact1_diff_trend",
    "v1_market_hurst_exp_diff_mean_below_p10",
    "v1_market_interval_cv_diff_kurt",
    "v1_opposite_direction_trend_diff_corr_buy_order_trend_diff",
    "v1_opposite_direction_trend_diff_corr_market_interval_cv_diff",
    "v1_opposite_direction_trend_diff_mean",
    "v1_opposite_direction_trend_diff_mean_above_p90",
    "v1_price_radiation_diff_corr_abs_impact2_diff",
    "v1_price_radiation_diff_corr_market_hurst_exp_diff",
    "v1_price_radiation_diff_corr_market_interval_cv_diff",
    "v1_price_ratio_diff_corr_avg_trade_amount_diff",
    "v1_price_ratio_diff_corr_price_radiation_diff",
    "v1_price_ratio_diff_corr_same_direction_trend_diff",
    "v1_price_ratio_diff_mean_below_p10",
    "v1_price_ratio_diff_mean_below_p10_dup2",
    "v1_price_ratio_diff_p5",
    "v1_same_direction_trend_diff_corr_trade_count",
    "v1_same_direction_trend_diff_mean_above_p90",
    "v1_today_order_id_corr_abs_avg_trade_amount_diff",
    "v1_today_order_id_entropy_1d",
    "v1_today_order_id_p25",
    "v1_today_order_id_skew",
    "v1_trade_count_mean",
    "v1_trade_duration_diff_max_range_product",
    "v1_turnover_diff_corr_trade_count",
    "v2_abs_avg_trade_amount_diff_autocorr1",
    "v2_abs_avg_trade_amount_diff_corr_abs_price_radiation_diff",
    "v2_abs_counterparty_order_diff_entropy_1d",
    "v2_abs_impact2_diff_mean_above_p90",
    "v2_abs_opposite_direction_trend_diff_corr_abs_buy_order_trend_diff",
    "v2_abs_price_ratio_diff_p25",
    "v2_abs_same_direction_trend_diff_corr_abs_sell_order_trend_diff",
    "v2_abs_signed_turnover_diff_corr_abs_trade_duration_diff",
    "v2_abs_signed_turnover_diff_p25",
    "v2_abs_signed_turnover_diff_p5",
    "v2_abs_signed_turnover_diff_trend",
    "v2_avg_trade_amount_diff_median",
    "v2_avg_trade_amount_diff_p25",
    "v2_market_hurst_exp_diff_kurt",
    "v2_opposite_direction_trend_diff_mean",
    "v2_price_radiation_diff_corr_abs_impact1_diff",
    "v2_price_radiation_diff_corr_market_interval_cv_diff",
    "v2_price_ratio_diff_corr_abs_avg_trade_amount_diff",
    "v2_price_ratio_diff_corr_abs_same_direction_trend_diff",
    "v2_price_ratio_diff_corr_avg_trade_amount_diff",
    "v2_today_order_id_autocorr1_abs",
    "v2_turnover_diff_corr_price_ratio_diff",
    "v2_turnover_diff_p25",
];

const IDX: [usize; 67] = [
    1654, 244, 1748, 1588, 21, 1599, 110, 480, 832, 392, 470, 1699, 1723, 1734, 283, 106, 1220,
    51, 1090, 577, 871, 213, 1391, 1413, 14, 806, 1308, 1321, 1323, 1049, 1052, 1054, 840, 840,
    224, 1390, 805, 903, 1870, 264, 132, 43, 1923, 1005, 2466, 3596, 3856, 2772, 3693, 2242,
    3681, 3489, 2241, 2197, 2549, 2010, 2230, 2169, 1972, 3265, 3281, 3023, 3028, 3007, 2486,
    2924, 2224,
];

pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}

/// 交易日历（升序），只读一次。
static TRADING_DAYS: LazyLock<Vec<i64>> = LazyLock::new(|| {
    let path = data_paths::basic_info_path("").join("calendar.csv");
    std::fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter_map(|x| x.trim().parse::<i64>().ok())
                .collect()
        })
        .unwrap_or_default()
});

/// 上一个交易日（严格小于 date 的最大交易日）。
fn last_trading_day(date: i64) -> Option<i64> {
    let days = TRADING_DAYS.as_slice();
    if days.is_empty() {
        return None;
    }
    let idx = days.partition_point(|&d| d < date);
    if idx == 0 {
        None
    } else {
        Some(days[idx - 1])
    }
}

/// 读一个交易日的逐笔，复刻旧框架的 `read_trade` + `adjust_afternoon`。
///
/// 返回 (n, 7) 的 f64 矩阵，列序与旧 hm90.go 的 `prepare()` 一致：
///   `[整数秒, price, volume, turnover, flag, bid_order, ask_order]`
/// 行序 = 上午全部（文件序）在前、下午全部（文件序）在后，与 pandas 的 concat 一致
/// （同秒的行靠输入顺序决定名次，order_pair 用的是稳定排序，所以行序必须对上）。
///
/// 三件事都不借 fast_csv_reader，原因各不同：
///  · 撤单：旧代码是 `df[df.flag != 32]` 的数值比较，库里那条是按字节比 b"32"，
///    遇到 "32.0" 不生效。这里按解析出来的整数比。
///  · 窗口：pandas 的 between_time 是**微秒级闭区间**，按整秒截断会把边界那一秒多算进来。
///  · 精度：price/volume/turnover 必须是 f64。impact1 / impact2 / price_radiation
///    对价格极敏感，实测只把 price 换成 f32，000001@20250303 的 impact1_diff
///    就有 20660/28965 行相对误差超过 1e-4。
fn read_day(code: &str, date: i64) -> Option<Array2<f64>> {
    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;
    const EXCHANGE_OFFSET_US: i64 = 8 * 3600 * 1_000_000;

    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in data_paths::level2_roots() {
        let path = root
            .join(date.to_string())
            .join("transaction")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    let mut morning: Vec<[f64; 7]> = Vec::new();
    let mut afternoon: Vec<[f64; 7]> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue; // 表头 / 空行
        }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < 15 {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < 15 {
            fields[commas] = &line[start..];
        }
        if commas + 1 < 15 {
            continue; // 列数不够，pandas 也读不出这一行
        }

        let exchtime_us = us_prefix(fields[COL_EXCHTIME]);
        if exchtime_us == 0 {
            continue;
        }
        let flag = int_prefix(fields[COL_FLAG]) as i32;
        if flag == 32 {
            continue;
        }
        let total_us = exchtime_us + EXCHANGE_OFFSET_US;
        let tod = total_us.rem_euclid(DAY_US);
        let mut row = [
            0.0f64,
            num(fields[COL_PRICE]),
            num(fields[COL_VOLUME]),
            num(fields[COL_TURNOVER]),
            flag as f64,
            int_prefix(fields[COL_BID_ORDER]) as f64,
            int_prefix(fields[COL_ASK_ORDER]) as f64,
        ];
        if (MORNING_START_US..=MORNING_END_US).contains(&tod) {
            row[0] = total_us.div_euclid(1_000_000) as f64;
            morning.push(row);
        } else if (AFTERNOON_START_US..=AFTERNOON_END_US).contains(&tod) {
            row[0] = (total_us - AFTERNOON_SHIFT_US).div_euclid(1_000_000) as f64;
            afternoon.push(row);
        }
    }
    morning.append(&mut afternoon);
    let n = morning.len();
    let mut out = Array2::<f64>::zeros((n, 7));
    for (i, row) in morning.iter().enumerate() {
        for j in 0..7 {
            out[[i, j]] = row[j];
        }
    }
    Some(out)
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

/// 取字段的整数前缀。`"32.0"` 给 32，与旧框架 `df.flag != 32` 的数值比较一致。
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
/// 真遇到科学计数法写法就退回浮点解析。
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

/// 小数转 f64，照抄 pandas 的 C 解析器（见下面 parse_f64_pandas 的说明）。
/// 空字段给 NaN（pandas 也把空字段读成 NaN）。
fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
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
/// 与 src/okay_database/hm033.rs 里的同名函数一致，这里复制一份免得跨系列互相依赖。
///
/// ！！这不是「正确舍入」的十进制转二进制：它只把**前 17 位有效数字**
/// 用 `number = number*10 + digit` 累加进 double，多出来的整数位转成指数，
/// 多出来的小数位直接丢掉，最后用缓存好的 10 的幂做**一次**乘或除。
/// 实测 000001@20250303 有 260/76488 个 turnover 与 Rust `str::parse` 差 1 ulp。
/// 这 1 ulp 看着无所谓，实测会让 6 个配对行的 impact1/impact2 翻面，
/// 把入选因子里的 4 项推到 1.7e-4~7.2e-4（换回 pandas 的解析值后差异归零），所以必须照抄。
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

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let prev = last_trading_day(date)?;
    let t2 = read_day(code, date)?;
    let t1 = read_day(code, prev)?;
    if t2.nrows() == 0 || t1.nrows() == 0 {
        // 旧代码遇到空表会直接崩在 .first().unwrap()，效果就是这一天算不出来。
        return None;
    }

    let (result1, cols1) =
        order_pair_metrics_pipeline::calculate_order_pair_metrics_more_f64_inner(
            t2.view(),
            t1.view(),
            TOLERANCE_V1,
        );
    let (result2, cols2) =
        order_pair_metrics_pipeline::calculate_order_pair_metrics_more_v2_f64_inner(
            t2.view(),
            t1.view(),
            TOLERANCE_V2,
        );

    // 走 f64 的特征通路（src/features_f64.rs，src/features.rs 的孪生实现）。
    // src/features.rs 内部是 f32，实测 44 列上 1958 项里 244 项相对误差 > 1e-4、最大 18%，
    // 入选的 67 项里也有 9 项超（最大 2.8e-4），所以这里不能用它。
    // 空配对矩阵给半量 NaN，保证总长度恒定。
    let (vals1, _) = if result1.nrows() == 0 {
        (vec![f64::NAN; HALF], Vec::new())
    } else {
        features_f64::get_features_factors_rust_full(&result1.view(), &cols1, true)
    };
    let (vals2, _) = if result2.nrows() == 0 {
        (vec![f64::NAN; HALF], Vec::new())
    } else {
        features_f64::get_features_factors_rust_full(&result2.view(), &cols2, true)
    };

    let mut full = vals1;
    full.extend(vals2);
    if full.len() != FULL {
        // 长度不是预期的两份 1870：宁可整条给 NaN，也不要错位取值。
        return None;
    }
    // 流水线对外是 f32（TaskResult.facs 是 f32），只在最后一步降精度。
    Some(IDX.iter().map(|&i| full[i] as f32).collect())
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
